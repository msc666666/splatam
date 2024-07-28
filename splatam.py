import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):             #选择合适的数据集
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,             
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):          #定义一个函数，从彩色和深度图像中生成一个点云，可以进行点云变换，遮罩选择和均方距离计算，函数中包含各种参数
    width, height = color.shape[2], color.shape[1]             #核心步骤：计算像素索引和深度
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)            #初始化点云
    if transform_pts:                  #变换点云
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:            #计算均方距离
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud          #给点云着色
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask       #基于遮罩选择点
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:       #返回结果
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):          #这个函数用于初始化参数和变量来进行3D高斯分布的密集化处理
    num_pts = init_pt_cld.shape[0]         #首先初始化参数
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]       #means3D是每个高斯的3D均值
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]         #未归一化的旋转四元数
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")         #初始化所有高斯的不透明度为0
    if gaussian_distribution == "isotropic":                   #log_scales是高斯分布的对数尺度；；；；对于各向同性(isotropic)的高斯分布，尺度是均匀的；对于各向异性(anisotropic)的高斯分布，尺度是不同方向上的。
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {               #创建参数字典
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))              
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots                 #初始化相机的未归一化旋转四元数
    params['cam_trans'] = np.zeros((1, 3, num_frames))      #初始化相机的平移向量

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))          #将所有参数转换为torch.nn.Parameter类型，并设置requires_grad是true，来启用梯度计算

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}                  #variable中包含了其他辅助变量的字典

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")          #打印配置信息
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False       #检查配置信息，不存在设置为false
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")           #打印加载完成后的配置信息

    # Create Output Directories   #创建一个输出目录
    output_dir = os.path.join(config["workdir"], config["run_name"])       #第一个参数是config中指定的工作目录路径；第二个参数是运行的唯一名称
    eval_dir = os.path.join(output_dir, "eval")          #eval_dir是为评估目录
    os.makedirs(eval_dir, exist_ok=True)          #创建eval_dir目录，防止引发目录重建的错误。
    
    # Init WandB           #Wandb（Weights and Biases），用于机器学习实验跟踪和可视化的工具
    if config['use_wandb']:     #检查是否使用Wandb
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0      #三个变量用于初始化步骤计数器
        wandb_run = wandb.init(project=config['wandb']['project'],       #初始化一个新的Wandb运行，project指定项目名称；entity指定Wandb实体和团队名称；group指定运行组名称，对于同时运行多个相关实验有用；name指定运行的名称；config将整个配置传给Wandb，用于记录所有实验参数。
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])                  #用于确定在pytorch中使用的设备。

    # Load Dataset           #下载数据
    print("Loading Dataset ...")
    dataset_config = config["data"]          #获取数据集配置
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])         #看dataset_config中没有gradslam_data_cfg键，就创建一个新的字典，设置dataset_name ;如果存在gradslam_data_cfg，则调用dataset_config加载配置
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False               #用于处理不同分辨率的图像
    # Poses are relative to the first frame          #数据集中的姿势是相对于第一帧
    dataset = get_dataset(                       #获取数据集
        config_dict=gradslam_data_cfg,          #config_dict是之前处理过的gradslam_data_cfg
        basedir=dataset_config["basedir"],       #数据集的基础目录
        sequence=os.path.basename(dataset_config["sequence"]),  #sequence是数据集序列的名称，通过dataset_config来获取文件名
        start=dataset_config["start"],
        end=dataset_config["end"],         #这两句是定义要加载帧的开始和结束的位置
        stride=dataset_config["stride"],      #指定加载的帧的步长
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],     #指定的期望的图像的高度和宽度
        device=device,          #之前设置的pytorch设备
        relative_pose=True,      #表示姿势是相对于第一个帧
        ignore_bad=dataset_config["ignore_bad"],          #指定是否忽略坏的帧
        use_train_split=dataset_config["use_train_split"],     #指定是否使用训练集分割
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)      #如果为-1(列表倒数第一个序号时-1)，那么num_frames设置为原始数据集的长度，也就是帧的数量

    # Init seperate dataloader for densification if required    #初始化一个单独的数据加载器用于数据集的稠密化
    if seperate_densification_res:     #检查是否需要稠密化数据，属于布尔型数据
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,        #传入定义好的gradslam_data_cfg配置字典
            basedir=dataset_config["basedir"],    #基础目录，指定数据存储的位置
            sequence=os.path.basename(dataset_config["sequence"]),    #使用os.path.basename提取文件名
            start=dataset_config["start"],
            end=dataset_config["end"],          #指定从哪开始，到哪结束。
            stride=dataset_config["stride"],      #用于指定帧之间的间隔，中间隔几个帧
            desired_height=dataset_config["densification_image_height"],     #稠密化数据的期望得到的图像的高度
            desired_width=dataset_config["densification_image_width"],       #期待的图像的宽度。
            device=device,    #指定pytorch设备
            relative_pose=True,    #数据中的姿势都是相对于第一个帧
            ignore_bad=dataset_config["ignore_bad"],    #判断是否忽略坏帧，即无法读取或者损坏的图像
            use_train_split=dataset_config["use_train_split"],  #分割训练集和测试集
        )
        # Initialize Parameters, Canonical & Densification Camera parameters          #初始化参数，包含常规和稠密化相机的参数
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])             #基于不同条件初始化不同的参数集，配置相应的相机参数
    
    # Init seperate dataloader for tracking if required    #初始化一个用于跟踪的数据显示器
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,         #配置一个字典，用来设置数据集的参数
            basedir=dataset_config["basedir"],     #数据集的基本目录
            sequence=os.path.basename(dataset_config["sequence"]),       #数据集的序列名
            start=dataset_config["start"],
            end=dataset_config["end"],              
            stride=dataset_config["stride"],        #开始帧，结束帧，帧间隔
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],      #tracking图像的期望的高度和宽度
            device=device,         #加载设备
            relative_pose=True,       #使用相对姿态；绝对姿态指相机相对于固定参考系的姿态，相对姿态时相机i相对于另一个相机的姿态。
            ignore_bad=dataset_config["ignore_bad"],     #判断是否忽略坏数据
            use_train_split=dataset_config["use_train_split"],      #使用训练数据集划分
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]          #获取数据中的第一帧，返回包括图像颜色，相机内参
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)        #
        tracking_intrinsics = tracking_intrinsics[:3, :3]       #取得相机内参的前三行和前三列
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())      #setup_camera用于创建和配置相机对象的函数，宽度为2，高度为1，把相机内参转换为numpy数组
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []          #用于存储关键帧
    keyframe_time_indices = []       #用于存储关键帧的时间索引
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []    #用于存储所有帧的真实姿态
    tracking_iter_time_sum = 0   #跟踪迭代的总运行时间
    tracking_iter_time_count = 0   #跟踪迭代次数的计数器
    mapping_iter_time_sum = 0       #建图的总运行时间
    mapping_iter_time_count = 0     #建图迭代次数的计数器
    tracking_frame_time_sum = 0      #跟踪每帧的总运行时间之和
    tracking_frame_time_count = 0    #跟踪处理帧的计数器
    mapping_frame_time_sum = 0       #建图的每帧的运行时间之和
    mapping_frame_time_count = 0     #建图的处理帧的计数器

    # Load Checkpoint             #加载模型训练的检查点
    if config['load_checkpoint']:  #判断是否需要加载检查点
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")      #获取检查点的时间索引checkpoint_time_idx，并打印出来
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")    #ckpt_path：构建检查点文件的路径；np.load：加载.npz文件中的参数
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}    #将参数转换为pytorch张量，移动到gpu上，同时设置为需要梯度计算requires_grad_(True)
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()       #初始化用于计算的变量
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()          #从.npy文件中加载关键帧时间索引，并转换为列表
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)         #遍历从开始到检查点时间索引的每一帧
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}       #创建一个包含当前帧信息的字典 curr_keyframe 并将其添加到 keyframe_list 中
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)        #初始化关键帧信息
    else:
        checkpoint_time_idx = 0         #不加检查点，将checkpoint_time_idx设置为0
    
    # Iterate over Scan          #遍历扫描数据
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):      #遍历每一帧
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]        #一帧一帧的加载数据
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)         #计算地面真实姿态的逆矩阵gt_w2c，用于将相机坐标系中的点转换到世界坐标系中
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255         #将（H，W，C）转换成（C，H，W），并且将像素值归一化到[0,1]
        depth = depth.permute(2, 0, 1)               #将深度图像的维度从(H,W,D)转换为（D，H，W）
        gt_w2c_all_frames.append(gt_w2c)             #用于更新地面真实姿态列表
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking   #优化当前时间步
        iter_time_idx = time_idx           #将iter_time_idx设置为time_idx
        # Initialize Mapping Data for selected frame       #初始化当前帧的映射数据
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}    #构建一个字典
        
        # Initialize Data for Tracking    #初始化跟踪数据
        if seperate_tracking_res:       #检查是否需要单独的跟踪数据集；使用单独的跟踪数据集是为了确保跟踪任务中使用最合适的分辨率，提高精度和速度
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255      #调整色彩维度顺序，将像素值归一化
            tracking_depth = tracking_depth.permute(2, 0, 1)               #调整深度图像的维度顺序
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}    #构建当前跟踪帧的数据字典
        else:
            tracking_curr_data = curr_data      #不用单独的跟踪数据集，就使用通用的数据字典

        # Optimization Iterations    #优化迭代次数
        num_iters_mapping = config['mapping']['num_iters']   
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:   #第一帧的相机姿态通常假设为已知的参考姿态
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking     #下面的函数进行跟踪过程的优化迭代
        tracking_start_time = time.time()   #开始计时
        if time_idx > 0 and not config['tracking']['use_gt_poses']:    #在不使用真实姿态时跟踪初始化
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)    #lrs是学习率，通过lrs初始化优化器，使每次跟踪迭代都从相同学习率开始
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)    #初始化最小损失
            # Tracking Optimization
            iter = 0     #迭代次数
            do_continue_slam = False    #是否继续slam的标志
            num_iters_tracking = config['tracking']['num_iters']    #tracking的总迭代次数
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)              #调用get_loss函数，计算当前帧的损失值和相关变量
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)      #使用wandb就报告当前损失
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)         #通过反向传播计算梯度，并使用优化器更新参数，并且将梯度清零
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress     #报告进度
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers    #更新运行时间
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1    #记录每次迭代的运行时间
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})          #检查当前迭代次数是否达到预设的最大迭代次数，满足则停止，不满足，增加迭代次数
                    else:
                        break

            progress_bar.close()  #完成当前帧的跟踪后，关闭进度条
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():             #更新最佳旋转和位移到参数字典中
                # Get the ground truth pose relative to frame 0             #如果指定使用真实姿态，则将真实姿态转换为四位数和位移，并更新参数字典
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters   更新相机参数
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()      #计算跟踪结束时间
        tracking_frame_time_sum += tracking_end_time - tracking_start_time     #更新跟踪帧的时间总和
        tracking_frame_time_count += 1          #更新跟踪帧的次数
 
        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:      #如果满足条件，就进入报告进度的过程
            try:
                # Report Final Tracking Progress   #报告跟踪进度
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():    #禁用梯度计算，提高评估效率
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)       #报告进度的方式不同
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')          #异常处理，保存参数检查点，打印错误消息

        # Densification & KeyFrame-based Mapping        #稠密化和关键帧映射操作
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:en
                    # Load RGBD frames incremtally instead of all frames        #增量加载帧
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]     #加载当前时间步的RGD-D帧
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)    #和前边一样，改变维度，对颜色进行归一化
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data     #不单独使用稠密化数据集，就直接使用当前帧

                # Add new Gaussians to the scene based on the Silhouette  #在稠密化过程中基于当前帧的轮廓向场景中添加新的高斯点
                params, variables = add_new_gaussians(params, variables, densify_curr_data,           #arams是模型的参数；variables是存储临时变量；densify_curr_data是当前帧的稠密化数据
                                                      config['mapping']['sil_thres'], time_idx,      #前边这个是轮廓阈值，用来确定在什么地方添加高斯点；后边这个是
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])     #前边是均方距离方法，用于计算新的高斯点的位置；后边是高斯分布的配置
                post_num_pts = params['means3D'].shape[0]       #计算新的高斯点的数量，params['means3D']存储所有高斯点的位置信息，通过获取其形状的第一个维度，可以得到当前高斯点的总数
                if config['use_wandb']:       #日志记录
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():     #禁用梯度计算
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())               #params['cam_unnorm_rots'][..., time_idx]：获取当前时间步未归一化的旋转向量，使用F.normalize进行归一化
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()           #获取平移向量
                curr_w2c = torch.eye(4).cuda().float()      #创建一个单位矩阵
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)      #构建旋转矩阵
                curr_w2c[:3, 3] = curr_cam_tran           #设置平移向量
                # Select Keyframes for Mapping      #选择关键帧
                num_keyframes = config['mapping_window_size']-2      #确定关键帧的数量
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)          #调用关键帧选择函数
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]                  #获取选中的关键帧时间索引，将关键帧索引转换为对应的时间索引
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)         #添加最后一个关键帧
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)     #将当前时间步添加到选中的关键帧列表里
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")          #打印选中的关键帧

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)            #重置优化器

            # Mapping
            mapping_start_time = time.time()          #初始化建图开始时间
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):        #迭代进行建图优化
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]       #随机选择一个帧的索引
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']               #索引为-1，就用当前帧；否则用关键帧
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]          #获取当前帧之前的所有地面真值位姿保存在iter_gt_w2c中
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}          #将以上数据打包为一个字典
                # Loss for current frame        #当前帧的损失值
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)      #调用损失计算函数
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop      #对当前损失值反向传播，计算梯度
                loss.backward()
                with torch.no_grad():    #在这个块中所有操作都不记录梯度
                    # Prune Gaussians   #修剪高斯分布
                    if config['mapping']['prune_gaussians']:   #检查是否需要修剪高斯分布
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])  #执行修剪；config['mapping']['pruning_dict']修剪策略配置
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})     #用wandb报告修剪后的高斯分布数量
                    # Gaussian-Splatting's Gradient-based Densification      #基于高斯分布的3D重建方法，通过梯度优化实现稠密化
                    if config['mapping']['use_gaussian_splatting_densification']:       # 检查配置文件中是否启用了基于高斯分布的稠密化方法
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])      # 调用 densify 函数，通过优化器 optimizer 和当前迭代次数 iter 对参数 params 和变量 variables 进行稠密化处理
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})     # 将稠密化后的高斯分布数量（params['means3D'].shape[0]）和当前的步骤（wandb_mapping_step）记录到 wandb 中
                    # Optimizer Update    #优化器更新
                    optimizer.step() 
                    optimizer.zero_grad(set_to_none=True)      #清零梯度
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)          #使用wandb报告进展
                # Update the runtime numbers         #更新运行时间
                iter_end_time = time.time()      # 
                mapping_iter_time_sum += iter_end_time - iter_start_time    #计算当前迭代的运行时间，并加到mapping_iter_time_sum上
                mapping_iter_time_count += 1     #增加count的次数
            if num_iters_mapping > 0:
                progress_bar.close()          #如果存在迭代次数，就关闭进度条，释放资源
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1       #作用同上，但是不知道为啥

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:      #判断是否需要报告全局进度
                try:       #捕获异常
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")         #初始化进度条
                    with torch.no_grad():      #禁用梯度计算
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:       #捕获异常
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])             #获取检查点路径
                    save_params_ckpt(params, ckpt_output_dir, time_idx)                 #保存当前参数的检查点
                    print('Failed to evaluate trajectory.')                 #打印错误消息
        
        # Add frame to keyframe list                #在列表中添加关键帧
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation获取当前帧的未归一化的旋转参数并归一化
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()         #获取当前帧的平移参数
                curr_w2c = torch.eye(4).cuda().float()     #构建当前帧的世界到相机的变换矩阵
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info       #初始化关键帧信息
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list              #将当前关键帧添加到关键帧列表
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration          #定期检查点保存和清理缓存
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:           #相隔checkpoint_interval帧，检查点保存一次
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step           #增加wandb时间步长
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()                      #清理gpu缓存

    # Compute Average Runtimes                 #计算和打印跟踪和映射过程的平均时间
    if tracking_iter_time_count == 0:      #计算平均时间
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")           #打印平均运行时间
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:             #记录到wandb中
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters            #在整个过程中评估最终的参数          用eval函数实现
    with torch.no_grad():          #使用无梯度上下文管理器
        if config['use_wandb']:          #检查wandb设置
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])              #调用各种评估函数
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them           #将相机参数添加到参数字典中，并将其保存到指定目录
    params['timestep'] = variables['timestep']      #添加时间步信息
    params['intrinsics'] = intrinsics.detach().cpu().numpy()      #相机内参
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()         #初始相机位姿
    params['org_width'] = dataset_config["desired_image_width"]    #图像原始宽度和高度
    params['org_height'] = dataset_config["desired_image_height"]    
    params['gt_w2c_all_frames'] = []    #所有帧的地面真实位姿 
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)   #关键帧时间索引
    
    # Save Parameters
    save_params(params, output_dir)         #保存参数

    # Close WandB Run      #结束wandb运行
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)