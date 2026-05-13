# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from scene import Scene
from scene.classifier import Classifier, TrajectoryGRU
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_seperate1, render_seperate, render_feature, render_probs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import imageio
from utils.system_utils import searchForMaxIteration
from utils.generate_sam import seg_anything
import concurrent.futures
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="imageio")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import random

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

color = [np.array([0.9, 0.9, 0.9, 0.6]), np.array([0,0.5,0.1,0.8]), np.array([0, 0, 0, 0.6]),]

def show_mask(mask, image, obj_id=1, random_color = False):
    image = np.array(image)
    if image.shape[-1] == 3:
        image = np.concatenate([image, np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)  # 添加 Alpha 通道
        
    h, w = mask.shape[-2:]
    mask_color = (mask.reshape(h, w, 1).cpu().numpy() * color[obj_id].reshape(1, 1, -1) * 255).astype(np.uint8)  # 乘 255 变成 [0, 255] 范围

    # 叠加 mask 到原始图像
    alpha = mask_color[..., 3:] / 255.0  # 提取 mask 的透明度通道
    image = (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)  # alpha 混合

    return image

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

def visualize_and_save(pred_obj, probs, pred_dy, probs_dy, pred_obj_all, probs_all, save_path):
    imgs = [
        pred_obj.detach().cpu().numpy(), 
        probs, 
        pred_dy.detach().cpu().numpy(), 
        probs_dy, 
        pred_obj_all.detach().cpu().numpy(), 
        probs_all
    ]
    cmaps = ['gray', 'viridis'] * 3
    titles = ['mask_st', 'probs_st', 'mask_dy', 'probs_dy', 'mask', 'probs_all']

    plt.figure(figsize=(18, 4))
    for i, (img, cmap, title) in enumerate(zip(imgs, cmaps, titles)):
        ax = plt.subplot(1, 6, i+1)
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if cmap == 'viridis':
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def cosine_similarity(f1, f2):
    f1 = f1.flatten(1)
    f2 = f2.flatten(1)
    return F.cosine_similarity(f1, f2, dim=1).mean()

def erode_mask(mask: torch.Tensor, kernel_size=3, iterations=1):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1).float()  # (N,1,H,W)
    else:
        mask = mask.float()

    # 卷积核，全1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)

    for _ in range(iterations):
        # 用padding保证输出大小不变
        # 进行腐蚀：卷积结果等于kernel元素总数说明该区域全是1
        conv = F.conv2d(mask, kernel, padding=kernel_size//2)
        mask = (conv == kernel.numel()).float()

    return mask.squeeze(1)  # (N,H,W) or (H,W)

from scipy.ndimage import label, sum as ndi_sum
def remove_small_components(mask: torch.Tensor, min_area: int = 50):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    labeled, num_features = label(mask_np)  # 获取连通区域
    sizes = ndi_sum(mask_np, labeled, index=np.arange(1, num_features + 1))

    # 创建保留区域的 mask
    clean_mask_np = np.zeros_like(mask_np)
    for i, size in enumerate(sizes):
        if size >= min_area:
            clean_mask_np[labeled == (i + 1)] = 1

    return torch.from_numpy(clean_mask_np).to(mask.device)


def intersect_masks(conf: torch.Tensor, sam_masks: torch.Tensor, intersect_threshold: float):
    
    coarse_mask = conf.unsqueeze(-1).float()  # [H, W, 1]

    sam_masks_flat = sam_masks.view(-1)  # [H*W]
    obj_ids = torch.unique(sam_masks_flat)
    N = obj_ids.shape[0]
    H, W = sam_masks.shape
    
    fine_masks_flat = (sam_masks_flat[None, :] == obj_ids[:, None]).float()  # [N_obj, H*W]
    fine_masks = fine_masks_flat.view(N, H, W, 1)  # [N_obj, H, W, 1]

    intersection = (fine_masks * coarse_mask).sum(dim=(1, 2, 3))  # [N]
    area = fine_masks.sum(dim=(1, 2, 3)) + 1e-6
    iou = intersection / area  # [N]

    selected = iou >= intersect_threshold  # [N]
    if selected.sum() == 0:
        final_mask = torch.zeros_like(coarse_mask)
    else:
        final_mask = (fine_masks[selected].sum(dim=0) >= 0.5).to(coarse_mask.dtype)

    return final_mask


def fill_gap(mask: torch.Tensor, kernel_size: int):
    """
    args:
        mask with gaps: (height, width, 1)
    return:
        mask without gaps: (height, width, 1)
    """
    height, width = mask.shape[:2]
    if kernel_size%2==0: kernel_size += 1
    pad_size = kernel_size // 2

    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask) / (kernel_size**2)
    mask_padded = F.pad(mask.reshape(1, 1, height, width), (pad_size, )*4, mode='reflect')
    mask_smooth = F.conv2d(mask_padded, kernel, padding='valid').reshape(mask.shape)

    return ((mask_smooth + mask) >= 0.5).to(mask)


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, cam_type, dataset_source_path, gru):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    
    mask_path = os.path.join(dataset_source_path, "final_mask")
    makedirs(mask_path, exist_ok=True)

    
    render_images, gt_list, pred_obj_mask_list = [], [], [], 

    with torch.no_grad():
        trajectory_list = []


        for tsp in range(0, len(views), len(views) // 4):   # selected_indices_in_all
            time = torch.tensor(views[tsp].time).cuda().repeat(gaussians._xyz.shape[0], 1)
            *_, dx, _, _, _ = gaussians._deformation(gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time)
            id = classifier._mlp_tjy(dx.detach())
            trajectory_list.append(id)
    

        tmp = []+trajectory_list       
        identidy_encoing = classifier._mlp(gaussians.get_xyz.detach())   # , hidden
        tmp.append(identidy_encoing)

        trajectory_feats = torch.stack(tmp)
        hidden = gru(trajectory_feats)

        logits3d = classifier._classifier(hidden.unsqueeze(1).permute(2, 0, 1))
        category = torch.sigmoid(logits3d).squeeze(0)  # 仍然是 torch.Size([2, 134541, 1])
        pred_obj = (category > 0.5) # 到这里classifier都是没有梯度的
        
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt  = to8b(view.original_image).transpose(1,2,0)
        gt_list.append(gt)
        rendering_curr = view.original_image.cuda() # output["render"]
        render_images.append(to8b(rendering_curr).transpose(1,2,0))
        
        
        render_pkg_prob = render_probs(view, gaussians, pipeline, bg_color = torch.tensor([1, 0, 0], dtype=torch.float32, device="cuda"), classifier = classifier, seperate_mask = pred_obj.squeeze(), category = category)
        rendered_prob = render_pkg_prob["rendered_prob"]  
        rendered_prob = rendered_prob.clamp(0.0, 1.0)
        conf = rendered_prob[1]>0.5
        
        sam_masks = torch.from_numpy(seg_anything(view, dataset_source_path, idx).astype(np.int32))

        if conf.squeeze(0).shape != sam_masks.shape:
            sam_masks = F.interpolate(
                            sam_masks.unsqueeze(0).unsqueeze(0).float(),
                            size=conf.shape[-2:],
                            mode='nearest'
                        ).squeeze().long()  # 最终转换成整数ID

        residual_mask_upper_sam = intersect_masks(conf.squeeze(0), sam_masks.cuda(), 0.75)  # 0.3 越高 越bu容易包含东西   chicken 0.4 0.5 0.5 前期可以多一些噪声   0.3 torch

        robust_zero_mask = fill_gap(residual_mask_upper_sam, 5)
        
        # [torchvision.utils.save_image(mask.permute(2, 0, 1), f"residual_mask_{i:02d}.png") for i, mask in enumerate(residual_list)]

        mask_to_save = robust_zero_mask.permute(2, 0, 1).float()  # (1, 960, 536)
        save_image(mask_to_save, os.path.join(mask_path, f'{idx:06d}.png'))

        img_with_mask_rendered = show_mask(mask_to_save, gt.copy())
        pred_obj_mask_list.append(img_with_mask_rendered)

    imageio.mimwrite(os.path.join(render_path, 'video_mask.mp4'), pred_obj_mask_list, fps=30)

def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = False
        dataset.eval = False
        num_classes = 2
        print("Num classes: ",num_classes)

        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        classifier = Classifier(hyperparam, dataset.feature_dim, num_classes)
        gru = TrajectoryGRU()
        cam_type = scene.dataset_type
        
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        gru_params = torch.load(os.path.join(load_path,"gru.pth"))
        
        gaussians.restore(gaussians_params, opt, stage)
        classifier.restore(classifier_params, opt)
        gru.restore(gru_params)

        # 是否Load .pth文件
        # checkpoint = './output/hypernerf/chicken-loss_all_woobj3d/chkpnt_coarse_3000.pth'
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt, stage = 'coarse')    # coarse 不包含 load_pose

       
        background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
        # background = [0, 0, 0]

        
        render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type, dataset.source_path, gru)
        # if (not skip_test) and (len(scene.getTestCameras()) > 0):
        #     render_set(dataset.model_path, "test", loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, cam_type)
        # if not skip_video:
        #     render_set(dataset.model_path, "video", loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, classifier, cam_type)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true", default=True)
    parser.add_argument("--configs", type=str, default = "arguments/hypernerf/default.py")
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--cam_view", type=str, default='cam16')
    parser.add_argument("--num_classes", type=int, default=2)
    cmdlne_string = ['--model_path', './output/hypernerf/hand1/']    # 也决定了读哪个数据集
    args = get_combined_args(parser, target_cfg="scene", cmdlne_string = cmdlne_string)
    
    
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode, args.cam_view)