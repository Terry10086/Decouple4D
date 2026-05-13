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
from scene.classifier import Classifier
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature, render_seperate, render_feature
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

color = [np.array([0.9, 0.9, 0.9, 0.6]), np.array([1,0.5,0.1,0.4]), np.array([0, 0, 0, 0.6]),]

def show_mask(mask, image, obj_id=1, random_color = False):
    image = np.array(image)
    if image.shape[-1] == 3:
        image = np.concatenate([image, np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)  # 添加 Alpha 通道
        
    h, w = mask.shape[-2:]
    mask_color = (mask.reshape(h, w, 1) * color[obj_id].reshape(1, 1, -1) * 255).astype(np.uint8)  # 乘 255 变成 [0, 255] 范围

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

def intersect_masks(coarse_mask: torch.Tensor, 
                    fine_masks: torch.Tensor, 
                    intersect_threshold: float) -> torch.Tensor:
    """
    Given a coarse mask and a set of fine masks, return the fine mask based on the intersection.
    
    args:
        coarse_mask: (h, w, 1)
        masks: (n, h, w, 1)
    return:
        fine_mask: (h, w, 1)
    """
    selected = (
        torch.sum(coarse_mask[None, ...] * fine_masks, dim=(1,2,3)) / torch.sum(fine_masks, dim=(1,2,3))
    ) >= intersect_threshold # (n,)
    fine_mask = torch.zeros_like(coarse_mask) if selected.sum()==0 \
                else (fine_masks[selected].sum(dim=0) >= 0.5).to(coarse_mask.dtype)
    return fine_mask

def fill_gap(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
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

def cosine_similarity(f1, f2):
    f1 = f1.flatten(1)
    f2 = f2.flatten(1)
    return F.cosine_similarity(f1, f2, dim=1).mean()

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, cam_type, dataset_source_path):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    
    mask_path = os.path.join(dataset_source_path, "final_mask")
    makedirs(mask_path, exist_ok=True)


    render_images, gt_list = [], [],  

    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt  = to8b(view.original_image).transpose(1,2,0)
        gt_list.append(gt)
        output = render(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type)       # fine stage !!!
        rendering_curr = view.original_image.cuda() # output["render"]
        render_images.append(to8b(rendering_curr).transpose(1,2,0))
        
        # time = time=torch.tensor(view.time).cuda().repeat(gaussians.get_xyz.shape[0],1)
        # *_, hidden_currt = gaussians._deformation(gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time)
        # hidden_currt_norm = F.normalize(hidden_currt, dim=-1)

        # 1. 渲染每个视角在不同时刻的结果
        residual_list = []
        if False:
            diff_scores = []
            for tsp in range(1, 20):
                view_time = views[(view.uid + tsp * 5) % len(views)]
                time = time=torch.tensor(view_time.time).cuda().repeat(gaussians.get_xyz.shape[0],1)
                *_, hidden_before_mlp = gaussians._deformation(gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time)
                hidden_before_mlp_norm = F.normalize(hidden_before_mlp, dim=-1)
                similarity = F.cosine_similarity(hidden_currt_norm, hidden_before_mlp_norm, dim=-1).mean()
                dissimilarity = 1 - similarity  # 越大越不相似
                diff_scores.append((view_time.uid, dissimilarity.item()))

            k = 5
            topk = sorted(diff_scores, key=lambda x: x[1], reverse=True)[:k]

            for tsp in range(k):
                view_selected = views[topk[tsp][0]]
                output = render(view, gaussians, pipeline, background, stage="coarse", cam_type=cam_type, time = view_selected.time) 
                rendering_curr = TF.gaussian_blur(rendering_curr, kernel_size=5, sigma=1.0)
                residual = torch.abs(output["render"] - rendering_curr).mean(dim=0, keepdim=True).permute(1,2,0)
                residual_mask_base = (residual <= residual.mean()).to(residual)
                residual_mask_upper = (residual <= torch.quantile(residual, 0.95)).to(residual) # (h, w, 1)  筛掉残差最大的95% 遮挡、边界、光照剧烈变化
                # residual.mean() may be larger than quantile, so take the union
                residual_mask_upper = ((residual_mask_base + residual_mask_upper) >= 0.5).to(residual)
                residual_mask_upper_sam = intersect_masks(residual_mask_upper, sam_masks_list[idx].cuda().unsqueeze(-1), 0.85)  # 越高 越容易包含东西尽量
                residual_list.append(residual_mask_upper_sam)

            stacked_masks = torch.stack(residual_list, dim=0)
            zero_counts = (stacked_masks == 0).sum(dim=0)
            threshold = int(len(residual_list) * 0.8)
            robust_zero_mask = (zero_counts >= threshold).to(stacked_masks.dtype)
            robust_zero_mask = robust_zero_mask.view(stacked_masks.shape[1:])
            robust_zero_mask = fill_gap(robust_zero_mask, 5)
        
        # 3dgs
        else:
            output = render(view, gaussians, pipeline, background, stage="coarse", cam_type=cam_type) 
            rendering_curr = TF.gaussian_blur(rendering_curr, kernel_size=5, sigma=1.0)
            residual = torch.abs(output["render"] - rendering_curr).mean(dim=0, keepdim=True).permute(1,2,0)
            residual_mask_base = (residual <= residual.mean()).to(residual)
            residual_mask_upper = (residual <= torch.quantile(residual, 0.95)).to(residual) # (h, w, 1)  筛掉残差最大的95% 遮挡、边界、光照剧烈变化
            # residual.mean() may be larger than quantile, so take the union
            residual_mask_upper = ((residual_mask_base + residual_mask_upper) >= 0.5).to(residual)

            sam_masks_list = seg_anything(views, dataset_source_path, idx)
            residual_mask_upper_sam = intersect_masks(residual_mask_upper, sam_masks_list.cuda().unsqueeze(-1), 0.85)  # 越高 越容易包含东西尽量
            # residual_list.append(residual_mask_upper_sam)

            robust_zero_mask = fill_gap(residual_mask_upper_sam, 5)

        
        # [torchvision.utils.save_image(mask.permute(2, 0, 1), f"residual_mask_{i:02d}.png") for i, mask in enumerate(residual_list)]

        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)

        mask_to_save = robust_zero_mask.permute(2, 0, 1).float()  # (1, 960, 536)
        save_image(mask_to_save, os.path.join(mask_path, f'{idx:06d}.png'))

    imageio.mimwrite(os.path.join(render_path, 'video_gt.mp4'), gt_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_rgb.mp4'), render_images, fps=30)

    

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
        cam_type = scene.dataset_type
        
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        # print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        # gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        # classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        
        # gaussians.restore(gaussians_params, opt, stage)
        # classifier.restore(classifier_params, opt)

        # 是否Load .pth文件
        checkpoint = './output/hypernerf/chicken-loss_all_woobj3d/chkpnt_coarse_3000.pth'
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, stage = 'coarse')    # coarse 不包含 load_pose

       
        background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
        # background = [0, 0, 0]

        if not skip_train:
            render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type, dataset.source_path)
        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, cam_type)
        if not skip_video:
            render_set(dataset.model_path, "video", loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, classifier, cam_type)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str, default = "arguments/hypernerf/chicken.py")
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--cam_view", type=str, default='cam16')
    parser.add_argument("--num_classes", type=int, default=2)
    cmdlne_string = ['--model_path', './output/hypernerf/chicken-loss_all_woobj3d/']    # 也决定了读哪个数据集
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