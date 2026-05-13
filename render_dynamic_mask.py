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
from gaussian_renderer import render, render_contrastive_feature, render_seperate, render_feature,tracking
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
import concurrent.futures
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="imageio")
import matplotlib.pyplot as plt

def get_color(index, total):
    hue = index / total  # 从 0 到 1 均匀分布
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

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

color = [np.array([0.9, 0.9, 0.9, 0.6]), np.array([0.0, 0.0, 1.0, 0.6]), np.array([0, 0, 0, 0.6]),]

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

def save_numpy_mask(mask, path, count):
    filename = os.path.join(path, '{0:05d}.png'.format(count))
    imageio.imwrite(filename, (mask.astype(np.uint8) * 255))  # 转换为 0-255 并保存

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    dy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")
    st_path = os.path.join(model_path, name, "ours_{}".format(iteration), "static")
    colored_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "colored_mask")
    makedirs(dy_path, exist_ok=True)
    makedirs(st_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(colored_mask_path, exist_ok=True)

    gt_list, dynamic, static = [], [], []
    dynamic_mask_images, dynamic_mask_list, static_mask_image, static_mask_list = [], [], [], []
    track_list = []
    track_img_lis = []
    
    max_indices = None
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt  = to8b(view.original_image).transpose(1,2,0)
        gt_list.append(gt)

        # tracking
        track_pkg = tracking(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type, is_eval = True, max_indices=max_indices)

        max_indices = track_pkg['max_indices']
        points2d_max = track_pkg['points2d_max']
        rendered_dy = track_pkg['rendered_dy']

        track_img_lis.append(to8b(rendered_dy).transpose(1,2,0))

        pts = points2d_max.detach().cpu().numpy().astype(int)
        for i, pt in enumerate(pts):
            u, v = pt
            color = get_color(i, len(pts))
            cv2.circle(gt, (u, v), radius=4, color=color, thickness=-1)
        # 画红色圆点，半径为4
        cv2.circle(gt, (u, v), radius=4, color=(0, 0, 255), thickness=-1)
        track_list.append(gt)


        # render dynamic_mask
        # render_pkg = render(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type, is_eval = True)
        # dynamic_mask, static_mask, histogram_dy, histogram_st = render_pkg["dynamic_mask"], render_pkg["static_mask"], render_pkg["histogram_dy"], render_pkg["histogram_st"]
        

        # dynamic_mask_images.append(to8b(dynamic_mask).transpose(1,2,0))
        # static_mask_image.append(to8b(static_mask).transpose(1,2,0))
        

        # non_black_mask = dynamic_mask_images[-1][...,0] > 25

        # colored_mask_dy = show_mask(non_black_mask, gt, obj_id=1, random_color = False)
        # # imageio.imwrite(os.path.join(colored_mask_path, '{0:05d}'.format(idx) + ".png"), colored_mask_dy)
        # dynamic_mask_list.append(colored_mask_dy)
        # save_numpy_mask(non_black_mask, dy_path, idx)

        # non_black_mask = static_mask_image[-1][...,0] > 25
        # colored_mask_st = show_mask(non_black_mask, gt, obj_id=1, random_color = False)
        # static_mask_list.append(colored_mask_st)
        # save_numpy_mask(non_black_mask, st_path, idx)
        


    # imageio.mimwrite(os.path.join(render_path, 'video_gt.mp4'), gt_list, fps=30)
    # imageio.mimwrite(os.path.join(render_path, 'video_static_mask.mp4'), static_mask_list, fps=30)
    # imageio.mimwrite(os.path.join(render_path, 'video_dynamic_mask.mp4'), dynamic_mask_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_tracking.mp4'), track_list, fps=10)
    imageio.mimwrite(os.path.join(render_path, 'video_tracking_render.mp4'), track_img_lis, fps=10)

    
    
    

def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = True
        dataset.eval = False
        num_classes = 2
        print("Num classes: ",num_classes)

        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        
        # 是否Load .pth文件
        # checkpoint = '/media/yangtongyu/T9/code2/sa4d-time_variant_ie/output/hypernerf/broom/chkpnt_fine_3000.pth'
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt, stage = 'coarse')    # coarse 不包含 load_pose

        classifier = Classifier(hyperparam, dataset.feature_dim, num_classes)
        cam_type = scene.dataset_type
        loaded_iter = 3000
        
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        
        gaussians.restore(gaussians_params, opt, stage)
        classifier.restore(classifier_params, opt)

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type)
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
    parser.add_argument("--configs", type=str)
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--cam_view", type=str, default='cam16')
    parser.add_argument("--num_classes", type=int, default=2)
    cmdlne_string = ['--model_path', 'output/hypernerf/broom/']
    args = get_combined_args(parser, target_cfg="feature", cmdlne_string = cmdlne_string)
    
    
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode, args.cam_view)