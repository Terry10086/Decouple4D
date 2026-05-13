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
from gaussian_renderer import render, render_contrastive_feature, render_seperate, render_probs
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
import torch.nn.functional as F
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

color = [np.array([0.9, 0.9, 0.9, 0.6]), np.array([0,0.5,0.1,0.6]), np.array([0, 0, 0, 0.6]),]

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

def show_mask1(mask, image, obj_id=1, random_color = False):
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

def visualize_and_save(probs, probs_dy, probs_all, save_path):
    imgs = [
        #pred_obj.detach().cpu().numpy(), 
        probs, 
        #pred_dy.detach().cpu().numpy(), 
        probs_dy, 
        #pred_obj_all.detach().cpu().numpy(), 
        probs_all
    ]
    cmaps = [ 'viridis'] * 3
    titles = [ 'probs_st', 'probs_dy', 'probs_all']

    plt.figure(figsize=(12, 4))
    for i, (img, cmap, title) in enumerate(zip(imgs, cmaps, titles)):
        ax = plt.subplot(1, 3, i+1)
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if cmap == 'viridis':
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

import cv2

def save_rendered_prob_gray(prob, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 取平均作为单通道（也可以选某一通道）
    prob_img = (prob * 255).clip(0, 255).astype(np.uint8)
    
    cv2.imwrite(save_path, prob_img)
    # plt.figure(figsize=(8, 6))
    # im = plt.imshow(prob_img, cmap='viridis', vmin=0.0, vmax=1.0)
    # plt.colorbar(im, fraction=0.046, pad=0.04)
    # plt.axis('off')  # 去掉坐标轴
    # plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.close()

    # plt.imsave(save_path, prob_img, cmap='viridis', vmin=0.0, vmax=1.0)


def resize_to_max_shape(*video_lists):
    # 获取所有图像的最大高宽
    max_h = max(img.shape[0] for v in video_lists for img in v)
    max_w = max(img.shape[1] for v in video_lists for img in v)

    # 就地修改每个列表中的图像大小
    for v in video_lists:
        for i in range(len(v)):
            v[i] = cv2.resize(v[i], (max_w, max_h))


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, cam_type, use_BCE=True, gru = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    render_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "render")
    makedirs(render_path_img, exist_ok=True)
    depth_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    makedirs(depth_path_img, exist_ok=True)
    static_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "static")
    makedirs(static_path_img, exist_ok=True)
    dynamic_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")
    makedirs(dynamic_path_img, exist_ok=True)
    probs_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
    makedirs(probs_path_img, exist_ok=True)
    gt_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gt_path_img, exist_ok=True)

    pred_obj_mask_list, render_images, gt_list, dynamic, static, gt_mask_list, render_tensor = [], [], [], [], [], [], [], 
    depth_images, depth_images_st, depth_images_dy, depth_tensor, static_tensor, dynamic_tensor =  [], [], [], [], [], []
    probability = []
    residual_tensor, gt_tensor = [], []
    
    # save_path = os.path.normpath(os.path.join(model_path, 'tsp_times.txt'))
    # with open(save_path, 'r') as f:
    #     tsp_times = [float(line.strip()) for line in f.readlines()]
    with torch.no_grad():
        trajectory_list = []
        for tsp in range(0,len(views),len(views) // 4):
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


    # 看不见的mask内取交集
    if iteration>20000 and name == 'test': # iteration>20000 and name == 'train':
        final_delete_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)     # torch.Size([281137])
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render_seperate(view, gaussians, pipeline, background, cam_type=cam_type, seperate_mask = pred_obj.squeeze(), is_eval = True)
            image_tensor, st_tensor, dy_tensor, st_index2d = render_pkg["render"],  render_pkg['render_st'], render_pkg['render_dy'], render_pkg['points2d_st']
            
            idx = torch.round(st_index2d.detach()).long()
            idx_static = torch.where(pred_obj.squeeze()==0)[0]     # 静态gs在整个gs中的idx
            idx_dynamic = torch.where(pred_obj.squeeze()==1)[0]

            gt_obj = view.objects.cuda().long()[:,:,0]
            obj_tensor = gt_obj.unsqueeze(0)
            mask_dy = (obj_tensor == 1).float()     # 2D mask动态位置
            _, h, w = mask_dy.shape
            valid_mask = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
            valid_index = idx[valid_mask]
            delete_mask = mask_dy[0, valid_index[:, 1], valid_index[:, 0]] == 1     # 这里获得的静态GS，其中1的位置表示在dynamic mask中
            final_delete_mask[idx_static[valid_mask]] &= delete_mask        # 这里还是原始GS的顺序,1表示在dynamic mask里
            final_delete_mask[idx_dynamic] = False

        print('before prune:', gaussians.get_xyz.shape[0])
        gaussians.prune_by_dynamic_mask(final_delete_mask)
        category = category[~final_delete_mask]
        pred_obj = pred_obj[~final_delete_mask]
        print('after prune:', gaussians.get_xyz.shape[0])

    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")): # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if view.uid<48:
        #     continue
        # if view.uid >97:
        #     break
        
        gt  = to8b(view.original_image).transpose(1,2,0)
        gt_list.append(gt)
        gt_tensor.append(view.original_image)

        
        render_pkg = render_seperate(view, gaussians, pipeline, background, cam_type=cam_type, seperate_mask = pred_obj.squeeze(), is_eval = True, category = category)
        image_tensor, viewspace_point_tensor, visibility_filter, radii, st_tensor, dy_tensor, st_index2d = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
            render_pkg["radii"], render_pkg['render_st'], render_pkg['render_dy'], render_pkg['points2d_st']
        
        
        # render mask
        render_pkg_prob = render_probs(view, gaussians, pipeline, bg_color = torch.tensor([1, 0, 0], dtype=torch.float32, device="cuda"), classifier = classifier, seperate_mask = pred_obj.squeeze(), category = category)
        rendered_prob = render_pkg_prob["rendered_prob"]  
#         dr_normalized = render_pkg_prob['dr'] / render_pkg_prob['dr'].norm(dim=1, keepdim=True).clamp(min=1e-8)
#         theta = 2 * torch.acos(dr_normalized[:, 0].clamp(-1.0, 1.0))
#         theta_deg = theta * 180 / torch.pi
#         
#         dr_normalized = render_pkg_prob['dr'] / render_pkg_prob['dr'].norm(dim=1, keepdim=True).clamp(min=1e-8)
#         prior_mask_motion = (render_pkg_prob['dx'].norm(dim=-1, keepdim=True) > render_pkg_prob['dx'].norm(dim=-1, keepdim=True).mean()).float()  
#         prior_mask_scale = (render_pkg_prob['ds'].norm(dim=-1, keepdim=True) > 3*render_pkg_prob['ds'].norm(dim=-1, keepdim=True).mean()).float()  
#         prior_mask_rotate = (theta_deg.unsqueeze(-1) > 3* theta_deg.mean()).float() 
# 
#         prior_mask = (prior_mask_motion + prior_mask_scale + prior_mask_rotate).clamp(max=1.0)
#         pred_obj = prior_mask
        if idx ==0:
            pts = render_pkg['means3D_new']
            s_new = render_pkg['s_new']
            r_new = render_pkg['r_new']
            # gaussians.prune_by_dynamic_mask(pred_obj.squeeze())
            gaussians.save_ply(os.path.join(render_path,"ours_deformed_pc.ply"), pred_obj, pts, s_new, r_new)   # pred_obj prior_mask


        rendered_prob = rendered_prob.clamp(0.0, 1.0)
        probs_all = rendered_prob[1].detach().cpu().numpy().squeeze()>0.5
        save_rendered_prob_gray(probs_all, os.path.join(probs_path_img, '{0:05d}.png'.format(idx)))

        prob_map = rendered_prob[1].detach().cpu().numpy().squeeze()  # [H, W], float in [0,1]

        plt.imsave(
            os.path.join(probs_path_img, '{0:05d}-prob.png'.format(idx)),
            prob_map,
            cmap='coolwarm',
            vmin=0,
            vmax=1
        )
#         
        # render_fut = render_probs1(view, gaussians, pipeline, background, classifier = classifier, seperate_mask = pred_obj.squeeze(), is_eval = False, category=category, identity_encoding = None)
        # probs_st = render_fut["rendered_feature_map_st"][0].detach().cpu().numpy().squeeze()      # torch.Size([32, 960, 536])
        # probs_dy = render_fut["rendered_feature_map_dy"][0].detach().cpu().numpy().squeeze()
        # probs_all = render_fut["render"][0].detach().cpu().numpy().squeeze()
        # 
        # visualize_and_save(probs_st, probs_dy, probs_all, os.path.join(probs_path_img, '{0:05d}.png'.format(idx)))

        # depth
#         depth_keys = ["depth", "depth_st", "depth_dy"]
#         merged_depth = torch.cat([render_pkg[key] for key in depth_keys[:-1]],dim=2).squeeze()
#         merged_depth_normalized = (merged_depth - merged_depth.min()) / (merged_depth.max() - merged_depth.min())
#         colormap = (cm.jet(merged_depth_normalized.cpu().numpy())* 255).astype(np.uint8)
#         imageio.imwrite(os.path.join(depth_path_img, '{0:05d}'.format(idx) + ".png"), colormap)
# 
#         for key in depth_keys:
#             depth = render_pkg[key].cpu().numpy().squeeze()
# 
#             # depth_normalized = (depth - merged_depth.min().cpu().numpy()) / (merged_depth.max().cpu().numpy() - merged_depth.min().cpu().numpy())
#             depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
#             colormap = cm.jet(depth_normalized)
#             
#             if key == "depth":
#                 depth_images.append((colormap * 255).astype(np.uint8))
#             elif key == "depth_st":
#                 depth_images_st.append((colormap * 255).astype(np.uint8))
#             elif key == "depth_dy":
#                 depth_images_dy.append((colormap * 255).astype(np.uint8))
#             elif key == "probs":
#                 probability.append((colormap * 255).astype(np.uint8))

        
        render_tensor.append(image_tensor)
        static_tensor.append(st_tensor)
        dynamic_tensor.append(dy_tensor)
        
        # depth_tensor.append(merged_depth)

        dynamic.append(to8b(dy_tensor).transpose(1,2,0))
        static.append(to8b(st_tensor).transpose(1,2,0))
        render_images.append(to8b(image_tensor).transpose(1,2,0))

        # img_with_mask = show_mask(to8b(view.objects[:,:,0]), gt.copy())
        # gt_mask_list.append(img_with_mask)
        mask_to_save = (rendered_prob[1].detach().cpu()>0.5).float()

        img_with_mask_rendered = show_mask1(mask_to_save, gt.copy())
        pred_obj_mask_list.append(img_with_mask_rendered)

    
    multithread_write(gt_tensor, gt_path_img)
    multithread_write(render_tensor, render_path_img)
    multithread_write(depth_tensor, depth_path_img)
    multithread_write(static_tensor, static_path_img)
    multithread_write(dynamic_tensor, dynamic_path_img)

    resize_to_max_shape(
        gt_list, render_images, dynamic, static,
        pred_obj_mask_list, gt_mask_list,
        probability
    )
    imageio.mimwrite(os.path.join(render_path, 'video_gt.mp4'), gt_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_rgb.mp4'), render_images, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_dynamic.mp4'), dynamic, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_static.mp4'), static, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_mask.mp4'), pred_obj_mask_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_gt_mask.mp4'), gt_mask_list, fps=30)
    
    imageio.mimwrite(os.path.join(render_path, "video_depth.mp4"), depth_images, fps=30)
    imageio.mimwrite(os.path.join(render_path, "video_depth_static.mp4"), depth_images_st, fps=30)
    imageio.mimwrite(os.path.join(render_path, "video_depth_dynamic.mp4"), depth_images_dy, fps=30)
    imageio.mimwrite(os.path.join(render_path, "video_depth_probs.mp4"), probability, fps=30)
    

def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = True
        num_classes = 2
        # dataset.eval = False        # 渲染全部
        print("Num classes: ",num_classes)

        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        classifier = Classifier(hyperparam, dataset.feature_dim, num_classes)
        cam_type = scene.dataset_type

        gru = TrajectoryGRU()
        
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        # loaded_iter = 17000
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        gru_params = torch.load(os.path.join(load_path,"gru.pth"))
        
        gaussians.restore(gaussians_params, opt, stage)
        classifier.restore(classifier_params, opt)
        gru.restore(gru_params)


        # 是否Load .pth文件
        # checkpoint = './output/hypernerf/composite/chkpnt_coarse_3000.pth'
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt, stage = 'coarse')    # coarse 不包含 load_pose

        bg_color = [1,1,1] # if dataset.white_background else [0, 0, 0]
        # bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
        

        if not skip_train:
            render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)
        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)
        if not skip_video:
            render_set(dataset.model_path, "video", loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true", default=True)
    parser.add_argument("--configs", type=str, default='arguments/hypernerf/default.py')
    parser.add_argument("--mode", type=str, default="scene")
    cmdlne_string = ['--model_path', 'output/uw/sardine_2/']
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