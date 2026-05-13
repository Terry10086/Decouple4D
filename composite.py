import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from scene import Scene
from scene.classifier import Classifier, TrajectoryGRU
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature, render_seperate, render_probs, render_seperate_composite
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


def resize_to_max_shape(*video_lists):
    # 获取所有图像的最大高宽
    max_h = max(img.shape[0] for v in video_lists for img in v)
    max_w = max(img.shape[1] for v in video_lists for img in v)

    # 就地修改每个列表中的图像大小
    for v in video_lists:
        for i in range(len(v)):
            v[i] = cv2.resize(v[i], (max_w, max_h))

def merge_gaussians_params(params1, params2):
    (
        active_sh_degree1, xyz1, deform_state1, deformation_table1,
        features_dc1, features_rest1, scaling1, rotation1, opacity1,
        objects_dc1, max_radii2D1, xyz_grad1, denom1,
        opt_dict1, spatial_lr_scale1
    ) = params1

    (
        active_sh_degree2, xyz2, deform_state2, deformation_table2,
        features_dc2, features_rest2, scaling2, rotation2, opacity2,
        objects_dc2, max_radii2D2, xyz_grad2, denom2,
        opt_dict2, spatial_lr_scale2
    ) = params2

    # 假设 active_sh_degree 和 spatial_lr_scale 是标量且相同
    active_sh_degree = active_sh_degree1
    spatial_lr_scale = spatial_lr_scale1

    # 合并 tensor 类型的参数
    index = xyz1.shape[0]
    xyz = torch.cat([xyz1, xyz2], dim=0)
    features_dc = torch.cat([features_dc1, features_dc2], dim=0)
    features_rest = torch.cat([features_rest1, features_rest2], dim=0)
    scaling = torch.cat([scaling1, scaling2], dim=0)
    rotation = torch.cat([rotation1, rotation2], dim=0)
    opacity = torch.cat([opacity1, opacity2], dim=0)
    objects_dc = torch.cat([objects_dc1, objects_dc2], dim=0)
    max_radii2D = torch.cat([max_radii2D1, max_radii2D2], dim=0)
    xyz_gradient_accum = torch.cat([xyz_grad1, xyz_grad2], dim=0)
    denom = torch.cat([denom1, denom2], dim=0)

    # 合并 deform_state 和 deformation_table：保留一个或按你的逻辑处理
    deform_state = deform_state1  # 或自定义合并逻辑
    deformation_table = deformation_table1  # 或自定义合并逻辑

    # 合并字典（假设没有 key 冲突）
    opt_dict = {**opt_dict1, **opt_dict2}

    # 打包为合并后的参数列表
    merged_gaussians_params = [
        active_sh_degree,
        xyz,
        deform_state,
        deformation_table,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        objects_dc,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale
    ]

    return merged_gaussians_params, index


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def sep_gaussian(model_path, name, iteration, views, gaussians, pipeline, background, classifier, cam_type, use_BCE=True, gru = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)

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
    if iteration>20000 and name == 'train': 
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

    # save static and dynamic
    torch.save(gaussians.capture(proj=~pred_obj.squeeze()), os.path.join(render_path, "gaussians_static.pth"))
    torch.save(gaussians.capture(proj=pred_obj.squeeze()), os.path.join(render_path, "gaussians_dynamic.pth"))

    gaussians.prune_by_dynamic_mask(pred_obj.squeeze())
    gaussians.save_ply(os.path.join(render_path,"deformed_point_cloud.ply"))   # pred_obj prior_mask
    
    # gaussians.save_ply(os.path.join(render_path,"deformed_point_cloud.ply"))


    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, index):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    render_path_img = os.path.join(model_path, name, "ours_{}".format(iteration), "editing")
    makedirs(render_path_img, exist_ok=True)

    print("\033[31m" + "editing path: " + render_path_img + "\033[0m")

    pred_obj_mask_list, render_images, gt_list, dynamic, static, gt_mask_list, render_tensor = [], [], [], [], [], [], [], 
    gt_tensor = []
    
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt  = to8b(view.original_image).transpose(1,2,0)
        gt_list.append(gt)
        gt_tensor.append(view.original_image)

        staic_bkg = views[15]
        render_pkg = render_seperate_composite(view, gaussians, pipeline, background, cam_type=cam_type, index = index, staic_bkg = staic_bkg,idx=idx)
        image_tensor = render_pkg["render"]
        
        
        # if idx ==6:
        #     pts = render_pkg['means3D_new']
        #     s_new = render_pkg['s_new']
        #     r_new = render_pkg['r_new']
        #     gaussians.save_ply(os.path.join(render_path,"deformed_point_cloud.ply"))   # pred_obj prior_mask

        render_images.append(to8b(image_tensor).transpose(1,2,0))
        render_tensor.append(image_tensor)
    
    multithread_write(render_tensor, render_path_img)
    resize_to_max_shape(
        gt_list, render_images
        # , dynamic, static,
        # pred_obj_mask_list, gt_mask_list,
        # depth_images, depth_images_st, depth_images_dy, probability
    )

    # imageio.mimwrite(os.path.join(render_path, 'video_gt.mp4'), gt_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_rgb_edit.mp4'), render_images, fps=30)
    



def save_sd(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = True
        num_classes = 2
        print("Num classes: ",num_classes)

        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        classifier = Classifier(hyperparam, dataset.feature_dim, num_classes)
        cam_type = scene.dataset_type

        gru = TrajectoryGRU()
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        gru_params = torch.load(os.path.join(load_path,"gru.pth"))
        
        gaussians.restore(gaussians_params, opt, stage)
        classifier.restore(classifier_params, opt)
        gru.restore(gru_params)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        sep_gaussian(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)

        # if not skip_train:
        #     render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)
        # if (not skip_test) and (len(scene.getTestCameras()) > 0):
        #     render_set(dataset.model_path, "test", loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)
        # if not skip_video:
        #     render_set(dataset.model_path, "video", loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, classifier, cam_type, gru=gru)


def composite(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str, scene2):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = True
        dataset.eval = False
        num_classes = 2
        print("Num classes: ",num_classes)

        # gaussians_dy = GaussianModel(dataset.sh_degree, mode, hyperparam)
        # gaussians_st = GaussianModel(dataset.sh_degree, mode, hyperparam)
        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        
        cam_type = scene.dataset_type
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path1 = os.path.normpath(os.path.join(scene.model_path, "train", "ours_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path1, str(loaded_iter)))

        loaded_iter = searchForMaxIteration(os.path.join(os.path.dirname(os.path.dirname(scene.model_path)), scene2, "point_cloud"))
        load_path2 = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(scene.model_path)), scene2, "train", "ours_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path2, str(loaded_iter)))

        params_dy = torch.load(os.path.join(load_path1,"gaussians_dynamic.pth"))    # 
        params_st = torch.load(os.path.join(load_path2,"gaussians_static.pth"))     # 

        
        # merged_params = params_dy    
        # index = gaussians._xyz.shape[0]
        merged_params, index = merge_gaussians_params(params_dy, params_st)
        gaussians.restore(merged_params, opt, stage)
        
        # gaussians_dy.restore(params_dy, opt, stage)
        # gaussians_st.restore(params_st, opt, stage)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, index)
        

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
    
    safe_state(args.quiet)

    # 动静态分开保存高斯
    save_sd(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode, args.cam_view)
    
    # 组合高斯
    # scene_name = 'S1'   # static
    # composite(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode, args.cam_view, scene_name)