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
from scene.classifier import Classifier, BetterObjectsProjector
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

def bilinear_sample_dino_feat(dino_feat, uv, H=137, W=77, mask=None, chunk=50000, orig_W = 536, orig_H = 960):
    feat_input = dino_feat.unsqueeze(0)  # [1, C, H, W]
    N = uv.shape[0]
    C = dino_feat.shape[0]

    uv = uv.clone()
    uv[:, 0] = uv[:, 0] / orig_W * W
    uv[:, 1] = uv[:, 1] / orig_H * H

    if mask is not None:
        mask = mask.bool()
        uv_valid = uv[mask]  # 只采样 mask==1 的点
        valid_indices = mask.nonzero(as_tuple=False).squeeze(1)
    else:
        uv_valid = uv
        valid_indices = torch.arange(N, device=uv.device)

    M = uv_valid.shape[0]
    all_feats = []

    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        uv_chunk = uv_valid[start:end]

        # 归一化坐标到 [-1, 1]
        norm_uv = uv_chunk.clone()
        norm_uv[:, 0] = (uv_chunk[:, 0] / (W - 1)) * 2 - 1  # u → x
        norm_uv[:, 1] = (uv_chunk[:, 1] / (H - 1)) * 2 - 1  # v → y
        grid = norm_uv.view(1, -1, 1, 2)  # [1, M, 1, 2]

        # 插值
        sampled = F.grid_sample(feat_input.float(), grid, mode='bilinear', align_corners=True)  # [1, C, M, 1]
        sampled = sampled.squeeze(0).squeeze(2).T  # [M, C]
        all_feats.append(sampled)

    sampled_feats = torch.cat(all_feats, dim=0)  # [M, C]

    # 构造全 0 初始化的输出（保持和输入 uv 一致大小）
    output_feats = torch.zeros((N, C), dtype=sampled_feats.dtype, device=uv.device)
    output_feats[valid_indices] = sampled_feats  # 仅更新有效点

    return output_feats  # shape: [N, C]


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, projector, cam_type, dataset_source_path, use_dino = False, dino_list = None, dino_list_id = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    mask_path = os.path.join(dataset_source_path, "final_mask")
    makedirs(mask_path, exist_ok=True)

    render_images, gt_list = [], [],  
    
    num_batches = len(views)
    # accumulated_feat = torch.zeros((gaussians._xyz.shape[0], 768), dtype=torch.float32, device='cuda')
    # count_accumulated = torch.zeros((gaussians._xyz.shape[0], 1), dtype=torch.float32, device='cuda')

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     gt  = to8b(view.original_image).transpose(1,2,0)
    #     gt_list.append(gt)
    #     output = render(view, gaussians, pipeline, background, stage="fine", cam_type=cam_type, is_eval = True)       # fine stage !!!
    #     uv = output["points2d"]
    #     mask = output["final_mask"]

    #     dino_feat = dino_list[view.uid].cuda().squeeze() # torch.Size([1, 1, 137, 77, 768])
    #     dino_feat = dino_feat.permute(2, 0, 1)   

    #     H, W = dino_feat.shape[1], dino_feat.shape[2]
    #     sampled_feat = bilinear_sample_dino_feat(dino_feat, uv, H, W, mask, chunk=100000, orig_H=output["render"].shape[1], orig_W=output["render"].shape[2])  # torch.Size([362649, 768])

    #     valid_indices = mask.nonzero(as_tuple=False).squeeze(1) # index torch.Size([165854])
    #     accumulated_feat += sampled_feat
    #     count_accumulated[valid_indices] += 1
        
    #     if name in ["train", "test"]:
    #         gt = view.original_image[0:3, :, :]
    #         gt_list.append(gt)

    # valid_mask = (count_accumulated.squeeze(-1) > 0)

    # average_feat = accumulated_feat[valid_mask] / count_accumulated[valid_mask]
    # average_feat_norm = F.normalize(average_feat, dim=1)


    average_feat_norm = projector(gaussians._objects_dc.unsqueeze(-1).unsqueeze(-1)).squeeze().squeeze()

    pca = PCA(n_components=64)
    features_pca = pca.fit_transform(average_feat_norm.cpu().numpy())

    # 4. KMeans 聚类
    from sklearn.cluster import KMeans
    
    labels = KMeans(n_clusters=2, random_state=0).fit_predict(features_pca)
    # label_counts = np.bincount(labels)
    # print("Cluster label counts:", label_counts)

    # from sklearn.cluster import MiniBatchKMeans

    # k = 10  # 可调整
    # features = average_feat_norm.cpu().numpy()

    # kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, random_state=0)
    # cluster_labels = kmeans.fit_predict(features)  # shape: [362649]

    # print("Cluster label counts:", np.bincount(labels))
    import open3d as o3d
    

    colors = plt.get_cmap('tab10')(labels % 4)[:, :3]  # RGB in [0,1]

    N = gaussians._xyz.shape[0]
    all_points = gaussians._xyz.detach().cpu().numpy()  # (N, 3)
    all_colors = np.zeros((N, 3), dtype=np.float32)
    # valid_mask_np = valid_mask.cpu().numpy()  # 转成 numpy bool 数组
    # all_colors[valid_mask_np] = colors  # colors 对应 valid 点的颜色
    all_colors = colors


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # 保存为可视化用的点云文件
    o3d.io.write_point_cloud("clustered_points.ply", pcd)
    imageio.mimwrite(os.path.join(render_path, 'video_gt.mp4'), gt_list, fps=30)
    imageio.mimwrite(os.path.join(render_path, 'video_rgb.mp4'), render_images, fps=30)

    

def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str, use_dino = False, dino_list = None, dino_list_id = None):
    with torch.no_grad():
        stage = 'fine'
        dataset.object_masks = True
        dataset.eval = False
        num_classes = 2
        print("Num classes: ",num_classes)

        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        classifier = Classifier(hyperparam, dataset.feature_dim, num_classes)
        projector = BetterObjectsProjector().cuda()
        cam_type = scene.dataset_type
        
        # load checkpoint
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        classifier_params = torch.load(os.path.join(load_path,"classifier.pth"))
        projector_params = torch.load(os.path.join(load_path,"projector.pth"))

        gaussians.restore(gaussians_params, opt, stage)
        classifier.restore(classifier_params, opt)
        projector.restore(projector_params)


        # 是否Load .pth文件
        # checkpoint = './output/hypernerf/chicken/chkpnt_fine_14000.pth'
        # (model_params, _) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt, stage = 'fine')    # coarse 不包含 load_pose
       
        background = torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")
        # background = [0, 0, 0]

        
        render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, projector, cam_type, dataset.source_path, use_dino = use_dino, dino_list = dino_list, dino_list_id = dino_list_id)

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
    cmdlne_string = ['--model_path', './output/hypernerf/chicken/']
    args = get_combined_args(parser, target_cfg="scene", cmdlne_string = cmdlne_string)
    
    
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

        # load dino
    use_dino = True
    if use_dino:
        dino_list = []
        dino_list_id = []
        from glob import glob
        dino_dir = os.path.join(os.path.dirname(args.source_path), "dinos", "images")
        dino_paths = sorted(glob(os.path.join(dino_dir, "*.npy")))
        if len(dino_paths)>0:
            for i in range(len(dino_paths)):
                dino_path = os.path.join(dino_paths[i])
                features = np.load(dino_path)
                filename = os.path.basename(dino_path)
                dino_list.append(torch.from_numpy(features))
                import re
                dino_list_id.append(int(re.search(r'\d+$', os.path.splitext(filename)[0]).group()))


    render_sets(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.skip_video, args.mode, args.cam_view,
                use_dino=use_dino,
             **({"dino_list": dino_list, "dino_list_id": dino_list_id} if use_dino else {}))