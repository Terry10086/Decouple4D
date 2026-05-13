import os, sys
import torch
from scene import Scene
from scene.classifier import Classifier
from tqdm import tqdm
from os import makedirs
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import imageio
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="imageio")
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.amg import calculate_stability_score

import torch
from torchvision.utils import save_image
from collections import OrderedDict
from scipy.ndimage import label

def save_sam_mask_visualization(sam_mask, output_path):
    n_masks = sam_mask.shape[0]
    colors = torch.rand(sam_mask.shape[0], 1, 1, 3).cuda()  # 为每个 mask 生成一个随机颜色
    sam_mask_vis = torch.zeros(sam_mask.shape[1], sam_mask.shape[2], 3).cuda()
    for i in range(n_masks):
        mask = sam_mask[i]  
        mask_color = colors[i]  
        mask = mask.squeeze(-1)  # 去掉最后的维度, shape: (960, 536)

        # 将当前 mask 的颜色叠加到图像中
        sam_mask_vis += (mask.unsqueeze(-1) * mask_color)  # 叠加每个 mask 的颜色

    sam_mask_vis = sam_mask_vis.permute(2, 0, 1)
    save_image(sam_mask_vis, output_path)  # 确保将图像移到 CPU 后再保存

def save_compressed_mask(sam_mask, save_path):
    sam_mask = sam_mask.squeeze(-1).cpu().numpy().astype(np.bool_)  # [328, 960, 536]
    packed = np.packbits(sam_mask, axis=-1)  # 在最后一维压缩
    np.savez_compressed(save_path, packed_mask=packed, original_shape=sam_mask.shape)


def generate_colored_mask(sam_index_mask: np.ndarray, device="cuda"):
    """
    将 sam_index_mask (H, W) 转为彩色可视化图像，每个 ID 显示为随机颜色。

    Args:
        sam_index_mask (np.ndarray): shape (H, W), 每个像素是一个 int ID
        device (str): "cuda" 或 "cpu"

    Returns:
        sam_mask_vis (torch.Tensor): shape [H, W, 3]，RGB图，float32, 值在[0, 1]
    """
    height, width = sam_index_mask.shape
    alive_indexs = np.unique(sam_index_mask)

    sam_mask_vis = torch.zeros((height, width, 3), dtype=torch.float32, device=device)

    for mask_index in alive_indexs:
        mask_tensor = torch.from_numpy((sam_index_mask == mask_index).astype(np.float32)).to(device)  # [H, W]
        random_color = torch.rand(1, 1, 3, device=device)  # 随机颜色 [1, 1, 3]
        sam_mask_vis += mask_tensor.unsqueeze(-1) * random_color  # 广播乘法 + 累加

    sam_mask_vis = sam_mask_vis.clamp(0, 1)  # 保证颜色值在合法范围
    return sam_mask_vis

def load_compressed_mask(load_path):
    data = np.load(load_path)
    packed = data['packed_mask']
    shape = tuple(data['original_shape'])
    unpacked = np.unpackbits(packed, axis=-1)
    # unpackbits 会导致最后一维长度变为8的倍数，裁掉多余的部分
    num_bits = np.prod(shape)
    unpacked = unpacked.reshape(-1)[:num_bits].reshape(shape)
    return torch.from_numpy(unpacked.astype(np.uint8))

def seg_anything(view, base_path, idx):
    visual_path = os.path.join(base_path, 'sam_semantic_visualize')
    save_path = os.path.join(base_path, 'sam_semantic_mask')
    os.makedirs(visual_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    sam_masks = []

    # print("\033[91mGenerating SAM\033[0m")
    save_file_name = f"{idx:06d}.png"
    save_path_idx = os.path.join(save_path, save_file_name)
    
    if os.path.exists(save_path_idx):
        # print(f"Skipping {save_file_name} as it already exists.")
        semantic_mask = np.array(Image.open(save_path_idx))
        return semantic_mask


    gt = view.original_image.permute(1,2,0)     # (h, w, 3)

    # use SAM to get fine masks
    sam = sam_model_registry['vit_h'](checkpoint=r'/media/yangtongyu/T9/code2/sa4d-time_variant_ie/submodules/sam_vit_h_4b8939.pth').cuda()
    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=800, 
        )
    
    sam_image = (gt.cpu().numpy() * 255).astype(np.uint8)
    height, width = sam_image.shape[0], sam_image.shape[1]

    sam_results = sorted(mask_generator.generate(sam_image), key=(lambda x: x['area']), reverse=True)
    # some pixels are not included in sam_results, so fix them
    sam_index_mask = np.ones((height, width, 1), dtype=int) * -1
    for index, sam_result in enumerate(sam_results):
        sam_index_mask[sam_result['segmentation']] = index
    index_end = len(sam_results)
    index = index_end
    # for ij in range(height*width):
    #     i, j = ij // width, ij % width
    #     if sam_index_mask[i, j] != -1: continue
    #     neighbors = []
    #     if i > 0 and sam_index_mask[i-1, j] >= index_end: neighbors.append(sam_index_mask[i-1, j])
    #     if j > 0 and sam_index_mask[i, j-1] >= index_end: neighbors.append(sam_index_mask[i, j-1])
    #     if i > 0 and j > 0 and sam_index_mask[i-1, j-1] >= index_end: neighbors.append(sam_index_mask[i-1, j-1])
    #     neighbors = np.unique(neighbors)
    #     if len(neighbors) == 0:
    #         sam_index_mask[i, j] = index
    #         index += 1
    #     else:
    #         sam_index_mask[i, j] = neighbors[0]
    #         for neighbor in neighbors[1:]:
    #             sam_index_mask[sam_index_mask==neighbor] = neighbors[0]
    sam_index_mask = sam_index_mask.squeeze()

    # 找到未被 SAM 覆盖的区域（-1）
    unlabeled_mask = (sam_index_mask == -1)

    # 对未标记区域进行连通区域标记
    labeled, num_features = label(unlabeled_mask)  # 每个连通区域分配一个整数label（从1开始）

    # 计算新的 label 起始值，避免与已有 index 冲突
    label_offset = index  # index 是你之前设定的起始索引
    labeled[labeled > 0] += label_offset

    # 填入 sam_index_mask
    sam_index_mask[unlabeled_mask] = labeled[unlabeled_mask]

    # 更新 index
    index = label_offset + num_features

    # 如果你需要恢复成原始的 shape (H, W, 1)
    sam_index_mask = sam_index_mask[..., None]

    alive_indexs = sorted(np.unique(sam_index_mask))


    index_map = {val: idx for idx, val in enumerate(alive_indexs)}
    id_map_np = np.vectorize(lambda x: index_map[x])(sam_index_mask)  # shape: (H, W)
    semantic_mask = torch.from_numpy(id_map_np).long().cuda()         # shape: [H, W]
    # 保存文件名
    output_file_name = f"{idx:06d}.png"
    visual_path_idx = os.path.join(visual_path, output_file_name)
    semantic_path_idx = os.path.join(save_path, output_file_name)

    # 1️⃣ 保存语义mask（单通道ID图像）为PNG
    # 先转为CPU的PIL图像（uint8）
    semantic_mask = semantic_mask.cpu().numpy().astype(np.uint16).squeeze()  # 若ID超过255可转成np.uint16
    Image.fromarray(semantic_mask).save(semantic_path_idx)

    # 2️⃣ 保存可视化图（RGB图像）
    # sam_mask_vis 是 [H, W, 3]，float32 范围 [0, 1]
    sam_mask_vis = generate_colored_mask(semantic_mask, device="cuda")
    save_image(sam_mask_vis.permute(2, 0, 1), visual_path_idx)  # 转为 [C, H, W]

    
    return semantic_mask


def seg_anything_whole_frames(views, base_path):
    sam = sam_model_registry['vit_h'](checkpoint=r'/media/yangtongyu/T9/code2/sa4d-time_variant_ie/submodules/sam_vit_h_4b8939.pth').cuda()
    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=800, 
        )
    
    visual_path = os.path.join(base_path, 'sam_semantic_visualize')
    save_path = os.path.join(base_path, 'sam_semantic_mask')
    os.makedirs(visual_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    

    print("\033[91mGenerating SAM\033[0m")
    for idx, view in tqdm(enumerate(views), desc="Processing views", unit="view"):

        save_file_name = f"{idx:06d}.png"
        save_path_idx = os.path.join(save_path, save_file_name)
        
        if os.path.exists(save_path_idx):
            continue
        else:
            gt = view.original_image.permute(1,2,0)     # (h, w, 3)
            sam_image = (gt.cpu().numpy() * 255).astype(np.uint8)
            height, width = sam_image.shape[0], sam_image.shape[1]

            sam_results = sorted(mask_generator.generate(sam_image), key=(lambda x: x['area']), reverse=True)
            # some pixels are not included in sam_results, so fix them
            sam_index_mask = np.ones((height, width, 1), dtype=int) * -1
            for index, sam_result in enumerate(sam_results):
                sam_index_mask[sam_result['segmentation']] = index
            index_end = len(sam_results)
            index = index_end
            sam_index_mask = sam_index_mask.squeeze()

            # 找到未被 SAM 覆盖的区域（-1）
            unlabeled_mask = (sam_index_mask == -1)

            # 对未标记区域进行连通区域标记
            labeled, num_features = label(unlabeled_mask)  # 每个连通区域分配一个整数label（从1开始）

            # 计算新的 label 起始值，避免与已有 index 冲突
            label_offset = index  # index 是你之前设定的起始索引
            labeled[labeled > 0] += label_offset

            # 填入 sam_index_mask
            sam_index_mask[unlabeled_mask] = labeled[unlabeled_mask]

            # 更新 index
            index = label_offset + num_features

            # 如果你需要恢复成原始的 shape (H, W, 1)
            sam_index_mask = sam_index_mask[..., None]

            alive_indexs = sorted(np.unique(sam_index_mask))

            index_map = {val: idx for idx, val in enumerate(alive_indexs)}
            id_map_np = np.vectorize(lambda x: index_map[x])(sam_index_mask)  # shape: (H, W)
            semantic_mask = torch.from_numpy(id_map_np).long().cuda()         # shape: [H, W]
            # 保存文件名
            output_file_name = f"{idx:06d}.png"
            visual_path_idx = os.path.join(visual_path, output_file_name)
            semantic_path_idx = os.path.join(save_path, output_file_name)

            # 1️⃣ 保存语义mask（单通道ID图像）为PNG
            # 先转为CPU的PIL图像（uint8）
            semantic_mask = semantic_mask.cpu().numpy().astype(np.uint16).squeeze()  # 若ID超过255可转成np.uint16
            Image.fromarray(semantic_mask).save(semantic_path_idx)

            # 2️⃣ 保存可视化图（RGB图像）
            # sam_mask_vis 是 [H, W, 3]，float32 范围 [0, 1]
            sam_mask_vis = generate_colored_mask(semantic_mask, device="cuda")
            save_image(sam_mask_vis.permute(2, 0, 1), visual_path_idx)  # 转为 [C, H, W]

def seg_anything_whole_frames_tracking(views, base_path):
    visual_path = os.path.join(base_path, 'sam_semantic_visualize')
    read_path = os.path.join(base_path, 'object_mask')
    save_path = os.path.join(base_path, 'sam_semantic_mask')
    os.makedirs(visual_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    print("\033[91mGenerating SAM\033[0m")
    for idx, view in tqdm(enumerate(views), desc="Processing views", unit="view"):

        save_file_name = f"{idx:05d}.png"   # f"frame_{idx+1:05d}.png"
        save_path_idx = os.path.join(read_path, save_file_name)
        objects = Image.open(save_path_idx)
        sam_index_mask = np.array(objects)

        # 找到未被 SAM 覆盖的区域（-1）
        unlabeled_mask = (sam_index_mask == 0)

        # 对未标记区域进行连通区域标记
        labeled, num_features = label(unlabeled_mask)  # 每个连通区域分配一个整数label（从1开始）


#         
#         # 计算新的 label 起始值，避免与已有 index 冲突
#         label_offset = sam_index_mask.max()  # index 是你之前设定的起始索引
#         labeled[labeled > 0] += label_offset
# 
#         # 填入 sam_index_mask
#         sam_index_mask[unlabeled_mask] = labeled[unlabeled_mask]

        area_threshold = 800
        for region_id in range(1, num_features + 1):
            region_mask = (labeled == region_id)
            region_area = np.sum(region_mask)
            if region_area > area_threshold:
                continue  # 留给大空洞处理

            # 找该小空洞区域边界的邻居ID
            neighbor_ids = set()
            ys, xs = np.where(region_mask)
            for y, x in zip(ys, xs):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if ny < 0 or nx < 0 or ny >= sam_index_mask.shape[0] or nx >= sam_index_mask.shape[1]:
                            continue
                        neighbor_id = sam_index_mask[ny, nx]
                        if neighbor_id != 0:
                            neighbor_ids.add(neighbor_id)

            # 如果包围 ID 是唯一的，直接用这个 ID 填充整个小空洞
            if len(neighbor_ids) == 1:
                fill_id = list(neighbor_ids)[0]
                sam_index_mask[region_mask] = fill_id
            else:
                # 多个ID包围，选最多的
                id_count = {}
                for y, x in zip(ys, xs):
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if ny < 0 or nx < 0 or ny >= sam_index_mask.shape[0] or nx >= sam_index_mask.shape[1]:
                                continue
                            neighbor_id = sam_index_mask[ny, nx]
                            if neighbor_id != 0:
                                id_count[neighbor_id] = id_count.get(neighbor_id, 0) + 1
                if id_count:
                    fill_id = max(id_count, key=id_count.get)
                    sam_index_mask[region_mask] = fill_id







        # 如果你需要恢复成原始的 shape (H, W, 1)
        sam_index_mask = sam_index_mask[..., None]

        alive_indexs = sorted(np.unique(sam_index_mask))

        index_map = {val: idx for idx, val in enumerate(alive_indexs)}
        id_map_np = np.vectorize(lambda x: index_map[x])(sam_index_mask)  # shape: (H, W)
        semantic_mask = torch.from_numpy(id_map_np).long().cuda()         # shape: [H, W]
        # 保存文件名
        output_file_name = f"{idx:06d}.png"
        visual_path_idx = os.path.join(visual_path, output_file_name)
        semantic_path_idx = os.path.join(save_path, output_file_name)

        # 1️⃣ 保存语义mask（单通道ID图像）为PNG
        # 先转为CPU的PIL图像（uint8）
        semantic_mask = semantic_mask.cpu().numpy().astype(np.uint16).squeeze()  # 若ID超过255可转成np.uint16
        Image.fromarray(semantic_mask).save(semantic_path_idx)

        # 2️⃣ 保存可视化图（RGB图像）
        # sam_mask_vis 是 [H, W, 3]，float32 范围 [0, 1]
        sam_mask_vis = generate_colored_mask(semantic_mask, device="cuda")
        save_image(sam_mask_vis.permute(2, 0, 1), visual_path_idx)  # 转为 [C, H, W]
