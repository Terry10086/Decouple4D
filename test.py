import os
import numpy as np
from tqdm import tqdm

def convert_npz_to_npy(npz_dir, npy_save_dir):
    os.makedirs(npy_save_dir, exist_ok=True)
    files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

    for f in tqdm(files, desc="Converting .npz to .npy"):
        npz_path = os.path.join(npz_dir, f)
        data = np.load(npz_path)

        packed = data['packed_mask']
        shape = tuple(data['original_shape'])

        # 解压：np.unpackbits 会变成 (N_bits,) 形状
        total_bits = np.prod(shape)
        unpacked = np.unpackbits(packed)[:total_bits]  # 裁掉填充位
        mask = unpacked.reshape(shape).astype(np.uint8)

        # 保存为 .npy
        save_name = f.replace('.npz', '.npy')
        np.save(os.path.join(npy_save_dir, save_name), mask)

convert_npz_to_npy("/media/yangtongyu/T9/code2/sa4d-time_variant_ie/data/hypernerf/chickchicken/sam_mask", "/media/yangtongyu/T9/code2/sa4d-time_variant_ie/data/hypernerf/chickchicken/sam_mask_npy")
