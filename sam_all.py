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
from utils.generate_sam import seg_anything_whole_frames, seg_anything_whole_frames_tracking
import concurrent.futures
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="imageio")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

def render_set(views, dataset_source_path):
    # seg_anything_whole_frames(views, dataset_source_path)
    seg_anything_whole_frames_tracking(views, dataset_source_path)

    

def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, cam_view: str):
    with torch.no_grad():
        dataset.object_masks = False
        dataset.eval = False
        
        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False, cam_view=cam_view, is_eval = True)    # iteration=0, 不载入模型
        
        render_set(scene.getTrainCameras(), dataset.source_path)

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
    parser.add_argument("--configs", type=str, default = "arguments/hypernerf/default.py")
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--cam_view", type=str, default='cam16')
    parser.add_argument("--num_classes", type=int, default=2)
    cmdlne_string = ['--model_path', './output/hypernerf/S7/']    # 也决定了读哪个数据集
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