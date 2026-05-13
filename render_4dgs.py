#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import imageio
import numpy as np
import torch
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
# from gaussian_renderer import GaussianModel
from scene import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures

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
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # breakpoint()
        
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
        
        # rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)#["render"]
        # gaussians.save_ply("./deformed_point_cloud.ply")
        # sys.exit(0)
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        if name in ["train", "test", "video"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(to8b(gt).transpose(1,2,0))
        # if idx >= 10:
            # break
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    # print("writing training images.")

    multithread_write(gt_list, gts_path)
    # print("writing rendering images.")
    multithread_write(render_list, render_path)

    # render_images = render_images[::2] + render_images[1::2]
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb_4dgs.mp4'), render_images, fps=30)
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_gt.mp4'), gt_list, fps=30)
    
def render_sets(opt, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, mode: str, checkpoints):
    with torch.no_grad():
        dataset.object_masks = False 
        gaussians = GaussianModel(dataset.sh_degree, mode, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, mode=mode, shuffle=False)

        print("Load ckpt:", checkpoints)
        (model_params, first_iter) = torch.load(checkpoints)
        gaussians.restore(model_params, opt)

        # load checkpoint
        # from utils.system_utils import searchForMaxIteration
        # loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        # load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        # print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        # gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        # gaussians.restore(gaussians_params, opt, 'fine')
        

        cam_type = scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", first_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)
        if not skip_test:
            render_set(dataset.model_path, "test", first_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video",first_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
            
if __name__ == "__main__":
    # Set up command line argument parser
    scene_name = "hand1"

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--skip_train", default=False)
    parser.add_argument("--skip_test", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str, default='arguments/hypernerf/default.py')
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--checkpoints", type=str, default=f"./output/hypernerf/{scene_name}/chkpnt_fine_14000.pth")    # ./output/robust/statue/point_cloud
    cmdlne_string = ['--model_path', f'./output/hypernerf/{scene_name}']
    args = get_combined_args(parser, target_cfg="scene", cmdlne_string = cmdlne_string)
    # args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(op.extract(args), model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.mode,args.checkpoints)