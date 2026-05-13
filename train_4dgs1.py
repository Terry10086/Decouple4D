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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, loss_cls_3d
from gaussian_renderer import network_gui, render_all, render, render_seperate, render_seperate2
import sys
from scene import Scene, GaussianModel
from scene.classifier import Classifier, BetterObjectsProjector
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, get_combined_args
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from utils.system_utils import searchForMaxIteration
from utils.camera_opt import CameraOptModule
from colorama import Fore, Style
import torch.nn.functional as F
import torchvision 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
TENSORBOARD_FOUND = False

def entropy_loss(p, eps=1e-5, k=2.5):
    # p 是概率值张量，保证数值稳定
    p = torch.clamp(p, eps, 1 - eps)
    p_skewed = p ** k
    loss = -(p_skewed * torch.log(p_skewed) + (1 - p_skewed) * torch.log(1 - p_skewed))
    # loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return loss.mean()


def save_img(tensor,idx):
    from PIL import Image
    tensor_image = tensor.detach().cpu()  # 确保 tensor 在 CPU 上
    tensor_image = tensor_image.numpy()  # 转换为 NumPy 数组

    # 归一化到 0-255 并转换为 uint8
    tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min()) * 255
    tensor_image = tensor_image.astype(np.uint8)

    # 维度转换 (C, H, W) -> (H, W, C)
    tensor_image = np.transpose(tensor_image, (1, 2, 0))

    # 保存图片
    image = Image.fromarray(tensor_image)
    filename = f"/media/yangtongyu/T9/code2/sa4d-time_variant_ie/output/output_{idx}.png"
    image.save(filename)

def should_iterate(iteration, decay_start=500, decay_rate=0.0005):  # decay_rate值越大，下降越快
    import math
    if iteration < decay_start:
        return True  # 前 decay_start 轮每次都迭代
    return math.exp(-decay_rate * (iteration - decay_start)) > 0.1  # 控制衰减

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

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations,
                        checkpoint_iterations, checkpoint, debug_from,
                        gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0

    gaussians.training_setup(opt)   # 设置优化器
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            print(Fore.MAGENTA + "Load ckpt:" + Style.RESET_ALL, Fore.MAGENTA + str(checkpoint) + Style.RESET_ALL)
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()


    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()

        gaussians.update_learning_rate(0, iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        loss = Ll1

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss.backward()
        if torch.isnan(loss).any():
            print("Coarse loss is nan,end training, reexecv program now.")
            # os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        # TODO: bugs here, no depth map in render return
                        pass
                        # render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(init=True), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")
                
def training_initialization(dataset, hyper, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from, expname, mode):
    dataset.object_masks = False     # 载入mask
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, mode, hyper)       # scene
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, mode=mode, load_coarse=None)      # 此时选择哪种优化器已经由mode决定好了, dataset
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations,
                        checkpoint_iterations, checkpoint, debug_from,
                        gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer) 
    # scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations,
    #                          checkpoint_iterations, checkpoint, debug_from,
    #                          gaussians, scene, "fine", tb_writer, opt.coarse_iterations,timer)   # fine stage同样迭代3k次                      

    return timer, gaussians, scene, tb_writer

def training_seperation(tb_writer, dataset, hyper, opt, pipe, expname, testing_iterations=None, checkpoint_iterations=None, checkpoint=None, debug_from=None, cam_view=None, timer = None, num_classes = None,  gaussians = None, scene = None, use_BCE=False, use_dino = False, dino_list = None, dino_list_id = None):
    # initialize scene (optimizer, but same GS)
    stage = 'fine'
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    viewpoint_stack = None
    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    if opt.pose_opt == True:
        gaussians.init_RT_seq(len(viewpoint_stack))
        gaussians.set_seq_idx(0)

    
    gaussians.training_setup(opt)   # 设置优化器: GS + deformation + pose  , stage='semantic'

    first_iter = 0
    # 所有的ckpt都从point cloud 文件夹下导入
    if checkpoint:
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        
        print("Load ckpt:", os.path.join(load_path, str(loaded_iter)))
        gaussians_params = torch.load(os.path.join(load_path,"gaussians.pth"))
        gaussians.restore(gaussians_params, opt, stage = 'fine')  # , stage='semantic'
        
        first_iter = loaded_iter


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]     # [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    final_iter = opt.feature_iterations     # 5000

    
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    average_feat_norm = None


    print("data loading done")
    
    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack :
            viewpoint_stack =  temp_list.copy()
        
        if iteration < 100:
            viewpoint_cam = temp_list[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
        gaussians.update_learning_rate(viewpoint_cam.uid, iteration, stage)       # 这里只更新gaussian相关学习率
        gaussians.set_seq_idx(viewpoint_cam.uid)

        
        loss = 0
        loss_cls_obj = None
        # gt_obj = viewpoint_cam.objects.cuda().long()
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 2. 获得类别后，用不同的类别分开获得rendered img
        if iteration % 500 ==0:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type, render_motion =True)
            torchvision.utils.save_image(render_pkg['rendered_motion'],rf'/media/yangtongyu/T9/code2/sa4d-time_variant_ie/output/hypernerf/composite/video/{iteration:06d}.png')
        else:
            render_pkg = render_seperate2(viewpoint_cam, gaussians, pipe, background, cam_type=scene.dataset_type)
        image_tensor, viewspace_point_tensor, visibility_filter, radii, = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1 = l1_loss(image_tensor, gt_image)
        loss += (1.0 - opt.lambda_dssim) * Ll1   # 

        if viewpoint_cam.uid == 0:
            loss += render_pkg['dx'].norm(dim=-1).mean()
            loss += render_pkg['dr'].norm(dim=-1).mean()
            loss += render_pkg['ds'].norm(dim=-1).mean()
        
        psnr_ = psnr(image_tensor, gt_image).mean().double()
        
        if torch.isnan(loss).any():
            print("GS loss is nan, end training, reexecv program now.")
            print(loss_obj_3d.item(), loss_binarize.item(), prior_loss.item(), Ll1.item())
            break
        
        # loss += 0.1*render_pkg['dx'].norm(dim=-1).mean()
        loss.backward()
        
        
        # if iteration < opt.stop_update_classifier and torch.isnan(loss_obj).any():
        #     print("Classifier loss is nan, end training, reexecv program now.")
            # os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        viewspace_point_tensor_list = [viewspace_point_tensor]
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                postfix = {
                            "Loss": f"{ema_loss_for_log:.4f}",
                            "psnr": f"{psnr_:.2f}",
                            "point": f"{total_point}",
                        }

                if loss_cls_obj is not None:
                    postfix["cls"] = f"{loss_cls_obj:.4f}"

                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.feature_iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_seperate2, [pipe, background], stage, scene.dataset_type, use_BCE = use_BCE, dino = average_feat_norm)

            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        pass
                        
            timer.start()

            # Densification
            if iteration < opt.densify_until_iter:  # 10000
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                opacity_threshold = opt.opacity_threshold_coarse    # 0.005
                densify_threshold = opt.densify_grad_threshold_coarse   # 0.0002

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage) # split & clone
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:      # 2000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:     # False
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                
                    
            # Optimizer step
            if iteration < opt.feature_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                point_cloud_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(iteration)))      
                os.makedirs(point_cloud_path, exist_ok=True)

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(gaussians.capture(), os.path.join(point_cloud_path, "gaussians.pth"))
                

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str) 
    print(Fore.MAGENTA + "Output folder: {}".format(os.path.normpath(args.model_path)) + Style.RESET_ALL)

    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, use_BCE = False, dino =None):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()

def backup_current_script(path):
    import shutil
    name = "train_seperate.py"
    script_path = os.path.join('./',name)
    backup_path = os.path.join("./output/", path)   # , "backup"
    os.makedirs(backup_path, exist_ok=True)  # 创建备份文件夹（如果不存在）
    backup_path = os.path.normpath(os.path.join(backup_path, name))

    shutil.copy(script_path, backup_path)
    print(f"Backup saved: {backup_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,10000,20000]+list(range(5000, 20001, 1000)))
    parser.add_argument("--coarse_save_iterations", nargs="+", type=int, default=[3000])    # only coarse
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 7000, 10000, 14000])  # 保存全部.pth
    parser.add_argument("--start_checkpoint", type=str, default = "./output/hypernerf/hand1/chkpnt_fine_13999.pth")   # ./output/hypernerf/chicken-loss_all_woobj3d/chkpnt_coarse_3000.pth
    parser.add_argument("--is_continue", type=bool, default = False)   # 是否导入.pth
    parser.add_argument("--expname", type=str, default = "./hypernerf/hand1")
    parser.add_argument("--configs", type=str, default = "arguments/hypernerf/default.py")  # arguments/hypernerf/default.py arguments/dnerf/bouncingballs.py
    parser.add_argument("--mode", type=str, default="scene")
    parser.add_argument("--cam_view", type=str, default='cam16')
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)     # read configs
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # init GS:  
    timer, gaussians, scene, tb_writer = training_initialization(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.coarse_save_iterations, args.start_checkpoint, args.debug_from, args.expname, args.mode)
    backup_current_script(args.expname)

    with open(os.path.join(args.model_path, "feature_cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # load dino
    use_dino = False
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
                dino_list.append(torch.from_numpy(features).cuda())
                import re
                dino_list_id.append(int(re.search(r'\d+$', os.path.splitext(filename)[0]).group()))
        


    training_seperation(tb_writer,
        lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
        expname=args.expname,
        testing_iterations=args.test_iterations,    # [3000, 7000, 14000]
        checkpoint_iterations=args.checkpoint_iterations,   # [3000, 14000, 20000]
        checkpoint=args.is_continue,
        cam_view=args.cam_view,
        timer=timer,
        num_classes=args.num_classes,
        gaussians=gaussians,
        scene=scene,
        use_BCE=True,
        use_dino=use_dino,
         **({"dino_list": dino_list, "dino_list_id": dino_list_id} if use_dino else {})
    )
    
    
    print("\nTraining complete.")