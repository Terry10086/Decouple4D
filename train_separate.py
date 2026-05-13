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
from gaussian_renderer import network_gui, render_all, render, render_seperate, render_seperate1, render_seperate2, render_depth_mask, render_probs
import sys
from scene import Scene, GaussianModel
from scene.classifier import Classifier, TrajectoryGRU
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
import math

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


def dy_threshold(iteration, factor=3.0, beta=1.0, the = 14000):
    t = min((iteration - the) / 3000.0, 1.0)
    dynamic_threshold = 1 + (factor - 1.0) * (1.0 - math.exp(-beta * t))
    return dynamic_threshold


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
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1
        # if stage == "fine" and hyper.time_smoothness_weight != 0:
        #     tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
        #     loss += tv_loss
        # if opt.lambda_dssim != 0:
        #     ssim_loss = ssim(image_tensor,gt_image_tensor)
        #     loss += opt.lambda_dssim * (1.0-ssim_loss)
        
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
                
def training_initialization(dataset, hyper, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from, expname, mode, object_masks=False):
    dataset.object_masks = object_masks     # 载入mask
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

def training_seperation(tb_writer, dataset, hyper, opt, pipe, expname, testing_iterations=None, checkpoint_iterations=None, checkpoint=None, debug_from=None, cam_view=None, timer = None, num_classes = None,  gaussians = None, scene = None, use_BCE=False, use_dino = False):
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

    classifier = Classifier(hyper, dataset.feature_dim, num_classes, use_BCE)
    gru = TrajectoryGRU()
    classifier.training_setup(opt)
    gaussians.training_setup(opt)   # 设置优化器: GS + deformation + pose , stage='semantic'
    gru.training_setup()

    gaussians.use_pose=False    # bg 处理成黑白以外的颜色

    first_iter = 0
    # 所有的ckpt都从point cloud 文件夹下导入
    if checkpoint:
        loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        load_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(loaded_iter)))
        
        print(Fore.MAGENTA + "Load ckpt:" + Style.RESET_ALL, Fore.MAGENTA + load_path + Style.RESET_ALL)

        # 4dgs
        gs_path = os.path.join(load_path,"gaussians.pth")
        if os.path.exists(gs_path):
            gaussians_params = torch.load(gs_path)
            gaussians.restore(gaussians_params, opt, stage)  # , stage='semantic'

        # # e-d3dgs
        # gaussians.load_ply(os.path.join(load_path, "point_cloud.ply"))
        # gaussians.load_model(os.path.join(load_path))
        
        classifier_path = os.path.join(load_path, "classifier.pth")
        if os.path.exists(classifier_path):
            classifier_params = torch.load(classifier_path)
            classifier.restore(classifier_params, opt)
        
        gru_path = os.path.join(load_path, "gru.pth")
        if os.path.exists(gru_path):
            gru_params = torch.load(gru_path)
            gru.restore(gru_params)

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
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    

    # sematic
    if use_dino:
        gaussians.init_semantic()
    print("data loading done")
    
    cls_criterion = torch.nn.BCELoss(reduction='none')
    k_plane = False
    clsfy = False
    all = False
    trajectory_feats = None
    gs_iter = 14000 # 7000 14000
    coarse_mask_iter = 19000    # 12000 19000
    

    flag_traj = True
    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack :
            viewpoint_stack =  temp_list.copy()
        
        # if iteration < 100:
        #     viewpoint_cam = temp_list[0]
        # else:

        
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))

        gaussians.update_learning_rate(viewpoint_cam.uid, iteration, stage)       # 这里只更新gaussian相关学习率
        gaussians.set_seq_idx(viewpoint_cam.uid)

        loss = 0
        
        gt_image = viewpoint_cam.original_image.cuda()

        
        if iteration<=gs_iter:
            k_plane = True
            if iteration==1:
                print("\033[92m[Stage 0 {}] >>>running 4DGS\033[0m".format(iteration))   # 绿色
        elif iteration<=coarse_mask_iter:
            clsfy = True
            k_plane = False
            if iteration==gs_iter+1:
                print("\033[93m[Stage 1 {}] >>> running classifier\033[0m".format(iteration))     # 黄色
        else:
            all = True
            clsfy = k_plane = False
            if iteration==coarse_mask_iter+1:
                print("\033[96m[Stage 3 {}] >>> running all\033[0m".format(iteration))       # 青蓝色

        if clsfy or all:
            if iteration % first_iter == 0 or flag_traj ==True:
                # if all:
                #     gaussians.reset_opacity()
                with torch.no_grad():
                    trajectory_list = []
                    for tsp in range(0, len(temp_list), len(temp_list) // 4):   # len(temp_list) // 4 tsp_times   (len(temp_list) - 1, -1, -len(temp_list) // 4)
                        time = torch.tensor(temp_list[tsp].time).cuda().repeat(gaussians._xyz.shape[0], 1)
                        *_, dx, _, _, _ = gaussians._deformation(gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time)
                        id = classifier._mlp_tjy(dx.detach())
                        trajectory_list.append(id)
                flag_traj = False

            tmp = []+trajectory_list       
            identidy_encoing = classifier._mlp(gaussians.get_xyz.detach())   # , hidden
            tmp.append(identidy_encoing)

            trajectory_feats = torch.stack(tmp)
            hidden = gru(trajectory_feats)

            logits3d = classifier._classifier(hidden.unsqueeze(1).permute(2, 0, 1))
            category = torch.sigmoid(logits3d).squeeze(0)  # 仍然是 torch.Size([2, 134541, 1])
            pred_obj = (category > 0.5) # 到这里classifier都是没有梯度的
                
        if k_plane:
            # 优化K-planes
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image_tensor, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            Ll1 = l1_loss(image_tensor, gt_image)
            loss += Ll1
            # if stage == "fine" and hyper.time_smoothness_weight != 0:
            #     tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            #     loss += tv_loss
            if viewpoint_cam.uid == 48:
                loss += render_pkg['dx'].norm(dim=-1).mean()
                loss += render_pkg['dr'].norm(dim=-1).mean()
                loss += render_pkg['ds'].norm(dim=-1).mean()
        elif clsfy:
            render_pkg_prob = render_probs(viewpoint_cam, gaussians, pipe, bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), classifier = classifier, seperate_mask = pred_obj.squeeze(), category = category)
            rendered_prob = render_pkg_prob["rendered_prob"]

            # 渲染每一个像素的gs属于1个类 2D ray_loss
            loss_logits = entropy_loss(rendered_prob.view(-1).unsqueeze(-1),k=1)   # logits.view(-1).unsqueeze(-1)
            
            lambda_val = 1
            loss += lambda_val * loss_logits

            # 2. 获得类别后，用不同的类别分开获得rendered img
            render_pkg = render_seperate1(viewpoint_cam, gaussians, pipe, background, cam_type=scene.dataset_type, seperate_mask = pred_obj.squeeze(), category = category)
            image_tensor, viewspace_point_tensor, visibility_filter, radii, rendered_image_st = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rendered_image_st"]
            Ll1 = l1_loss(image_tensor, gt_image)
            loss += Ll1
            # Ll1 = None
            
            # 很关键 # 静态先验
            # chiken 2 3.5 3.5  其余 1 2 2 
            motion_threshold = dy_threshold(iteration, factor=1, the = gs_iter) # chicken 2   torchocolate split-cookie 1   a... 1
            scale_threshold = dy_threshold(iteration, factor=1, beta=1.0, the = gs_iter)  # a... 0.5   其余 3
            rotate_threshold = dy_threshold(iteration, factor=2, beta=1.0, the = gs_iter)  # a... 2/不需要  
            
            dr_normalized = render_pkg_prob['dr'] / render_pkg_prob['dr'].norm(dim=1, keepdim=True).clamp(min=1e-8)
            theta = 2 * torch.acos(dr_normalized[:, 0].clamp(-1.0, 1.0))
            theta_deg = theta * 180 / torch.pi

            prior_mask_motion = (render_pkg_prob['dx'].norm(dim=-1, keepdim=True) > motion_threshold*render_pkg_prob['dx'].norm(dim=-1, keepdim=True).mean()).float()  
            prior_mask_scale = (render_pkg_prob['ds'].norm(dim=-1, keepdim=True) > scale_threshold*render_pkg_prob['ds'].norm(dim=-1, keepdim=True).mean()).float()  
            prior_mask_rotate = (theta_deg.unsqueeze(-1) > rotate_threshold*theta_deg.mean()).float()
            
            prior_mask = (prior_mask_motion+ prior_mask_scale).clamp(max=1.0)# prior_mask_scale prior_mask_rotate
            prior_loss = cls_criterion(category, prior_mask).mean()    # F.binary_cross_entropy

            # pkg = render_seperate2(viewpoint_cam, gaussians, pipe, bg_color = background, cam_type=scene.dataset_type)
            # prior_2d = pkg['rendered_motion']>2*pkg['rendered_motion'].mean()
            # prior_loss_2d = cls_criterion(rendered_prob, prior_2d.float()).mean()
            # loss+=prior_loss_2d


            # 使用mask
            if iteration>coarse_mask_iter-2000:     # 17000
                gt_obj = viewpoint_cam.objects.cuda()
                rendered_prob = rendered_prob.clamp(0.0, 1.0)
                    
                loss += cls_criterion(rendered_prob, gt_obj.float().permute(2,0,1)).mean()
                # loss += prior_loss
                
            else:
                loss += prior_loss
            
            if iteration%100==0: # iteration%100==0:
                pkg = render_seperate2(viewpoint_cam, gaussians, pipe, bg_color = background, cam_type=scene.dataset_type)    # 只渲染motion大于avg的GS
                torchvision.utils.save_image(
                    torch.cat([gt_image,  pkg['rendered_st'], pkg['rendered_dy'], pkg['rendered_motion'],pkg['rendered_scale'],pkg['rendered_theta_deg']], dim=2), # pkg['render'],
                            rf'./output/hypernerf/broom/sep1/com_{iteration:06d}-rotate.png'
                )

                render_pkg_sep = render_seperate(viewpoint_cam, gaussians, pipe, background, cam_type=scene.dataset_type, seperate_mask = pred_obj.squeeze(), category = category)
                img, st_tensor, dy_tensor = render_pkg_sep["render"], render_pkg_sep['render_st'], render_pkg_sep['render_dy']
                torchvision.utils.save_image(
                    torch.cat([gt_image, img, st_tensor, dy_tensor, rendered_image_st], dim=2),
                    rf'./output/hypernerf/broom/sep/com_{iteration:06d}.png'
                )

            # 3D 
            loss_binarize = entropy_loss(category, k=1)       # cookies chicken 2.5   teapot 1   S10 2
            loss += loss_binarize

        else:
            gt_obj = viewpoint_cam.objects.cuda()
            render_pkg = render_seperate(viewpoint_cam, gaussians, pipe, background, cam_type=scene.dataset_type, seperate_mask = pred_obj.squeeze(), is_eval = False, category = category)
            image_tensor, viewspace_point_tensor, visibility_filter, radii, st_tensor, dy_tensor = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
                render_pkg["radii"], render_pkg['render_st'], render_pkg['render_dy']

            obj_tensor = gt_obj.permute(2,0,1).float() 

            Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(image_tensor, gt_image)

            if dataset.white_background:
                # bg 处理成白色
                L_dy = l1_loss(gt_image * obj_tensor + (1 - obj_tensor), dy_tensor)
            else:
                L_dy = l1_loss(gt_image * obj_tensor, dy_tensor)

            mask_static = (obj_tensor == 0)
            L_st = l1_loss(st_tensor*mask_static, gt_image*mask_static) + 0.2 * (1-ssim(st_tensor*mask_static, gt_image*mask_static))

            loss += Ll1 + L_dy + L_st

            render_pkg_prob = render_probs(viewpoint_cam, gaussians, pipe, bg_color = torch.tensor([1, 0, 0], dtype=torch.float32, device="cuda"), classifier = classifier, seperate_mask = pred_obj.squeeze(), category = category)
            rendered_prob = render_pkg_prob["rendered_prob"]  
            rendered_prob = rendered_prob.clamp(0.0, 1.0)
            loss_cls_obj = cls_criterion(rendered_prob, gt_obj.float().permute(2,0,1)).squeeze().mean()
            loss += loss_cls_obj

            loss_logits = entropy_loss(rendered_prob.view(-1).unsqueeze(-1),k=1)   # logits.view(-1).unsqueeze(-1) 在americano下很重要
            loss += loss_logits

            loss_binarize = entropy_loss(category, k=1)
            loss += loss_binarize

            if iteration%100==0: # iteration%100==0:
                render_pkg_sep = render_seperate(viewpoint_cam, gaussians, pipe, background, cam_type=scene.dataset_type, seperate_mask = pred_obj.squeeze(), category = category)
                img, st_tensor, dy_tensor = render_pkg_sep["render"], render_pkg_sep['render_st'], render_pkg_sep['render_dy']
                torchvision.utils.save_image(
                    torch.cat([gt_image, img, st_tensor, dy_tensor,], dim=2),
                    rf'./output/hypernerf/broom/sep/com_{iteration:06d}.png'
                )

        if 'image_tensor' in locals():
            psnr_ = psnr(image_tensor, gt_image).mean().double()
        else:
            psnr_ = torch.tensor(0.0)

        
        loss.backward()
        
        if torch.isnan(loss).any():
            print("GS loss is nan, end training, reexecv program now.")
            print(loss_binarize.item(), prior_loss.item(), Ll1.item())
            break
        
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                if k_plane:
                    postfix = {
                                "Loss": f"{ema_loss_for_log:.4f}",
                                "psnr": f"{psnr_:.2f}",
                                "point": f"{total_point}", }
                else:
                    postfix = {
                            "Loss": f"{ema_loss_for_log:.4f}",
                            "psnr": f"{psnr_:.2f}",
                            "point": f"{total_point}",
                            "st": f"{torch.where(pred_obj.squeeze() == 0)[0].numel()}",
                            "dy": f"{torch.where(pred_obj.squeeze() == 1)[0].numel()}",}
            
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.feature_iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            if k_plane:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, classifier= None, use_BCE = use_BCE)
            else:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_seperate, [pipe, background], stage, scene.dataset_type, classifier= classifier, use_BCE = use_BCE, pred_obj =pred_obj, category = category)

            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        pass
                        
            timer.start()

            if ((k_plane and iteration < opt.densify_until_iter) or all) and gaussians._xyz.shape[0]<300000 :  #   opt.densify_until_iter or all
                flag_traj = True
                viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
                viewspace_point_tensor_list = [viewspace_point_tensor]
                # idx_dynamic = torch.where(pred_obj.squeeze()==1)[0]

                for idx in range(0, len(viewspace_point_tensor_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
    
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

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
                if iteration % 19099 == 0:     # opt.opacity_reset_interval
                    if torch.where(pred_obj.squeeze()==0)[0].shape[0] != 0:
                        mask = pred_obj.squeeze()==0
                        print("reset opacity")
                        gaussians.reset_opacity(mask = mask)

#                 elif iteration % opt.dynamic_delete == 0:  # 开始剪枝动态GS和静态中不太对的部分
#                     # pred_obj_soft = ((category <= 0.5) & (category > 0.1)).squeeze()
#                     _, h, w = mask_static.shape
#                     dy_index2d = render_pkg['points2d_dy']
#                     idx = torch.round(dy_index2d.detach()).long()
#                     idx_dynamic = torch.where(pred_obj.squeeze()==1)[0]
# 
#                     valid_mask = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
#                     valid_index = idx[valid_mask]
#                     delete_mask = mask_static[0, valid_index[:, 1], valid_index[:, 0]] == 1
#                     final_delete_mask = torch.zeros(pred_obj.shape[0], dtype=torch.bool, device=idx.device)     # torch.Size([281137])
#                     final_delete_mask[idx_dynamic[valid_mask]] = delete_mask        # 这里还是原始GS的顺序
# 
#                     # final_delete_mask |= pred_obj_soft
#                     gaussians.prune_by_dynamic_mask(final_delete_mask)
                    
                
                    
            # Optimizer step
            if k_plane:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if clsfy:
                classifier.optimizer.step()
                classifier.optimizer.zero_grad(set_to_none = True)
                gru.optimizer.step()
                gru.optimizer.zero_grad(set_to_none = True)
            if all:
#                 for param in gaussians._deformation.get_mlp_parameters():
#                     param.requires_grad_(False)
# 
#                 for param in gaussians._deformation.get_grid_parameters():
#                     param.requires_grad_(False)

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                classifier.optimizer.step()
                classifier.optimizer.zero_grad(set_to_none = True)
                gru.optimizer.step()
                gru.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                point_cloud_path = os.path.normpath(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(iteration)))      
                os.makedirs(point_cloud_path, exist_ok=True)

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(gaussians.capture(), os.path.join(point_cloud_path, "gaussians.pth"))
                torch.save(classifier.capture(), os.path.join(point_cloud_path, "classifier.pth"))
                torch.save(gru.capture(), os.path.join(point_cloud_path, "gru.pth"))
                

                

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, 
                    renderFunc, renderArgs, stage, dataset_type, classifier=None, use_BCE = False, pred_obj =None, category = None):
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
                    if classifier==None:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    else: 
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, seperate_mask = pred_obj.squeeze(), category = category, *renderArgs)["render"], 0.0, 1.0)

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
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    scene_name = "split-cookie-partial"  # split-cookie-partial  americano  chicken  torchocolate  cut-lemon-partial composite_2
    lp = ModelParams(parser, scene_name=scene_name)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,10000,20000]+list(range(5000, 30001, 1000)))
    parser.add_argument("--coarse_save_iterations", nargs="+", type=int, default=[3000])    # only coarse
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 5000, 7000, 10000, 12000, 14000, 17000, 19000, 23000, 25000])  # 保存全部.pth
    parser.add_argument("--start_checkpoint", type=str, default = f'./output/hypernerf/{scene_name}/chkpnt_coarse_3000.pth')   # ./output/hypernerf/{scene_name}/chkpnt_coarse_3000.pth
    parser.add_argument("--is_continue", type=bool, default = True)   # 是否导入.pth
    parser.add_argument("--expname", type=str, default = f"./hypernerf/{scene_name}-wo-rayloss")
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
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # init GS:  
    timer, gaussians, scene, tb_writer = training_initialization(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.coarse_save_iterations, args.start_checkpoint, args.debug_from, args.expname, 
                                                                args.mode, object_masks=True)
    backup_current_script(args.expname)

    with open(os.path.join(args.model_path, "feature_cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        


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
    )
    
    
    print("\nTraining complete.")
