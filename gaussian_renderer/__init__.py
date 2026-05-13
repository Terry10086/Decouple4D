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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.feature_gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
import sys
from gsplat import rasterization

def compute_velocity(dx, dt):
    velocity = torch.norm(dx, dim=1) / dt  
    return velocity  

def compute_pixel_coverage(points2d, H, W):
    rounded_points = torch.round(points2d).int()  # (N, 2)
    mask = (rounded_points[:, 0] >= 0) & (rounded_points[:, 0] < W) & \
           (rounded_points[:, 1] >= 0) & (rounded_points[:, 1] < H)

    filtered_points = rounded_points[mask]  # 去掉超出边界的点
    valid_indices = torch.nonzero(mask).squeeze()  # 获取未超出边界的索引

    # 计算像素覆盖频率
    histogram = torch.zeros((H, W), dtype=torch.int, device=points2d.device)
    indices = filtered_points.t()  # 转置成 (2, N) 以用于 scatter_add_
    histogram.index_put_((indices[1], indices[0]), torch.ones(filtered_points.shape[0], dtype=torch.int, device=points2d.device), accumulate=True)


    # 只出现一次的像素点对应的 mask
    single_coverage_mask = (histogram <= 3)     # 这个值用来修改 有几个点投影在了这个位置

    # 构造 delete_mask，找到 `filtered_points` 中所有 single_coverage_mask=True 的点
    delete_mask = single_coverage_mask[filtered_points[:, 1], filtered_points[:, 0]]

    # 根据 valid_indices 还原到原始 `points2d` 的 mask
    full_delete_mask = torch.zeros(points2d.shape[0], dtype=torch.bool, device=points2d.device)
    full_delete_mask[valid_indices] = delete_mask 
    full_delete_mask = ~mask | full_delete_mask

    return ~full_delete_mask, histogram


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
            override_mask = None, filtered_mask = None, prob_obj3d = None, is_eval = False, time = None, render_motion = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        if time == None:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        else:
            time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time = torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    
    dx = ds = dr = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)
    else:
        raise NotImplementedError


    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda") if override_mask is None else override_mask
    # mask = pc.get_mask if override_mask is None else override_mask

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    
    
    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    rendered_motion = None
    if render_motion:
        motion = dx.norm(dim=-1).unsqueeze(-1).repeat(1, 3)
        rendered_motion, *_ = rasterizer(
            means3D = means3D_final.detach(),
            means2D = means2D.detach(),      # 是否需要克隆
            shs = None,    # torch.Size([281137, 16, 3])
            colors_precomp = motion,
            opacities = opacity_final.detach(),
            mask = mask,
            scales = scales_final.detach(),
            rotations = rotations_final.detach(),
            cov3D_precomp = None)

    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    dynamic_mask = None
    if is_eval == True:
        # if time[0] == 0:
        #     velocity = torch.norm(dx, dim=1)
        # else:
        #     velocity = compute_velocity(dx, time[0])
        # theta_u = 3 * velocity.mean()       # dynamic
        # theta_l = 0.1 * velocity.mean()    # static

        # mask_dy = velocity > theta_u  
        # mask_st = velocity < theta_l  

        
        # means3D_dy, means2D_dy, shs_dy, opacities_dy, scales_dy, rotations_dy, mask_dynamic = \
        #     (tensor[mask_dy] for tensor in [means3D_final, means2D, shs_final, opacity, scales_final, rotations_final, mask])
        
        # means3D_st, means2D_st, shs_st, opacities_st, scales_st, rotations_st, mask_static = \
        #     (tensor[mask_st] for tensor in [means3D_final, means2D, shs_final, opacity, scales_final, rotations_final, mask])

        # _, _, _, points2d_dy  = rasterizer(
        #     means3D = means3D_dy,
        #     means2D = means2D_dy,
        #     shs = shs_dy,
        #     colors_precomp = None,
        #     opacities = opacities_dy,
        #     mask = mask_dynamic,
        #     scales = scales_dy,
        #     rotations = rotations_dy,
        #     cov3D_precomp = cov3D_precomp)
        
        # _, _ , _, points2d_st = rasterizer(
        #     means3D = means3D_st,
        #     means2D = means2D_st,
        #     shs = shs_st,
        #     colors_precomp = None,
        #     opacities = opacities_st,
        #     mask = mask_static,
        #     scales = scales_st,
        #     rotations = rotations_st,
        #     cov3D_precomp = cov3D_precomp)
        
        # delete_mask_dy, histogram_dy = compute_pixel_coverage(points2d_dy, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)) 
        # delete_mask_st, histogram_st = compute_pixel_coverage(points2d_st, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

        # final_mask_dy = mask_dy.clone()
        # final_mask_dy[mask_dy] &= delete_mask_dy

        # final_mask_st = mask_st.clone()
        # final_mask_st[mask_st] &= delete_mask_st
 

        # # after selection 
        # override_color = torch.zeros(means3D_final.shape, device="cuda")
        # override_color[final_mask_st] = 1.0

        # static_mask, _ , _, points2d_st = rasterizer(
        #     means3D = means3D_final,
        #     means2D = means2D,
        #     shs = None,
        #     colors_precomp = override_color,
        #     opacities = opacity,
        #     mask = mask,
        #     scales = scales_final,
        #     rotations = rotations_final,
        #     cov3D_precomp = cov3D_precomp)
        
        # override_color = torch.zeros(means3D_final.shape, device="cuda")
        # override_color[final_mask_dy] = 1.0
        # scales_final = torch.minimum(scales_final, torch.full_like(scales_final, 0.01))
        # dynamic_mask, _, _, points2d_dy  = rasterizer(
        #     means3D = means3D_final,
        #     means2D = means2D,
        #     shs = None,
        #     colors_precomp = override_color,
        #     opacities = opacity,
        #     mask = mask,
        #     scales = scales_final,
        #     rotations = rotations_final,
        #     cov3D_precomp = cov3D_precomp)

        # return {"dynamic_mask": dynamic_mask,
        #         "points2d_dy": points2d_dy,
        #         "static_mask": static_mask,
        #         "points2d_st" : points2d_st,
        #         "histogram_dy": histogram_dy,
        #         "histogram_st": histogram_st,
        #         }
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )

        depth, _, _ = rasterization(
                means = means3D_final, 
                quats = rotations_final, 
                scales = scales_final, 
                opacities = opacity_final.squeeze(-1), 
                colors = shs_final,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                ) 
        ones = torch.ones((means3D_final.shape[0], 1), dtype=means3D_final.dtype, device=means3D_final.device)
        p_orig1 = torch.cat([means3D_final, ones], dim=1)
        gs_z = (viewpoint_camera.world_view_transform.cuda().T[:3,:] @ p_orig1.T).T

        u, v = points2d[:, 0], points2d[:, 1]
        valid_uv_mask = (u >= 0) & (u < depth.shape[2]) & (v >= 0) & (v < depth.shape[1])

        final_mask = torch.zeros_like(gs_z[:, -1]).cuda()
        valid_z_mask = (gs_z[:, -1] > 0.5)
        combined_mask = valid_uv_mask & valid_z_mask    # N 选出uv投影在图像内的GS

        u_valid = u[combined_mask].long()
        v_valid = v[combined_mask].long()
        depth_valid = depth[0, v_valid, u_valid, 0]

        gs_z_valid = gs_z[combined_mask, -1]

        # 构建 depth 条件 mask
        depth_mask = (gs_z_valid <= depth_valid + 10)

        # 最终 mask：只在满足 combined_mask 的索引中更新为 1（交集）
        final_mask[combined_mask] = depth_mask.float()
        return {"render": rendered_image,
                "mask": rendered_mask,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "deformed_points": means3D_final,
                "points2d": points2d,
                "final_mask": final_mask}
    
    else:
        return {
            "render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "deformed_points": means3D_final,
            "points2d": points2d,
            "rendered_motion": rendered_motion,

            "dx": dx, 
            "dr": dr, 
            "ds": ds,
            
            }


def render_segmentation(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mask, t=None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc._xyz[mask]
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    if t:
        time = torch.tensor(t).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    
    means2D = screenspace_points
    opacity = pc._opacity[mask]
    shs = pc.get_features[mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling[mask]
        rotations = pc._rotation[mask]
        
    deformation_point = pc._deformation_table
    means3D_final, scales_final, rotations_final, opacity_final, shs_final, _ = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)

    # pc._xyz = means3D_final
    # pc.save_masked_ply("./test.ply", mask)
    # return None
    # sys.exit(0)
    
    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, _, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda"),
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "points2d": points2d}    

def render_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity
    shs = pc.get_features
    mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    deformation_point = pc._deformation_table
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
    
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_mask, radii = rasterizer.forward_mask(
        means3D = means3D_final,
        means2D = means2D,
        opacities = opacity,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


from diff_gaussian_rasterization_contrastive_f import GaussianRasterizationSettings as GaussianRasterizationSettingsContrastiveF
from diff_gaussian_rasterization_contrastive_f import GaussianRasterizer as GaussianRasterizerContrastiveF
# from scene.feature_gaussian_model import GaussianModel

def render_contrastive_feature(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, mlp = None, dropout = -1, prob_obj3d = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    
    # # ------------------------------ seperation ------------------------------ 
    # # 1. 根据predict获得每个GS的id
    # is_dynamic = prob_obj3d[:, 1] > 0.5
    # means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D.clone(), scales.clone(), rotations.clone(), opacity.clone(), shs.clone()

    # if is_dynamic.any():  
    #     means3D_final[is_dynamic], scales_final[is_dynamic], rotations_final[is_dynamic], \
    #     opacity_final[is_dynamic], shs_final[is_dynamic] = pc._deformation(
    #         means3D[is_dynamic], scales[is_dynamic], rotations[is_dynamic], 
    #         opacity[is_dynamic], shs[is_dynamic], time[is_dynamic]
    #     )
    
    # # ------------------------------ seperation ------------------------------ 

    # deform
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    identity_encoding = pc._mlp(means3D, time)

    # sam_features = pc.get_sam_features
    # if mlp:
    #     sam_features = mlp(sam_features)

    # if dropout > 0:
    #     rands = torch.rand(opacity.shape[0], device=opacity.device)
    #     dropout_mask = rands < dropout
    #     new_opacity = opacity.detach().clone()
    #     new_opacity[dropout_mask, :] = 0
    #     opacity = new_opacity
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_feature_map, radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = identity_encoding,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_feature_map,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "deformed_points": means3D_final}

def render_all(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
           override_mask = None, filtered_mask = None, mlp = None, dropout = -1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        
    # 设置正常图像的渲染
    rasterizer_4dgs = GaussianRasterizer(raster_settings=raster_settings)
    rasterizer_eid = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        
    deformation_point = pc._deformation_table
    
    means3D_final, scales_final, rotations_final, opacity_final, shs_final, _, hidden = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
    
    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    identity_encoding = pc._mlp(means3D, time)
    

    mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda") if override_mask is None else override_mask
    
    rendered_image, rendered_mask, radii, points2d = rasterizer_4dgs(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = None,
        opacities = opacity,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    rendered_feature_map, _ = rasterizer_eid(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = identity_encoding,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)


    return {"render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "deformed_points": means3D_final,
            "points2d": points2d,
            "rendered_feature_map": rendered_feature_map,
            }


def render_seperate(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
           override_mask = None, filtered_mask = None, prob_obj3d = None, seperate_mask = None, is_eval = False, category = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),    # 世界坐标系到相机坐标系的变换矩阵4x4
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),     # 相机坐标系投影到屏幕的透视投影矩阵4x4
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        

    idx_static = torch.where(seperate_mask==0)[0]
    idx_dynamic = torch.where(seperate_mask==1)[0]

    if idx_dynamic.numel() == 0:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)

        mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda")
        rendered_image, rendered_mask, radii, points2d = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = None,
            opacities = opacity,
            mask = mask,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = None)

        render_dy = torch.ones_like(rendered_image, requires_grad=True) 
        print('all static')

        # 返回渲染图像
        return {
            "render": rendered_image,
            "render_st": rendered_image.clone(),
            "render_dy": render_dy,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "deformed_points": means3D_final,  # 没有动态点
            "points2d": points2d,
            "points2d_dy":None,
            "points2d_st":points2d,
            "dx_st": None,
            "dr_st": None,
            "ds_st": None,

        }
    
    elif idx_static.numel() == 0:       # 全是动态
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, *_ = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)

        mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda")
        rendered_image, rendered_mask, radii, points2d = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = None,
            opacities = opacity,
            mask = mask,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = None)

        render_dy = rendered_image
        render_st = torch.ones_like(rendered_image, requires_grad=True) 
        print('all dynamic')

        # 返回渲染图像
        return {
            "render": rendered_image,
            "render_st": render_st,
            "render_dy": render_dy,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "deformed_points": means3D_final,  # 没有动态点
            "points2d": points2d,
            "points2d_dy": points2d,
            "points2d_st": None,
            "dx_st": None,
            "dr_st": None,
            "ds_st": None,
        }

    means3D_new = torch.zeros_like(means3D)
    r_new = torch.zeros_like(rotations)
    s_new = torch.zeros_like(scales)
    tensors_list = [means3D, scales, rotations, opacity, shs, means2D, time]
    static_tensors, dynamic_tensors = zip(*[(t[idx_static].clone(), t[idx_dynamic].clone()) for t in tensors_list])  # .clone()
    (means3D_st, scales_st, rotations_st, opacity_st, shs_st, means2D_st, time_st) = static_tensors
    (means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, means2D_dy, time_dy) = dynamic_tensors
    # means3D_dy_canonical = means3D_dy
    
    # means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, dx_dy, dr_dy, ds_dy, _ = pc._deformation(means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, time_dy)
    
    # add
    *_, dx_st, dr_st, ds_st, _ = pc._deformation(means3D_st, scales_st, rotations_st, opacity_st, shs_st, time_st)

    *_, opacity_dy, shs_dy, dx_dy, dr_dy, ds_dy, _ = pc._deformation(means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, time_dy)
    means3D_dy = dx_dy + means3D_dy # category[idx_dynamic].detach() * 
    scales_dy = ds_dy + scales_dy # category[idx_dynamic].detach() * 
    rotations_dy = dr_dy + rotations_dy     # category[idx_dynamic].detach() * 

    # 静态也在动
    means3D_st = dx_st + means3D_st
    scales_st = ds_st + scales_st
    rotations_st = dr_st + rotations_st

    means3D_new[idx_static] = means3D_st
    means3D_new[idx_dynamic] = means3D_dy

    r_new[idx_static] = rotations_st
    r_new[idx_dynamic] = rotations_dy

    s_new[idx_static] = scales_st
    s_new[idx_dynamic] = scales_dy


    scales_st, scales_dy = map(pc.scaling_activation, (scales_st, scales_dy))
    rotations_st, rotations_dy = map(pc.rotation_activation, (rotations_st, rotations_dy))
    opacity_st, opacity_dy = map(pc.opacity_activation, (opacity_st, opacity_dy))

    mask_st = torch.zeros((means3D_st.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask
    mask_dy = torch.zeros((means3D_dy.shape[0], 1), dtype=torch.float, device="cuda")
    

    # -------------------------------- seperation ----------------------------

    
    rendered_st, rendered_mask_st, radii_st, points2d_st = rasterizer(
        means3D = means3D_st,
        means2D = means2D_st,
        shs = shs_st,
        colors_precomp = None,
        opacities = opacity_st,
        mask = mask_st,
        scales = scales_st,
        rotations = rotations_st,
        cov3D_precomp = None)

    rendered_dy, rendered_mask_dy, radii_dy, points2d_dy = rasterizer(
        means3D = means3D_dy,
        means2D = means2D_dy,
        shs = shs_dy,
        colors_precomp = None,
        opacities = opacity_dy,
        mask = mask_dy,
        scales = scales_dy,
        rotations = rotations_dy,
        cov3D_precomp = None)

    means3D_final = torch.cat([means3D_dy, means3D_st], dim=0)
    scales_final = torch.cat([scales_dy, scales_st], dim=0)
    rotations_final = torch.cat([rotations_dy, rotations_st], dim=0)
    opacity_final = torch.cat([opacity_dy, opacity_st], dim=0)
    shs_final = torch.cat([shs_dy, shs_st], dim=0)
    means2D_final = torch.cat([means2D_dy,means2D_st],dim=0)
    mask_final = torch.cat([mask_dy, mask_st], dim=0)

    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D_final,      # 是否需要克隆
        shs = shs_final,    # torch.Size([281137, 16, 3])
        colors_precomp = None,
        opacities = opacity_final,
        mask = mask_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    depth, depth_st, depth_dy = None, None, None  

    # 重新排序
    N_dy = idx_dynamic.size(0)
    N_st = idx_static.size(0)
    final_to_original = torch.empty(N_dy + N_st, dtype=torch.long, device=idx_dynamic.device)
    final_to_original[:N_dy] = idx_dynamic      # 对应 dynamic
    final_to_original[N_dy:] = idx_static       # 对应 static

    radii_original = torch.empty_like(radii)
    radii_original[final_to_original] = radii

    # gsplat for rendering depth
    if is_eval == True:
        with torch.no_grad():
            focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
            K = torch.tensor(
                [
                    [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                    [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                    [0, 0, 1],
                ],
                device="cuda",
            )
            
            depth, _, _ = rasterization(
                means = means3D_final, 
                quats = rotations_final, 
                scales = scales_final, 
                opacities = opacity_final.squeeze(-1), 
                colors = shs_final,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                ) 
            

            depth_st, _, _ = rasterization(
                means = means3D_st, 
                quats = rotations_st, 
                scales = scales_st, 
                opacities = opacity_st.squeeze(-1), 
                colors = shs_st,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                ) 

            depth_dy, _, _ = rasterization(
                means = means3D_dy, 
                quats = rotations_dy, 
                scales = scales_dy, 
                opacities = opacity_dy.squeeze(-1), 
                colors = shs_dy,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                )
    

    
    return {"render": rendered_image,
            "render_st": rendered_st,
            "render_dy": rendered_dy,

            "mask": rendered_mask,
            "depth": depth,
            "depth_st": depth_st,
            "depth_dy": depth_dy,

            "viewspace_points": screenspace_points,     # means2D  用于densitify
            "visibility_filter" : radii_original > 0 & (seperate_mask == 0),   # (radii > 0) & (seperate_mask != 0)
            "radii": radii_original,

            "deformed_points": means3D_final,       # 全部的最终点做 clustering 
            "points2d": points2d,   # 每个GS在图像上的投影索引值
            "points2d_dy": points2d_dy,
            "points2d_st": points2d_st,
            # "dx_st": dx_st,
            # "dr_st": dr_st,
            # "ds_st": ds_st,
            # "means3D_dy": means3D_dy,
            # "means3D_dy_canonical": means3D_dy_canonical,
            # "rotations_dy": rotations_dy,
            "means3D_new": means3D_new,
            "r_new": r_new,
            "s_new": s_new,
            }

def render_seperate_composite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
            staic_bkg = None, time = None, index = None, idx=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        
        tanfovx = math.tan(staic_bkg.FoVx * 0.5)
        tanfovy = math.tan(staic_bkg.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(staic_bkg.image_height),
            image_width=int(staic_bkg.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=staic_bkg.world_view_transform.cuda(),    # 世界坐标系到相机坐标系的变换矩阵4x4
            projmatrix=staic_bkg.full_proj_transform.cuda(),     # 相机坐标系投影到屏幕的透视投影矩阵4x4
            sh_degree=pc.active_sh_degree,
            campos=staic_bkg.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        
    idx_dynamic = torch.arange(0, index)
    idx_static = torch.arange(index, means3D.shape[0])  # 静态是剩下的
    

    means3D_new = torch.zeros_like(means3D)
    r_new = torch.zeros_like(rotations)
    s_new = torch.zeros_like(scales)

    tensors_list = [means3D, scales, rotations, opacity, shs, means2D, time]
    static_tensors, dynamic_tensors = zip(*[(t[idx_static].clone(), t[idx_dynamic].clone()) for t in tensors_list])  # .clone()
    (means3D_st, scales_st, rotations_st, opacity_st, shs_st, means2D_st, time_st) = static_tensors
    (means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, means2D_dy, time_dy) = dynamic_tensors
    means3D_dy_canonical = means3D_dy
    
    
    *_, opacity_dy, shs_dy, dx_dy, dr_dy, ds_dy, _ = pc._deformation(means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, time_dy)
    means3D_dy = dx_dy + means3D_dy # category[idx_dynamic].detach() * 
    scales_dy = ds_dy + scales_dy # category[idx_dynamic].detach() * 
    rotations_dy = dr_dy + rotations_dy     # category[idx_dynamic].detach() * 

    means3D_new[idx_static] = means3D_st
    means3D_new[idx_dynamic] = means3D_dy

    r_new[idx_static] = rotations_st
    r_new[idx_dynamic] = rotations_dy

    s_new[idx_static] = scales_st
    s_new[idx_dynamic] = scales_dy


    scales_st, scales_dy = map(pc.scaling_activation, (scales_st, scales_dy))
    rotations_st, rotations_dy = map(pc.rotation_activation, (rotations_st, rotations_dy))
    opacity_st, opacity_dy = map(pc.opacity_activation, (opacity_st, opacity_dy))

    mask_st = torch.zeros((means3D_st.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask
    mask_dy = torch.zeros((means3D_dy.shape[0], 1), dtype=torch.float, device="cuda")
    
    # ---------------------------manual transformation------------------------
    from .transform_utils_torch import transform
    import numpy as np

    # ---------------------------------S5 背景--------------------------------------
    # scales_bias = 1
    # rotation_bias = torch.tensor([0, np.deg2rad(15), 0]).cuda()     # [,水平旋转,上下旋转]
    # motion_bias = torch.tensor([-10, 1, 0]).cuda()   # [左右 , 上下, 前后]
    # means3D_st, rotations_st, scales_st = transform(means3D_st, rotations_st, scales_st, scales_bias, motion_bias, rotation_bias) 
    # 
    # # S1 前景 
    # scales_bias = 1
    # rotation_bias = torch.tensor([0, 0, 0]).cuda() 
    # motion_bias = torch.tensor([-4, 0, -6]).cuda()   # [ , -1 人上, -前 +后]
    # ---------------------------------S5 背景--------------------------------------
    # 
    # ---------------------------------S1 背景--------------------------------------
    scales_bias = 1
    rotation_bias = torch.tensor([0, 0, 0]).cuda() 
    motion_bias = torch.tensor([-1, 0.6, -2]).cuda()   # [ -左+右, -1 人上, -前 +后]
    motion_bias = motion_bias + idx*torch.tensor([0.08,0,0]).cuda()

    means3D_dy, rotations_dy, scales_dy = transform(means3D_dy, rotations_dy, scales_dy, scales_bias, motion_bias, rotation_bias) 
    
    
    # -------------------------------- seperation ----------------------------
    means3D_final = torch.cat([means3D_dy, means3D_st], dim=0)
    scales_final = torch.cat([scales_dy, scales_st], dim=0)
    rotations_final = torch.cat([rotations_dy, rotations_st], dim=0)
    opacity_final = torch.cat([opacity_dy, opacity_st], dim=0)
    shs_final = torch.cat([shs_dy, shs_st], dim=0)
    means2D_final = torch.cat([means2D_dy,means2D_st],dim=0)
    mask_final = torch.cat([mask_dy, mask_st], dim=0)

    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D_final,      # 是否需要克隆
        shs = shs_final,    # torch.Size([281137, 16, 3])
        colors_precomp = None,
        opacities = opacity_final,
        mask = mask_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    depth, depth_st, depth_dy = None, None, None  

    
    return {"render": rendered_image,
            "mask": rendered_mask,

            "means3D_new": means3D_new,
            "r_new": r_new,
            "s_new": s_new,
            }


def render_feature(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, classifier = None, scaling_modifier = 1.0, mlp = None, dropout = -1, seperate_mask = None, category = None, is_eval = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc._scaling
    rotations = pc._rotation

    # deform
    idx_static = torch.where(seperate_mask==0)[0]
    idx_dynamic = torch.where(seperate_mask==1)[0]

    if idx_dynamic.numel() == 0:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)
        *_, kplane_feat = pc._deformation(means3D, scales, rotations, opacity, shs, time)
        identity_encoding = classifier._mlp(pc._objects_dc, time) # (means3D_final, time, shs_final)  # 

        rendered_feature_map, radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = None,
            colors_precomp = identity_encoding,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)

        # 返回渲染图像
        return {
                "render": rendered_feature_map,
                "deformed_points": means3D_final
                }
    
    # 这里只是为了进行调序，调序的目的是找到移动后的gs，获得xyz
    tensors_list = [means3D, scales, rotations, opacity, shs, means2D, time, pc._objects_dc]
    static_tensors, dynamic_tensors = zip(*[(t[idx_static].clone(), t[idx_dynamic].clone()) for t in tensors_list])  # .clone()
    (means3D_st, scales_st, rotations_st, opacity_st, shs_st, means2D_st, time_st, eid_st) = static_tensors
    (means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, means2D_dy, time_dy, eid_dy) = dynamic_tensors
    
    means3D_canonical = torch.cat([means3D_dy, means3D_st], dim=0)
    means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, *_ = pc._deformation(means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, time_dy)
    
    scales_st, scales_dy = map(pc.scaling_activation, (scales_st, scales_dy))
    rotations_st, rotations_dy = map(pc.rotation_activation, (rotations_st, rotations_dy))
    opacity_st, opacity_dy = map(pc.opacity_activation, (opacity_st, opacity_dy))

    means3D_final = torch.cat([means3D_dy, means3D_st], dim=0)
    scales_final = torch.cat([scales_dy, scales_st], dim=0)
    rotations_final = torch.cat([rotations_dy, rotations_st], dim=0)
    opacity_final = torch.cat([opacity_dy, opacity_st], dim=0)
    shs_final = torch.cat([shs_dy, shs_st], dim=0)
    means2D_final = torch.cat([means2D_dy, means2D_st],dim=0)
    eid = torch.cat([eid_dy, eid_st],dim=0)

    # *_, hidden = pc._deformation(means3D, scales, rotations, opacity, shs, time)        # 每一次输入给分类器的Hidden都是从全部gs拿到的feature
    identity_encoding = classifier._mlp(eid, time)     # 这里是canonical space中的xyz !!  (means3D_canonical, time, shs_final)
    rendered_feature_map_st = None
    rendered_feature_map_dy = None
    if is_eval == True:
        identity_encoding_static = classifier._mlp(eid_st, time_st)
        identity_encoding_dynamic = classifier._mlp(eid_dy, time_dy)

        rendered_feature_map_st, _ = rasterizer(
            means3D = means3D_st,
            means2D = means2D_st,
            shs = None,
            colors_precomp = identity_encoding_static,
            opacities = opacity_st,
            scales = scales_st,
            rotations = rotations_st,
            cov3D_precomp = cov3D_precomp)
        
        rendered_feature_map_dy, _ = rasterizer(
            means3D = means3D_dy,
            means2D = means2D_dy,
            shs = None,
            colors_precomp = identity_encoding_dynamic,
            opacities = opacity_dy,
            scales = scales_dy,
            rotations = rotations_dy,
            cov3D_precomp = cov3D_precomp)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_feature_map, radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D_final,
        shs = None,
        colors_precomp = identity_encoding,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # category_final = torch.cat([category[idx_dynamic], category[idx_static]], dim=0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_feature_map,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "deformed_points": means3D_final,
            "rendered_feature_map_st": rendered_feature_map_st,
            "rendered_feature_map_dy": rendered_feature_map_dy,
            # "category_final": category_final
            }



def tracking(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
           override_mask = None, filtered_mask = None, prob_obj3d = None, is_eval = False, max_indices=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, *_ = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
    else:
        raise NotImplementedError


    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    mask = torch.zeros((means3D_final.shape[0], 1), dtype=torch.float, device="cuda") if override_mask is None else override_mask
    
    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    if is_eval == True:
        if time[0] == 0:
            velocity = torch.norm(dx, dim=1)
            # topk = torch.topk(velocity, k=10)
            # max_indices = topk.indices  # shape: [10]
            theta_u = 80 * velocity.mean() 
            mask_dy = velocity > theta_u  

            means3D_dy, means2D_dy, shs_dy, opacities_dy, scales_dy, rotations_dy, mask_dynamic = \
            (tensor[mask_dy] for tensor in [means3D_final, means2D, shs_final, opacity, scales_final, rotations_final, mask])
        
            rendered_dy, _, _, points2d_dy  = rasterizer(
                means3D = means3D_dy,
                means2D = means2D_dy,
                shs = shs_dy,
                colors_precomp = None,
                opacities = opacities_dy,
                mask = mask_dynamic,
                scales = scales_dy,
                rotations = rotations_dy,
                cov3D_precomp = cov3D_precomp)
        
            delete_mask_dy,_ = compute_pixel_coverage(points2d_dy, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)) 
            delete_mask_dy = ~delete_mask_dy

            final_mask_dy = mask_dy.clone()
            final_mask_dy[mask_dy] &= delete_mask_dy

            max_indices = torch.where(final_mask_dy == 1)[0]
            print('\033[91m' + 'The number of dynamic points:' , max_indices.shape, '\033[0m')

        means3D_dy    = means3D_final[max_indices]
        means2D_dy    = means2D[max_indices]
        shs_dy        = shs_final[max_indices]
        opacities_dy  = opacity[max_indices]
        scales_dy     = scales_final[max_indices]
        rotations_dy  = rotations_final[max_indices]
        mask_dy       = mask[max_indices]
            
        rendered_dy, _, _, points2d_dy  = rasterizer(
            means3D = means3D_dy,
            means2D = means2D_dy,
            shs = shs_dy,
            colors_precomp = None,
            opacities = opacities_dy,
            mask = mask_dy,
            scales = scales_dy,
            rotations = rotations_dy,
            cov3D_precomp = cov3D_precomp)
        

        return {
                "points2d_max": points2d_dy,
                "max_indices": max_indices,
                "rendered_dy": rendered_dy
                }
    
    else:
        return {"render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "deformed_points": means3D_final,
            "points2d": points2d
            }


def render_seperate2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
           override_mask = None, filtered_mask = None, prob_obj3d = None, seperate_mask = None, is_eval = False, category = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),    # 世界坐标系到相机坐标系的变换矩阵4x4
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),     # 相机坐标系投影到屏幕的透视投影矩阵4x4
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be compuwobuyaoted from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation

    ####-------------------------------------------before------------------------------
    *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

    means3D_final = means3D + dx
    scales_final = scales + ds
    rotations_final = rotations + dr

    scales_final = pc.scaling_activation(scales_final)  # exp()
    rotations_final = pc.rotation_activation(rotations_final)   # normalize  正则化
    opacity_final = pc.opacity_activation(opacity)  # sigmoid
    mask = torch.zeros((means3D.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask

    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D,      # 是否需要克隆
        shs = shs,    # torch.Size([281137, 16, 3])
        colors_precomp = None,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    

    dr_normalized = dr / dr.norm(dim=1, keepdim=True).clamp(min=1e-8)
    theta = 2 * torch.acos(dr_normalized[:, 0].clamp(-1.0, 1.0))
    theta_deg = theta * 180 / torch.pi
    

    motion = dx.norm(dim=-1).unsqueeze(-1).repeat(1, 3) # theta_deg.unsqueeze(-1).repeat(1, 3) # ds.norm(dim=-1).unsqueeze(-1).repeat(1, 3)
    scale = ds.norm(dim=-1).unsqueeze(-1).repeat(1, 3)
    theta_deg = theta_deg.unsqueeze(-1).repeat(1, 3)

    rendered_motion, *_ = rasterizer(
        means3D = means3D_final,
        means2D = means2D,      # 是否需要克隆
        shs = None,    # torch.Size([281137, 16, 3])
        colors_precomp = motion,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    rendered_scale, *_ = rasterizer(
        means3D = means3D_final,
        means2D = means2D,      # 是否需要克隆
        shs = None,    # torch.Size([281137, 16, 3])
        colors_precomp = scale,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    rendered_theta_deg, *_ = rasterizer(
        means3D = means3D_final,
        means2D = means2D,      # 是否需要克隆
        shs = None,    # torch.Size([281137, 16, 3])
        colors_precomp = theta_deg,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    ######------------------------------------------------------------------
    motion_dx = dx.norm(dim=-1)>dx.norm(dim=-1).mean() # theta_deg # ds.norm(dim=-1)
    motion_ds = ds.norm(dim=-1)>ds.norm(dim=-1).mean()
    motion_dr = theta_deg>theta_deg.mean()
    seperate_mask =  motion_dx.clamp(max=1.0) 
    
    idx_static = torch.where(seperate_mask==0)[0]
    idx_dynamic = torch.where(seperate_mask==1)[0]
    tensors_list = [means3D_final, scales_final, rotations_final, opacity_final, shs, means2D, time]
    static_tensors, dynamic_tensors = zip(*[(t[idx_static].clone(), t[idx_dynamic].clone()) for t in tensors_list])  # .clone()
    (means3D_st, scales_st, rotations_st, opacity_st, shs_st, means2D_st, time_st) = static_tensors
    (means3D_dy, scales_dy, rotations_dy, opacity_dy, shs_dy, means2D_dy, time_dy) = dynamic_tensors 
    # add

    mask_st = torch.zeros((means3D_st.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask
    mask_dy = torch.zeros((means3D_dy.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask
    
    rendered_st, *_ = rasterizer(
        means3D = means3D_st,
        means2D = means2D_st,
        shs = shs_st,
        colors_precomp = None,
        opacities = opacity_st,
        mask = mask_st,
        scales = scales_st,
        rotations = rotations_st,
        cov3D_precomp = None)
    
    rendered_dy, *_ = rasterizer(
        means3D = means3D_dy,
        means2D = means2D_dy,
        shs = shs_dy,
        colors_precomp = None,
        opacities = opacity_dy,
        mask = mask_dy,
        scales = scales_dy,
        rotations = rotations_dy,
        cov3D_precomp = None)
    
    depth, depth_st, depth_dy = None, None, None  

    
    return {"render": rendered_image,
            "mask": rendered_mask,
            "depth": depth,
            "depth_st": depth_st,
            "depth_dy": depth_dy,

            "viewspace_points": screenspace_points,     # means2D  用于densitify
            "visibility_filter" : radii > 0,
            "radii": radii,

            "deformed_points": means3D_final,       # 全部的最终点做 clustering 
            "points2d": points2d,   # 每个GS在图像上的投影索引值
            "dx": dx, 
            "dr": dr, 
            "ds": ds,

            "rendered_motion": rendered_motion,
            "rendered_st": rendered_st,
            "rendered_dy": rendered_dy,
            "motion":motion,
            'rendered_scale': rendered_scale,
            'rendered_theta_deg': rendered_theta_deg
            }

def render_seperate1(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
           override_mask = None, filtered_mask = None, prob_obj3d = None, seperate_mask = None, is_eval = False, category = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),    # 世界坐标系到相机坐标系的变换矩阵4x4
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),     # 相机坐标系投影到屏幕的透视投影矩阵4x4
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be compuwobuyaoted from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    # idx_static = torch.where(seperate_mask==0)[0]
    # idx_dynamic = torch.where(seperate_mask==1)[0]

    ####-------------------------------------------before------------------------------
    *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

    means3D_final = means3D + dx.detach()
    scales_final = scales + ds.detach()
    rotations_final = rotations + dr.detach()

    # opacity= category * opacity
    scales_final = pc.scaling_activation(scales_final)  # exp()
    rotations_final = pc.rotation_activation(rotations_final)   # normalize  正则化
    opacity_final = pc.opacity_activation(opacity)  # sigmoid
    mask = torch.zeros((means3D.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask

    rendered_image, rendered_mask, radii, points2d = rasterizer(
        means3D = means3D_final,
        means2D = means2D,      # 是否需要克隆
        shs = shs,    # torch.Size([281137, 16, 3])
        colors_precomp = None,
        opacities = opacity_final,
        mask = mask,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)
    
    rendered_image_st = None
    rendered_image_st, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D.clone(),      # 是否需要克隆
        shs = shs,    # torch.Size([281137, 16, 3])
        colors_precomp = None,
        opacities = opacity_final,
        mask = mask,
        scales = pc.get_scaling,
        rotations = pc.get_rotation,
        cov3D_precomp = None)

    depth, depth_st, depth_dy = None, None, None  
    with torch.no_grad():
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )
        
        depth, _, _ = rasterization(
            means = means3D_final, 
            quats = rotations_final, 
            scales = scales_final, 
            opacities = opacity_final.squeeze(-1), 
            colors = shs,       # shs pc.get_features  torch.Size([281137, 16, 3])
            viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
            Ks = K[None], 
            backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed = False,
            sh_degree = pc.active_sh_degree,
            render_mode = 'D'
            ) 
            
    
    # gsplat for rendering depth
    if is_eval == True:
        with torch.no_grad():
            focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
            K = torch.tensor(
                [
                    [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                    [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                    [0, 0, 1],
                ],
                device="cuda",
            )
            
            depth, _, _ = rasterization(
                means = means3D_final, 
                quats = rotations_final, 
                scales = scales_final, 
                opacities = opacity_final.squeeze(-1), 
                colors = shs_final,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                ) 
            

            depth_st, _, _ = rasterization(
                means = means3D_st, 
                quats = rotations_st, 
                scales = scales_st, 
                opacities = opacity_st.squeeze(-1), 
                colors = shs_st,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                ) 

            depth_dy, _, _ = rasterization(
                means = means3D_dy, 
                quats = rotations_dy, 
                scales = scales_dy, 
                opacities = opacity_dy.squeeze(-1), 
                colors = shs_dy,       # shs pc.get_features  torch.Size([281137, 16, 3])
                viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
                Ks = K[None], 
                backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed = False,
                sh_degree = pc.active_sh_degree,
                render_mode = 'D'
                )
    

    
    return {"render": rendered_image,
            "mask": rendered_mask,
            "depth": depth,
            "depth_st": depth_st,
            "depth_dy": depth_dy,

            "viewspace_points": screenspace_points,     # means2D  用于densitify
            "visibility_filter" : radii > 0,
            "radii": radii,

            "deformed_points": means3D_final,       # 全部的最终点做 clustering 
            "points2d": points2d,   # 每个GS在图像上的投影索引值
            "dx": dx, 
            "dr": dr, 
            "ds": ds,
            "rendered_image_st": rendered_image_st,
            }

def render_feature1(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, classifier = None, scaling_modifier = 1.0, mlp = None, dropout = -1, seperate_mask = None, category = None, 
                    is_eval = False, dino = None, identity_encoding = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc._scaling
    rotations = pc._rotation

    # deform
    idx_static = torch.where(seperate_mask==0)[0]
    idx_dynamic = torch.where(seperate_mask==1)[0]
    rendered_feature_map_st = None
    rendered_feature_map_dy = None

    if idx_dynamic.numel() == 0:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity = pc.opacity_activation(opacity_final)
        # identity_encoding = classifier._mlp(means3D_final, time, shs_final)
        
        rendered_feature_map, radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = None,
            colors_precomp = identity_encoding,  # pc.get_objects.squeeze(),
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)

        # 返回渲染图像
        return {
                "render": rendered_feature_map,
                "deformed_points": means3D_final,
                "rendered_feature_map_st": rendered_feature_map_st,
                "rendered_feature_map_dy": rendered_feature_map_dy,
                }
    
    # 这里只是为了进行调序，调序的目的是找到移动后的gs，获得xyz
    
    *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

    means3D_final = means3D + dx.detach() * category.detach()
    scales_final = scales + ds.detach() * category.detach()
    rotations_final = rotations + dr.detach() * category.detach()

    scales_final = pc.scaling_activation(scales_final)  # exp()
    rotations_final = pc.rotation_activation(rotations_final)   # normalize  正则化
    opacity_final = pc.opacity_activation(opacity)  # sigmoid
    
    # identity_encoding = classifier._mlp(means3D, time, shs)     # 这里是canonical space中的xyz !!  (means3D_canonical, time, shs_final)
    # identity_encoding = pc.get_objects.squeeze()
    rendered_feature_map_st = None
    rendered_feature_map_dy = None
    if is_eval == True:
        identity_encoding_static = classifier._mlp(means3D_st, time_st) # classifier._mlp(eid_st, time_st)
        identity_encoding_dynamic = classifier._mlp(means3D_canonical[means3D_dy.shape[0]], time_dy) # classifier._mlp(eid_dy, time_dy)

        rendered_feature_map_st, _ = rasterizer(
            means3D = means3D_st,
            means2D = means2D_st,
            shs = None,
            colors_precomp = identity_encoding_static,
            opacities = opacity_st,
            scales = scales_st,
            rotations = rotations_st,
            cov3D_precomp = cov3D_precomp)
        
        rendered_feature_map_dy, _ = rasterizer(
            means3D = means3D_dy,
            means2D = means2D_dy,
            shs = None,
            colors_precomp = identity_encoding_dynamic,
            opacities = opacity_dy,
            scales = scales_dy,
            rotations = rotations_dy,
            cov3D_precomp = cov3D_precomp)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 

    valid_mask = (category<0.5).squeeze()
    rendered_feature_map_st, radii = rasterizer(
        means3D = means3D_final[valid_mask],
        means2D = means2D[valid_mask],
        shs = None,
        colors_precomp = identity_encoding[valid_mask],
        opacities = opacity_final[valid_mask],
        scales = scales_final[valid_mask],
        rotations = rotations_final[valid_mask],
        cov3D_precomp = None)
    
    rendered_feature_map_dy, radii = rasterizer(
        means3D = means3D_final[~valid_mask],
        means2D = means2D[~valid_mask],
        shs = None,
        colors_precomp = identity_encoding[~valid_mask],
        opacities = opacity_final[~valid_mask],
        scales = scales_final[~valid_mask],
        rotations = rotations_final[~valid_mask],
        cov3D_precomp = None)
    
    rendered_feature_map, radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = identity_encoding, # pc._objects_dc,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = None)

    # category_final = torch.cat([category[idx_dynamic], category[idx_static]], dim=0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_feature_map,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "deformed_points": means3D_final,
            "rendered_feature_map_st": rendered_feature_map_st,
            "rendered_feature_map_dy": rendered_feature_map_dy,
            # "category_final": category_final
            "dx": dx, 
            "dr": dr, 
            "ds": ds
            }

def render_depth_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cam_type=None,
        time = None, points2d = None, category = None, static_mask = None, dynamic_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        if time == None:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        else:
            time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        
    
    *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

    means3D_final = means3D + dx.detach() * category
    scales_final = scales + ds.detach() * category
    rotations_final = rotations + dr.detach() * category

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity)
    
    
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    depth, _, _ = rasterization(
            means = means3D_final, 
            quats = rotations_final, 
            scales = scales_final, 
            opacities = opacity_final.squeeze(-1), 
            colors = shs,       # shs pc.get_features  torch.Size([281137, 16, 3])
            viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda(), 
            Ks = K[None], 
            backgrounds=torch.tensor([1., 1., 1.], device='cuda:0')[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed = False,
            sh_degree = pc.active_sh_degree,
            render_mode = 'D'
            ) 
    
    ones = torch.ones((means3D_final.shape[0], 1), dtype=means3D_final.dtype, device=means3D_final.device)
    p_orig1 = torch.cat([means3D_final, ones], dim=1)
    gs_z = (viewpoint_camera.world_view_transform.cuda().T[:3,:] @ p_orig1.T).T

    u, v = points2d[:, 0], points2d[:, 1] 
    valid_uv_mask = (u >= 0) & (u < depth.shape[2]) & (v >= 0) & (v < depth.shape[1])
    valid_z_mask = (gs_z[:, -1] > 0.5)
    combined_mask = valid_uv_mask & valid_z_mask

    u_valid = u[combined_mask].long()
    v_valid = v[combined_mask].long()
    gs_z_valid = gs_z[combined_mask, -1]
    depth_valid = depth[0, v_valid, u_valid, 0]
    depth_mask = (gs_z_valid <= depth_valid + 5)


    threshold_st = torch.quantile(static_mask, 0.9)
    high_value_mask = static_mask[0] >= threshold_st # # torch.Size([1, 960, 536])
    high_value_hits_st = high_value_mask[v_valid, u_valid]

    threshold_dy = torch.quantile(dynamic_mask, 0.1)
    high_value_mask = dynamic_mask[0] <= threshold_dy
    high_value_hits_dy = high_value_mask[v_valid, u_valid]
    
    # 5. 组合条件（depth & static region）
    final_sub_mask = depth_mask & (high_value_hits_st | high_value_hits_dy)  # [M]

    # 6. 构造 full 高斯掩码
    final_mask = torch.zeros_like(gs_z[:, -1]).float().cuda()  # [N]
    final_mask[combined_mask] = final_sub_mask.float()

    return final_mask

def render_probs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, classifier = None, scaling_modifier = 1.0, mlp = None, dropout = -1, seperate_mask = None, category = None, 
                    is_eval = False, dino = None, identity_encoding = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕

    with torch.no_grad():
        means2D = screenspace_points
        opacity = pc._opacity
        shs = pc.get_features

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = pc._scaling
        rotations = pc._rotation
        
        *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

        means3D_final = means3D + dx.detach() #* category
        scales_final = scales + ds.detach() #* category
        rotations_final = rotations + dr.detach() #* category

        scales_final = pc.scaling_activation(scales_final)  # exp()
        rotations_final = pc.rotation_activation(rotations_final)   # normalize  正则化
        opacity_final = pc.opacity_activation(opacity)  # sigmoid
    
        mask = torch.zeros((means3D.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask

    rendered_prob, *_ = rasterizer(
        means3D = means3D_final.detach(),
        means2D = means2D.detach(),      # 是否需要克隆
        shs = None,    # torch.Size([281137, 16, 3])
        colors_precomp = category.repeat(1, 3),
        opacities = opacity_final.detach(),
        mask = mask.detach(),
        scales = scales_final.detach(),
        rotations = rotations_final.detach(),
        cov3D_precomp = None)
    
    
    return {
            "rendered_prob": rendered_prob,
            "dx": dx, 
            "dr": dr, 
            "ds": ds,
            }

def render_probs1(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, classifier = None, scaling_modifier = 1.0, mlp = None, dropout = -1, seperate_mask = None, category = None, 
                    is_eval = False, dino = None, identity_encoding = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 实例化一个光栅化器，将高斯点投影到屏幕

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    
    *_, dx, dr, ds, _ = pc._deformation(means3D, scales, rotations, opacity, shs, time)

    means3D_final = means3D + dx.detach() * category
    scales_final = scales + ds.detach() * category
    rotations_final = rotations + dr.detach() * category

    scales_final = pc.scaling_activation(scales_final)  # exp()
    rotations_final = pc.rotation_activation(rotations_final)   # normalize  正则化
    opacity_final = pc.opacity_activation(opacity)  # sigmoid
    
    mask = torch.zeros((means3D.shape[0], 1), dtype=torch.float, device="cuda") # if override_mask is None else override_mask

    rendered_prob, *_ = rasterizer(
        means3D = means3D_final.detach(),
        means2D = means2D.detach(),      # 是否需要克隆
        shs = None,    # torch.Size([281137, 16, 3])
        colors_precomp = category.repeat(1, 3),
        opacities = opacity_final.detach(),
        mask = mask.detach(),
        scales = scales_final.detach(),
        rotations = rotations_final.detach(),
        cov3D_precomp = None)
    
    valid_mask = (category<0.5).squeeze()
    rendered_feature_map_st,*_ = rasterizer(
        means3D = means3D_final[valid_mask],
        means2D = means2D[valid_mask],
        shs = None,
        colors_precomp = category.repeat(1, 3)[valid_mask],
        opacities = opacity_final[valid_mask],
        mask = mask[valid_mask],
        scales = scales_final[valid_mask],
        rotations = rotations_final[valid_mask],
        cov3D_precomp = None)
    
    rendered_feature_map_dy,*_ = rasterizer(
        means3D = means3D_final[~valid_mask],
        means2D = means2D[~valid_mask],
        shs = None,
        colors_precomp = category.repeat(1, 3)[~valid_mask],
        opacities = opacity_final[~valid_mask],
        mask = mask[~valid_mask],
        scales = scales_final[~valid_mask],
        rotations = rotations_final[~valid_mask],
        cov3D_precomp = None)
    

    return {
            "render": rendered_prob,
            "rendered_feature_map_st": rendered_feature_map_st,
            "rendered_feature_map_dy": rendered_feature_map_dy,

            }


