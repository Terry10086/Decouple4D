def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None,
            override_mask = None, filtered_mask = None, prob_obj3d = None, is_eval = False, time = None):
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
        
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, *_ = pc._deformation(means3D, scales, rotations, opacity, shs, time)
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
            "points2d": points2d}
