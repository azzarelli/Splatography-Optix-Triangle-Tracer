import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn.functional as F
from gsplat.rendering import rasterization
from utils.loss_utils import l1_loss


def quaternion_rotate(q, v):
    q_vec = q[:, :3]
    q_w = q[:, 3].unsqueeze(1)
    t = 2 * torch.cross(q_vec, v, dim=1)
    return v + q_w * t + torch.cross(q_vec, t, dim=1)

def rotated_softmin_axis_direction(r, s, temperature=10.0):
    # s: (N, 3), we want the direction of the smallest abs scale
    abs_s = torch.abs(s)

    # Step 1: Compute softmin weights (lower abs(s) => higher weight)
    weights = F.softmax(-abs_s * temperature, dim=1)  # (N, 3)

    # Step 2: Basis axes: x, y, z
    basis = torch.eye(3, device=s.device).unsqueeze(0)  # (1, 3, 3)

    # Step 3: Weighted sum of basis vectors
    soft_axis = torch.bmm(weights.unsqueeze(1), basis.repeat(s.size(0), 1, 1)).squeeze(1)  # (N, 3)

    # Step 4: Rotate the direction
    rotated = quaternion_rotate(r, soft_axis)  # (N, 3)

    return rotated


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           stage="fine", view_args=None, sources=None, optix_runner=None):
    """
    Render the scene.
    """

    extras = None
    
    try:
        mask_parser = "foreground" if view_args['show_mask'] == 1 else ""
    except:
        mask_parser=""
        
    if view_args is not None:
        if view_args['finecoarse_flag'] == False:
            means3D, scales, rotation, opacity, colors = pc.static_pass(select=mask_parser)
        else:       
            means3D, scales, rotation, opacity, colors = pc.deformation_pass(viewpoint_camera.time, select=mask_parser)

    else:
        means3D, scales, rotation, opacity, colors = pc.deformation_pass(viewpoint_camera.time, select=mask_parser)

    if view_args is not None and stage != 'test':
        pass
    else:
        view_args= {'vis_mode':'render'}
    
    rendered_image, rendered_depth, norms = None, None, None
    if view_args['vis_mode'] in ['render']:
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        means3D = means3D[mask]
        rotation = rotation[mask]
        scales = scales[mask]
        opacity = opacity[mask]
        colors = colors[mask]
        
        rendered_image, alpha, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,

            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            near_plane=0.3,

            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree,
            render_mode='RGB'
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        
        # if view_args['lighting']:
        #     pointsrc = sources[0]
        #     # Modify Colors og Gaussians based on Source Positions and behaviours
        #     origins, directions = generate_pointsrc_rays(pointsrc.get_xyz)
            
        #     colors_l = shadows_from_rays(pc, means3D, scales, rotation, colors, origins, directions, optix_runner)
            
        #     # Append sources 
        #     means3D, scales, rotation, colors_l, opacity = pointsrc.full_scene_construction(means3D, scales, rotation, colors_l, opacity)

            
        #     col_img, _, _ = rasterization(
        #         means3D, rotation, scales, opacity.squeeze(-1), colors_l,

        #         viewpoint_camera.w2c.unsqueeze(0).cuda(), 
        #         viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
        #         viewpoint_camera.image_width, 
        #         viewpoint_camera.image_height,
                
        #         rasterize_mode='antialiased',
        #         eps2d=0.1,
        #         sh_degree=pc.active_sh_degree,
        #         render_mode='RGB+D'
        #     )
        #     int_map = col_img.squeeze(0).permute(2,0,1)[0, ...].unsqueeze(0).unsqueeze(0) # 1,1,H,W
        #     depth_img = col_img.squeeze(0).permute(2,0,1)[-1, ...].unsqueeze(0).unsqueeze(0) # 1,1,H,W
            
        #     lighting = guided_filter(int_map, depth_img).squeeze(0).repeat(3,1,1)
            

        #     rendered_image = rendered_image * lighting + lighting*0.5
            

    elif view_args['vis_mode'] == 'alpha':
        _, rendered_image, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='RGB',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'D':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='D',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
        
    elif view_args['vis_mode'] == 'ED':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='ED',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'norms':
        norms = rotated_softmin_axis_direction(rotation, scales)

        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),norms,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)

    elif view_args['vis_mode'] == 'xyz':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),means3D,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_1':
        residual = torch.norm(means3D-means3D_, dim=-1).unsqueeze(-1).repeat(1,3)
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_3':
        residual = (means3D-means3D_).abs()
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)

    return {
        "render": rendered_image,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }



def render_coarse_batch(
    viewpoint_cams, pc, pipe, bg_color: torch.Tensor,scaling_modifier=1.0,
    stage="fine", iteration=0,kernel_size=0.1):
    """
    Render the scene.
    """
    means3D = pc.get_xyz    
    scales = pc.get_scaling_with_3D_filter
    rotations = pc.rotation_activation(pc._rotation)
    # colors = pc.get_color
    colors = pc.get_features
    opacity = pc.get_fine_opacity_with_3D_filter(pc.get_hopac)
    
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        rgb, _, _ = rasterization(
            means3D, rotations, scales, 
            opacity.squeeze(-1),colors,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            near_plane=0.3,

            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree,
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        
        # Train the backgroudn
        gt_img = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() > 0. # invert binary mask
        inv_mask = 1. - mask.float() 
        gt = gt_img * inv_mask
        # Blue gt and fill mask regions with this
        kernel_size = 51
        kernel = torch.ones((3, 1, kernel_size, kernel_size), dtype=gt.dtype, device=gt.device)
        kernel /= kernel_size * kernel_size

        # Apply depthwise convolution (groups=3 for 3 channels)
        blurred = F.conv2d(gt.unsqueeze(0) , kernel, padding=kernel_size//2, groups=3).squeeze(0)
        mask = mask.unsqueeze(0).repeat(3,1,1)
        gt[mask] = blurred[mask]

        L1 += l1_loss(rgb[:, 100:-100, 100:-100], gt[:, 100:-100, 100:-100])
        # L1 += l1_loss(rgb, gt)
    
    return  L1

def render_coarse_batch_foreground(viewpoint_cams, pc):
    """Render the foreground in batch form
    """
    means3D, scales, rotation, opacity, colors = pc.static_pass(select="foreground")
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        background = torch.rand(3).cuda()
        
        rgb, _, _ = rasterization(
            means3D, rotation, scales, 
            opacity.squeeze(-1), colors,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.foreground.active_sh_degree,
            backgrounds=background
        )
        
        rgb = rgb.squeeze(0).permute(2,0,1)
                
        gt = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() # > 0. # invert binary mask
        
        gt = gt*mask + (1.-mask)*background.unsqueeze(-1).unsqueeze(-1)
        
        L1 += l1_loss(rgb, gt)
    return  L1



def render_batch(
    viewpoint_cams, pc, datasettype):
    """
    Render the scene.
    """

    L1 = 0.

    for idx, viewpoint_camera in enumerate(viewpoint_cams):  
        means3D_final, scales_final, rotations_final, opacity_final, colors_final = pc.deformation_pass(viewpoint_camera.time)

        background = torch.rand(3).cuda()
        # Set up rasterization configuration
        rgb, alpha, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1), colors_final,
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            near_plane=0.3,
            sh_degree=pc.active_sh_degree,
            backgrounds=background

        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        gt_img = viewpoint_camera.original_image.cuda()

        L1 += l1_loss(rgb[:, 100:-100, 100:-100], gt_img[:, 100:-100, 100:-100])

    return L1


def render_depth_batch(
    viewpoint_cams, canon_cams,
    pc
    ):
    """
    Render the scene.
    """
    means3D = pc.get_xyz.detach()
    scales = pc.get_scaling_with_3D_filter.detach()
    rotations = pc._rotation.detach()
    colors = pc.get_features.detach()
    opacity = pc.get_opacity.detach()
    
    L1 = 0.

    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for viewpoint_camera, canon_camera in zip(viewpoint_cams, canon_cams):
        time = time*0. +viewpoint_camera.time
        
        # Render canon depth
        with torch.no_grad():
            distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
            mask = distances > 0.3

            means3D_final = means3D[mask]
            rotations_final = rotations[mask]
            scales_final = scales[mask]
            opacity_final = pc.get_coarse_opacity_with_3D_filter[mask].detach()
            colors_final = colors[mask]
            
            D, _, _ = rasterization(
                means3D_final, rotations_final, scales_final, 
                opacity_final.squeeze(-1),colors_final,
                viewpoint_camera.w2c.unsqueeze(0).cuda(), 
                viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
                viewpoint_camera.image_width, 
                viewpoint_camera.image_height,
                
                render_mode='D',
                rasterize_mode='antialiased',
                eps2d=0.1,
                sh_degree=pc.active_sh_degree
            )
            D = D.squeeze(0).permute(2,0,1)

        # Deform for current time step
        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
        )
        opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)        
        rotations_final = pc.rotation_activation(rotations_final)
        
        # Filter near-camera 3D viewpointss
        distances = torch.norm(means3D_final - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        means3D_final = means3D_final[mask]
        rotations_final = rotations_final[mask]
        scales_final = scales[mask]
        opacity_final = opacity_final[mask]
        colors_final = colors_final[mask]

        # Set up rasterization configuration
        D_t, _, _ = rasterization(
            means3D_final, rotations_final.detach(), scales_final.detach(), 
            opacity_final.squeeze(-1).detach(),colors_final.detach(),
            viewpoint_camera.w2c.unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='D',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        
        D_t = D_t.squeeze(0).permute(2,0,1)
        Q = (D-D_t).abs()

        Q = (Q - Q.min())/ (Q.max() - Q.min())
        Q_inv = 1. - Q
        with torch.no_grad():
            I_t = viewpoint_camera.original_image.cuda()
            I = canon_camera.original_image.cuda()
            P = (I-I_t).abs()
            P = (P - P.min())/(P.max() - P.min())
        
        L1 += (P*Q_inv).mean()
            
    
    return L1

def render_motion_point_mask(pc):
    """
    Render the scene.
    """
    means3D_collection = []
    for i in range(10):
        time = float(i)*0.1

        means3D_final,_,_,_,_ = pc.deform(time)
        

        
        means3D_collection.append(means3D_final.unsqueeze(0))
    
    means3D_collection = torch.cat(means3D_collection, dim=0) # K, N, 3, where K=10 for each time step
    displacement = ((means3D_collection - means3D_collection.mean(dim=0))**2).sum(dim=2).sqrt()  # K, N, 3
    motion_metric = displacement.mean(dim=0) # shape (N,)

    threshold = torch.quantile(motion_metric, 0.9)

    mask = (motion_metric >= threshold)

    return mask

def process_Gaussians(pc, time):
    means3D = pc.get_xyz
    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation
    # colors = pc.get_color
    colors = pc.get_features

    opacity = pc.get_opacity

    means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
        point=means3D, 
        rotations=rotations,
        scales = scales,
        times_sel=time, 
        h_emb=opacity,
        shs=colors,
    )
            
    opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)        
    rotations_final = pc.rotation_activation(rotations_final)
        
    return means3D_final, rotations_final, opacity_final, colors_final, scales

from utils.sh_utils import SH2RGB, eval_sh
from gaussian_renderer.ray_tracer import RaycastSTE

@torch.no_grad 
def render_triangles(viewpoint_camera, pc, optix_runner):
    """
    Render the scene for viewing
    """
    time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1)
    means3D, rotation, opacity, colors, scales = process_Gaussians(pc, time)
    
    x, d = viewpoint_camera.generate_rays()
    
    mag, dirs = pc.get_covmat
        
    cam_pos = viewpoint_camera.camera_center.to(means3D.device)  # you may need to adapt this
    view_dirs = means3D - cam_pos[None, :]
    view_dirs = view_dirs / (torch.linalg.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

    motion_mask = pc.target_mask
    
    verts, colors_v = generate_triangles(means3D, mag, dirs, colors, opacity, view_dirs, motion_mask)
    verts = verts.detach()
    N = 4
    # Forward through runner
    buffer_image = optix_runner(x, d, N, colors_v, verts, False)

    return buffer_image

def generate_triangles(means, mag, dirs, colors, opacity, view_dirs, motion_mask, thresh=0.05, scale_factor=1.5):
    """
    means:   (N,3)
    mag:     (N,2)          extents along the two in-plane axes
    dirs:    (N,2,3)        two in-plane direction vectors (should be unit)
    colors:  (N,16,3)       SH coeffs
    opacity: (N,1) or (N,)
    returns:
        verts_flat: (K*4*3, 3)   flattened triangle vertices
        tri_rgb:    (K, 3)     per-triangle RGB from SH DC
    """
    device, dtype = means.device, means.dtype

    mask = motion_mask
    means  = means[mask]          # (K,3)
    mag    = mag[mask]            # (K,2)
    dirs   = dirs[mask]           # (K,2,3)
    rgb = colors[mask] 

    # Normalize dirs to avoid scale bugs
    dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    # 4 sign combos (corners)
    signs = torch.tensor([[ 1,  1],
                          [-1,  1],
                          [-1, -1],
                          [ 1, -1]], device=device, dtype=dtype) * scale_factor  # (4,2)

    # Corner points: mean + half_extent*(s0*mag0*dir0 + s1*mag1*dir1)
    corner_offsets = (
        (signs[None, :, :, None] * mag[:, None, :, None] * dirs[:, None, :, :]).sum(dim=2)
    )  # (K,4,3)
    corners = means[:, None, :] + corner_offsets  # (K,4,3)

    # Build 4 triangles around center: (center, corner_i, corner_{i+1})
    corners_next = torch.roll(corners, shifts=-1, dims=1)        # (K,4,3)
    centers = means[:, None, :].expand(-1, 4, -1)                # (K,4,3)

    tris = torch.stack([centers, corners, corners_next], dim=2)  # (K,4,3,3)

    # Flatten verts like your original code expects
    verts_flat = tris.reshape(-1, 3)  # (K*4*3,3)

    tri_rgb = eval_sh(3, rgb.permute(0,2,1), view_dirs[mask])
    tri_rgb = (tri_rgb + 0.5).clamp(0.0, 1.0)

    return verts_flat, tri_rgb

def generate_triangles_plain(means, mag, dirs, scale_factor=1.5):
    """
    means:   (N,3)
    mag:     (N,2)          extents along the two in-plane axes
    dirs:    (N,2,3)        two in-plane direction vectors (should be unit)
    colors:  (N,16,3)       SH coeffs
    opacity: (N,1) or (N,)
    returns:
        verts_flat: (K*4*3, 3)   flattened triangle vertices
        tri_rgb:    (K, 3)     per-triangle RGB from SH DC
    """
    device, dtype = means.device, means.dtype


    # Normalize dirs to avoid scale bugs
    dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    # 4 sign combos (corners)
    signs = torch.tensor([[ 1,  1],
                          [-1,  1],
                          [-1, -1],
                          [ 1, -1]], device=device, dtype=dtype) * scale_factor  # (4,2)

    # Corner points: mean + half_extent*(s0*mag0*dir0 + s1*mag1*dir1)
    corner_offsets = (
        (signs[None, :, :, None] * mag[:, None, :, None] * dirs[:, None, :, :]).sum(dim=2)
    )  # (K,4,3)
    corners = means[:, None, :] + corner_offsets  # (K,4,3)

    # Build 4 triangles around center: (center, corner_i, corner_{i+1})
    corners_next = torch.roll(corners, shifts=-1, dims=1)        # (K,4,3)
    centers = means[:, None, :].expand(-1, 4, -1)                # (K,4,3)

    tris = torch.stack([centers, corners, corners_next], dim=2)  # (K,4,3,3)

    # Flatten verts like your original code expects
    verts_flat = tris.reshape(-1, 3)  # (K*4*3,3)

    return verts_flat


def generate_pointsrc_rays(light_pos, N=10000000, device='cuda'):
    g = torch.Generator(device=device)
    g.manual_seed(0)
    directions = torch.randn(N, 3, device=device, generator=g)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    origins = light_pos.view(1, 3).expand(N, 3)
    return origins, directions


@torch.no_grad 
def shadows_from_rays(pc, means3D, scales, rotation, colors, x, d, optix_runner):
    """
    Render the scene for viewing
    """    
    mag, dirs = pc.get_covmat_ip(rotation, scales)
        
    verts = generate_triangles_plain(means3D, mag, dirs, scale_factor=5.)
    verts = verts.detach()
    
    N = 4   
    hit_idxs = optix_runner.light_trace(x, d, N, verts).squeeze(0)

    # colors *=0

    # Distance based relighting
    lpos = x.mean(0).unsqueeze(0)
    dist2 = ((means3D - lpos) ** 2).sum(dim=-1)          # (G,)
    falloff = 1.0 / (dist2 + 1e-4)
    strength = 1.0
    
    I = 1.0          # luminous intensity scale
    eps = 1e-4
    falloff = (I / (dist2 + eps)).clamp(0.0, 1.0)

    falloff = (strength * falloff).clamp(0.0, 1.0)
        
    Y0 = 0.282095  # SH constant basis
    dc_white =  0.5 / Y0   # ~ +1.772
    dc_black = -0.5 / Y0   # ~ -1.772

    colors_l = torch.zeros_like(colors)

    # If colors is [N, 3*(L+1)^2] flattened per channel, this needs channel-aware indexing.
    # If it's [N, C, (L+1)^2], then DC is [:, :, 0].
    # Below assumes [N, 3, K] (common)
    colors_l[:, 1:] = 0.0
    colors_l[:, 0]  = dc_black          # make EVERYTHING black
    dc_hit = dc_black + (dc_white - dc_black) * falloff  # (G,)
    colors_l[hit_idxs, 0, :] = dc_hit[hit_idxs].unsqueeze(-1)
    return colors_l


def joint_bilateral_intensity(intensity, depth, k=7, sigma_s=20.0, sigma_d=0.001):
    """
    intensity: (1,1,H,W) float
    depth:     (1,1,H,W) float (meters or normalized)
    """
    pad = k // 2
    I = F.pad(intensity, (pad, pad, pad, pad), mode='reflect')
    D = F.pad(depth,     (pad, pad, pad, pad), mode='reflect')

    # (1, k*k, H*W)
    I_p = F.unfold(I, kernel_size=k)
    D_p = F.unfold(D, kernel_size=k)

    # center depth: (1,1,H*W)
    D0 = depth.view(1, 1, -1)

    # depth weights
    w_d = torch.exp(-0.5 * ((D_p - D0) / sigma_d) ** 2)

    # spatial weights (precompute)
    yy, xx = torch.meshgrid(
        torch.arange(-pad, pad+1, device=intensity.device),
        torch.arange(-pad, pad+1, device=intensity.device),
        indexing='ij'
    )
    w_s = torch.exp(-0.5 * (xx**2 + yy**2) / (sigma_s**2)).reshape(1, -1, 1)

    w = w_s * w_d
    out = (w * I_p).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True).clamp_min(1e-8))

    return out.view_as(intensity)

def _box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Fast box filter using avg_pool2d.
    x: (B, C, H, W)
    r: radius
    """
    k = 2 * r + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r)

@torch.no_grad()
def guided_filter(
                src: torch.Tensor,
                guide: torch.Tensor,
                r: int = 10,
                eps: float = 1e-3) -> torch.Tensor:
    """
    Guided filter for single-channel guide and src.

    guide: (B,1,H,W)   e.g., depth (recommended: normalized or log-depth)
    src:   (B,1,H,W)   intensity map to smooth
    r:     radius in pixels
    eps:   regularization (bigger => more smoothing, less edge following)

    returns: (B,1,H,W)
    """
    assert guide.ndim == 4 and src.ndim == 4
    assert guide.shape[:2] == (src.shape[0], 1) and src.shape[1] == 1
    assert guide.shape[-2:] == src.shape[-2:]

    I = guide
    p = src

    mean_I = _box_filter(I, r)
    mean_p = _box_filter(p, r)
    mean_Ip = _box_filter(I * p, r)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    q = mean_a * I + mean_b
    return q

