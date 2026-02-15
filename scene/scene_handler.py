import torch
import numpy as np
import open3d as o3d
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
import os
from plyfile import PlyData, PlyElement

from scene.old_deformation import deform_network as oldDeformationNet

from scene.gaussians.foreground import ForegroundGaussians
from scene.gaussians.background import BackgroundGaussians


class SceneHandler:
    
    def __init__(self, args):
        self.foreground = ForegroundGaussians(args)
        self.background = BackgroundGaussians(args)
        self.max_sh_degree = 3
        self.active_sh_degree = 3

    
    def deformation_pass(self, time, select=""):
        if select in ["foreground", ""]:
            means3D_fg, scales_fg, rotations_fg, opacity_fg, colors_fg = self.foreground.deform(time)
            if select == "foreground": return means3D_fg, scales_fg, rotations_fg, opacity_fg, colors_fg
        if select in ["background", ""]:
            means3D_bg, scales_bg, rotations_bg, opacity_bg, colors_bg = self.background.deform(time)
            if select == "background": return means3D_bg, scales_bg, rotations_bg, opacity_bg, colors_bg

        return torch.cat([means3D_fg, means3D_bg], dim=0), \
                torch.cat([scales_fg, scales_bg], dim=0), \
                torch.cat([rotations_fg, rotations_bg], dim=0), \
                    torch.cat([opacity_fg, opacity_bg], dim=0), \
                    torch.cat([colors_fg, colors_bg], dim=0)
                    
    def static_pass(self, select=""):
        
        if select in ["foreground", ""]:
            means3D_fg, scales_fg, rotations_fg, opacity_fg, colors_fg = self.foreground.nondeform()
            if select == "foreground": return means3D_fg, scales_fg, rotations_fg, opacity_fg, colors_fg
        
        if select in ["background", ""]:
            means3D_bg, scales_bg, rotations_bg, opacity_bg, colors_bg = self.background.nondeform() 
            if select == "background": return means3D_bg, scales_bg, rotations_bg, opacity_bg, colors_bg

        return torch.cat([means3D_fg, means3D_bg], dim=0), \
                torch.cat([scales_fg, scales_bg], dim=0), \
                torch.cat([rotations_fg, rotations_bg], dim=0), \
                    torch.cat([opacity_fg, opacity_bg], dim=0), \
                    torch.cat([colors_fg, colors_bg], dim=0)
                        
    
    def global_capture(self):
        return self.foreground.capture() + self.background.capture()
    
    def save_deformations(self, path):
        self.foreground.save_deformation(path)
        self.background.save_deformation(path)
    
    def load_deformations(self, path):
        self.foreground.load_model(path)
        self.background.load_model(path)
    
    def save_plys(self, path):
        self.foreground.save_ply(os.path.join(path, "foreground_scene.ply"))
        self.background.save_ply(os.path.join(path, "background_scene.ply"))
        
    def load_plys(self, path):
        check_files = os.listdir(path)
        if "point_cloud.ply" in check_files:
            chkpt_idx = path.split('_')[-1]
            self.load_from_old_model(path.replace(f"/point_cloud/iteration_{chkpt_idx}", ""), os.path.join(path, "point_cloud.ply"), chkpt_idx)
            return "oldmodel"
        else:
            self.foreground.load_ply(os.path.join(path, "foreground_scene.ply"))
            self.background.load_ply(os.path.join(path, "background_scene.ply"))

        return None

    def load_from_old_model(self, path_base, path, chkpt_idx):            
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        opac_names = sorted(opac_names, key = lambda x: int(x.split('_')[-1]))
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        self.active_sh_degree = self.max_sh_degree

        
        (model_params, first_iter) = torch.load(f'{path_base}/chkpnt_fine_{chkpt_idx}.pth')
        (_,_,_,_,_,_,_,_,_,filter_3D,_,_,_, target_mask) = model_params

        xyz = torch.from_numpy(xyz).cuda().float()
        scales = torch.from_numpy(scales).cuda().float()
        rots = torch.from_numpy(rots).cuda().float()
        opacities = torch.from_numpy(opacities).cuda().float()
        features_dc = torch.from_numpy(features_dc).cuda().permute(0,2,1).float()
        features_extra = torch.from_numpy(features_extra).cuda().permute(0,2,1).float()

        self.foreground.initialize(xyz[target_mask], scales[target_mask], rots[target_mask], opacities[target_mask], features_dc[target_mask], features_extra[target_mask])  
        self.foreground.filter_3D = filter_3D[target_mask]
        
        target_mask = ~target_mask
        self.background.initialize(xyz[target_mask], scales[target_mask], rots[target_mask], opacities[target_mask], features_dc[target_mask], features_extra[target_mask])  
        self.background.filter_3D = filter_3D[target_mask]
        
        
        weight_dict = torch.load(path.replace("point_cloud.ply", "deformation.pth"), map_location="cuda")
        oldDef = oldDeformationNet()
        oldDef.load_state_dict(weight_dict)
        oldDef = oldDef.to("cuda")
        
        # Update grids
        self.foreground._deformation.deformation_net.grid = oldDef.deformation_net.grid
        self.background._deformation.deformation_net.grid = oldDef.deformation_net.background_grid
  
        # Update MLP heads      
        self.foreground._deformation.deformation_net.spacetime_enc = oldDef.deformation_net.spacetime_enc
        self.foreground._deformation.deformation_net.pos_coeffs = oldDef.deformation_net.pos_coeffs
        self.foreground._deformation.deformation_net.rotations_deform = oldDef.deformation_net.rotations_deform
        self.foreground._deformation.deformation_net.shs_deform = oldDef.deformation_net.shs_deform
        
        self.background._deformation.deformation_net.spacetime_enc = oldDef.deformation_net.background_spacetime_enc
        self.background._deformation.deformation_net.pos_coeffs = oldDef.deformation_net.background_pos_coeffs
        
        self.foreground._deformation.deformation_net.is_old_model = True
        self.background._deformation.deformation_net.is_old_model = True

    def create_scene_from_pointcloud(self, pcd : BasicPointCloud):
        """Designed for handling the ViVO dataset
        """
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        target_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
        # Pre-defined corners from the ViVo dataset (theres one for each scene butre-using the same one doesnt cause problems)
        CORNERS = [[-1.38048, -0.1863],[-0.7779, 1.6705], [1.1469, 1.1790], [0.5832, -0.7245]]
        polygon = np.array(CORNERS)  # shape (4, 2)
        from matplotlib.path import Path
        path = Path(polygon)
        points_xy = fused_point_cloud[:, 1:].cpu().numpy()  # (N, 2)
        # Create mask for points inside polygon
        viable = torch.from_numpy(path.contains_points(points_xy)).cuda()
        
        pcds = fused_point_cloud[~viable].cpu().numpy().astype(np.float64)
        cols = fused_color[~viable].cpu().numpy().astype(np.float64)
        
        # Re-sample point cloud
        target = fused_point_cloud[viable]
        target_col = fused_color[viable]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcds)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        
        voxel_size = 0.02  # Adjust based on your data scale
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Convert back to PyTorch tensor
        bck_pcds = torch.tensor(np.asarray(downsampled_pcd.points), dtype=fused_point_cloud.dtype).cuda()
        bck_cols = torch.tensor(np.asarray(downsampled_pcd.colors), dtype=fused_color.dtype).cuda()
        
        pcds = bck_pcds
        cols = bck_cols
        
        fused_point_cloud = torch.cat([pcds, target], dim=0)
        fused_color = torch.cat([cols, target_col], dim=0)
        target_mask = torch.zeros((fused_color.shape[0], 1)).cuda()
        target_mask[cols.shape[0]:, :] = 1
        target_mask = (target_mask > 0.).squeeze(-1)
        
        err = 0.05
        
        # Need to set aabb for each deformation
        xyz_min = fused_point_cloud[target_mask].min(0).values - err
        xyz_max = fused_point_cloud[target_mask].max(0).values + err
        self.foreground.set_aabb(xyz_max, xyz_min)
        
        xyz_min = fused_point_cloud[~target_mask].min(0).values
        xyz_max = fused_point_cloud[~target_mask].max(0).values
        self.background.set_aabb(xyz_max, xyz_min)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud[target_mask]), 0.00000000001)
        dist2_else = torch.clamp_min(distCUDA2(fused_point_cloud[~target_mask]), 0.00000000001)
        dist2 = torch.cat([dist2_else, dist2], dim=0)
        
        
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Initialize opacities
        opacities = 1. * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        
        # Set h = 1 : As max_opac = sig(h) to set max opac = 1 we need h = logit(1)
        opacities[:, 0] = torch.logit(opacities[:, 0]*0.95)
        # Set w = 0.01 : As w_t = sig(w)*200, we need to set w = logit(w_t/200)
        opacities[:, 1] = (opacities[:, 1]*1.5)
        # Finally set mu to 0 as the start of the traniing
        opacities[:, 2] = torch.logit(opacities[:, 2]*0.5)
        
        
        print("Initializing Foreground/Background split")
        
        self.foreground.initialize(
            fused_point_cloud[target_mask], 
            scales[target_mask], 
            rots[target_mask], 
            opacities[target_mask], 
            features[target_mask,:,0:1].transpose(1, 2).contiguous(), 
            features[target_mask,:,1:].transpose(1, 2).contiguous()
        )
        
        self.background.initialize( 
            fused_point_cloud[~target_mask], 
            scales[~target_mask], 
            rots[~target_mask], 
            opacities[~target_mask], 
            features[~target_mask,:,0:1].transpose(1, 2).contiguous(), 
            features[~target_mask,:,1:].transpose(1, 2).contiguous()
        )
        
    def setup_optimizers(self, training_args):
        self.foreground.training_setup(training_args)
        self.background.training_setup(training_args)
        
    
    def compute_3D_filters(self, cameras, select=""):
        if select in ["foreground", ""]:
            self.foreground.compute_3D_filter(cameras)
        if select in ["background", ""]:
            self.background.compute_3D_filter(cameras)
        
    def optimizer_zero_grad(self, select=""):
        if select in ["foreground", ""]:
            self.foreground.optimizer.zero_grad(set_to_none=True)
        if select in ["background", ""]:
            self.background.optimizer.zero_grad(set_to_none=True)
            
    def oneUpShDegree(self,select=""):
        if select in ["foreground", ""]:
            self.foreground.oneupSHdegree()
        if select in ["background", ""]:
            self.background.oneupSHdegree()
    
    def update_learning_rates(self, iteration, select=""):
        if select in ["foreground", ""]:
            self.foreground.update_learning_rate(iteration)
        if select in ["background", ""]:
            self.background.update_learning_rate(iteration)

    def gaussian_constraint_loss(self):
        # Opacity Losses
        hopacloss = ((1.0 - self.foreground.get_hopac)**2).mean()
        hopacloss += ((1.0 - self.background.get_hopac)**2).mean()
        
        wopacloss = ((self.foreground.get_wopac).abs()).mean()
        wopacloss += ((self.background.get_wopac).abs()).mean()
        
        scale_exp = self.foreground.get_scaling_with_3D_filter
        # pg_loss = 0.001*(scale_exp.max(dim=1).values / scale_exp.min(dim=1).values).mean()
        max_gauss_ratio = 10
        # scale_exp = self.gaussians.get_scaling
        pg_loss = (
            torch.maximum(
                scale_exp.amax(dim=-1)  / scale_exp.amin(dim=-1),
                torch.tensor(max_gauss_ratio),
            )
            - max_gauss_ratio
        ).mean()
        
        return 0.01*hopacloss + wopacloss + pg_loss
    
    def deformation_constraint_loss(self, ts, l1, tv, mw):

        loss = self.foreground.compute_regulation(ts, l1, tv, mw)
        loss += self.background.compute_regulation(ts, l1, tv, mw)
        return loss