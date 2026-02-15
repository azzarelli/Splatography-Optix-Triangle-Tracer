import torch
from utils.general_utils import get_expon_lr_func
from torch import nn
import os
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness, compute_plane_tv

from gaussian_renderer import render_motion_point_mask


from scene.gaussians import GaussianModel

class ForegroundGaussians(GaussianModel):

    def __init__(self, args):
        super().__init__(args)
        self._deformation = deform_network(args)

    def set_aabb(self, xyz_max, xyz_min):
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)

    def training_setup(self, training_args):
        l = [
            # Gaussians
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},            
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    
            
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr

    def load_model(self, path):
        print("Loading foreground deformation from {}".format(path))
        weight_dict = torch.load(os.path.join(path,"foreground_deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
    
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "foreground_deformation.pth"))

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        # self._colors = optimizable_tensors["color"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.target_mask = self.target_mask[valid_points_mask]
        
    def densification_postfix(self, new_xyz,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "scaling" : new_scaling,
            "opacity": new_opacities,
            "rotation" : new_rotation,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
    
    def duplicate(self):
        new_xyz = self._xyz + torch.rand_like(self._xyz)*0.005
        new_features_dc = self._features_dc
        new_features_rest = self._features_rest
        new_scaling = self._scaling
        new_rotation = self._rotation
        new_opacities = self._opacity
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def dynamic_dupelication(self):
        """Duplicate points with highly dynamic motion - maybe the top 10% of points with the largest motions?
        """
        
        selected_pts_mask = render_motion_point_mask(self)
        
        new_xyz = self._xyz[selected_pts_mask] + torch.rand_like(self._xyz[selected_pts_mask])*0.005
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def prune(self,cam_list): # h_thresold, extent, max_screen_size,):
        """Very Simple Pruning for now
        """
        h = self.get_coarse_opacity_with_3D_filter
        prune_mask = (h < 0.05).squeeze(-1)
                
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
          
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight,
                           minview_weight):
        tvtotal = 0
        l1total = 0
        tstotal = 0
        
        wavelets = self._deformation.deformation_net.grid.waveplanes_list()
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self._deformation.deformation_net.grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvtotal += compute_plane_smoothness(grid)
            elif index in [2, 4, 5]:
                for grid in grids: # space time
                    tstotal += compute_plane_smoothness(grid)
                
                for grid in wavelets[index]:
                    l1total += torch.abs(grid).mean()
                    
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total # + minview_weight*col
