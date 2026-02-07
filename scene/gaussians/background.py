import torch
from utils.general_utils import get_expon_lr_func
import os
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness


from scene.gaussians import GaussianModel

class BackgroundGaussians(GaussianModel):

    def __init__(self, args):
        super().__init__(args)
        
        self._deformation = deform_network(args, "background")

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
        print("Loading foreground from {}".format(path))
        weight_dict = torch.load(os.path.join(path,"background_deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
    
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "background_deformation.pth"))
    
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
