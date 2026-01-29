import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network

from gaussian_renderer import render_motion_point_mask

class GPoint:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self,):
        self.setup_functions()

        self._xyz = torch.ones((1,3), dtype=torch.float, device='cuda')
        self._opacity = torch.ones((1,1), dtype=torch.float, device='cuda')
    
        
        self._features_dc = torch.zeros((1,1,3), dtype=torch.float, device='cuda')
        self._features_rest = torch.zeros((1,15,3), dtype=torch.float, device='cuda')
        
        self._scaling = self.scaling_inverse_activation(torch.ones((1,3), dtype=torch.float, device='cuda')*0.1)
        self._rotation = torch.ones((1,4), dtype=torch.float, device='cuda')
        

        # World-Up is X for Condense
        self._xyz[:, 0] *= 2.
        
        # Make Point Red
        self._features_dc[..., 0] += 1.
        self._features_rest[..., 0] += 0.1
        
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def full_scene_construction(self, means, scales, rots, feats, opacs):
        
        means = torch.cat([self.get_xyz, means], dim=0)
        scales = torch.cat([self.get_scaling, scales], dim=0)
        rots = torch.cat([self.get_rotation, rots], dim=0)
        feats = torch.cat([self.get_features, feats], dim=0)
        opacs = torch.cat([self.get_opacity, opacs], dim=0)
        
        return means, scales, rots, feats, opacs