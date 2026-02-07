import torch
import numpy as np
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

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
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        self.optimizer = None
        self.spatial_scale = 0        
        
        self.setup_functions()
        
        self._deformation = None


    
    """ Parametric Properties
    """
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    def get_fine_opacity_with_3D_filter(self, opacity):
        scales = self.get_scaling
        filter3D = self.filter_3D
            
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(filter3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_coarse_opacity_with_3D_filter(self):
        opacity = torch.sigmoid(self.get_opacity[:, 0]).unsqueeze(-1)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_hopac(self):
        return torch.sigmoid(self.get_opacity[:, 0]).unsqueeze(-1)
    @property
    def get_wopac(self):
        return self.get_opacity[:, 1]
    @property
    def get_muopac(self):
        return torch.sigmoid(self.get_opacity[:, 2]).unsqueeze(-1)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def get_covmat(self):
        w, x, y, z = self.get_rotation.unbind(-1)
        scale = self.get_scaling
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = torch.stack([
            torch.stack([1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)], dim=-1),
        ], dim=-2)
        
        e1 = torch.tensor([1,0,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)
        e2 = torch.tensor([0,1,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)

        # Scale local basis
        v1 = e1 * scale[:, [0]]  # (N,3)
        v2 = e2 * scale[:, [1]]  # (N,3)

        # Apply rotation: batch matmul (N,3,3) @ (N,3,1) -> (N,3,1)
        t_u = torch.bmm(R, v1.unsqueeze(-1)).squeeze(-1)  # (N,3)
        t_v = torch.bmm(R, v2.unsqueeze(-1)).squeeze(-1)  # (N,3)

        # Magnitudes
        m_u = torch.linalg.norm(t_u, dim=-1)
        m_v = torch.linalg.norm(t_v, dim=-1)

        # Directions (normalized)
        d_u = t_u / m_u.unsqueeze(-1)
        d_v = t_v / m_v.unsqueeze(-1)

        magnitudes = torch.stack([m_u, m_v], dim=-1)     # (N,2)
        directions = torch.stack([d_u, d_v], dim=1)      # (N,2,3)

        return magnitudes, directions
    
    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
                        
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.1
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.fx + camera.image_width / 2.0
            y = y / z * camera.fy + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.fx:
                focal_length = camera.fx
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    """ Functions
    """
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_dc,
            self._features_rest,
            self.filter_3D,
            self.spatial_lr_scale
        )

    def restore(self, model_args):
        (self.active_sh_degree,
        self._xyz,
        self._scaling,
        self._rotation,
        self._opacity,
        self._features_dc,
        self._features_rest,
        self.filter_3D,
        self.spatial_lr_scale) = model_args

    def initialize(self, xyz, scales, roations, opacities, f_dc, f_rest):
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(roations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self._features_dc = nn.Parameter(f_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(f_rest.requires_grad_(True))
        
        mean_foreground = xyz.mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(xyz - mean_foreground, dim=1)
        self.spatial_lr_scale = torch.max(dist_foreground).detach().cpu().numpy()

        self.active_sh_degree = 0
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._opacity.shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def load_ply(self, path):
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

        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree


    def deform(self, time):
        means3D_final, rotations_final, opacity_final, colors_final = self._deformation(
            times_sel=time,
            point=self.get_xyz,
            rotations=self._rotation,
            scales=self.get_scaling_with_3D_filter,
            h_emb=self.get_opacity,
            shs=self.get_features
        )

        opacity_final = self.get_fine_opacity_with_3D_filter(opacity_final)
        rotations_final = self.rotation_activation(rotations_final)
        
        return means3D_final, self.get_scaling_with_3D_filter, rotations_final, opacity_final, colors_final
    
    
    def nondeform(self):
        opacity_final = self.get_fine_opacity_with_3D_filter(torch.sigmoid(self._opacity[:, 0]).unsqueeze(-1))
        rotations_final = self.rotation_activation(self._rotation)
        
        return self.get_xyz, self.get_scaling_with_3D_filter, rotations_final, opacity_final, self.get_features
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                param = group["params"][0]
                
                # Ensure the optimizer state is initialized
                if param not in self.optimizer.state:
                    self.optimizer.state[param] = {}
                stored_state = self.optimizer.state[param]

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[param]

                new_param = nn.Parameter(tensor.requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state

                optimizable_tensors[group["name"]] = new_param
        return optimizable_tensors
    
    def reset_opacity(self):
        print('Resetting foreground opacity')
        opacities_new = self.get_opacity
        opacities_new[:,0] =  torch.tensor(0.05).item()
        opacities_new[:,1] = 1.5

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]