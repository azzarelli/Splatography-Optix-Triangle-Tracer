import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from utils.general_utils import strip_symmetric, build_scaling_rotation

class Deformation(nn.Module):
    def __init__(self, W=256, args=None, def_type=""):
        super(Deformation, self).__init__()
        self.W = W
        bound = args.bounds
        self.is_background = True if def_type == "background" else False
        args = args.scene_config if self.is_background else args.target_config
        
        self.grid = WavePlaneField(bound, args)

        self.args = args
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # inputs scaling, scalingmod=1.0, rotation
        self.covariance_activation = build_covariance_from_scaling_rotation


        self.create_net()

        
    def set_aabb(self, xyz_max, xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)

    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        self.spacetime_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))
        
        self.pos_coeffs = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        
        if self.is_background == False:
            self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
            self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 15*3))
    
    def query_spacetime(self, xyzs, t, covariances):
        time = torch.full_like(xyzs[:,0], t, device=xyzs.device).unsqueeze(-1)
        
        space, spacetime = self.grid(xyzs, time, covariances)

        st = self.spacetime_enc(space * spacetime)
        return st
    
    def forward(self,rays_pts_emb, rotations_emb, scale_emb, shs_emb, time_emb, h_emb):

        covariances = self.covariance_activation(scale_emb, 1., rotations_emb)
        dyn_feature = self.query_spacetime(rays_pts_emb,time_emb, covariances)
        
        
        # Change in position & opacity for both
        pts = rays_pts_emb + self.pos_coeffs(dyn_feature)      
        
        # opacity = h_emb[:,0].unsqueeze(-1)
        opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
        # w = (h_emb[:,1]**2).unsqueeze(-1)
        # mu = torch.sigmoid(h_emb[:,2]).unsqueeze(-1)
        
        # t = time_emb
        # opacity = torch.exp(-w * (t-mu)**2)
        
        # Background only condition - early exit
        if self.is_background:
            return pts, rotations_emb, opacity, shs_emb
 
        # Rotation
        rotations = rotations_emb + self.rotations_deform(dyn_feature)
        
        shs_emb[:, 1:] = shs_emb[:, 1:] + self.shs_deform(dyn_feature).view(-1, 15, 3)
        
        return pts, rotations, opacity, shs_emb
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    

class deform_network(nn.Module):
    def __init__(self, args, def_type="foreground") :
        super(deform_network, self).__init__()
        net_width = args.net_width
        self.deformation_net = Deformation(W=net_width,  args=args, def_type=def_type)
        

        self.apply(initialize_weights)
        
        self.to("cuda")

    def forward(self, point, rotations=None, scales=None, shs=None, times_sel=None, h_emb=None):

        return  self.deformation_net(
            point,
            rotations,
            scales,
            shs,
            times_sel, 
            h_emb=h_emb, 
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            