import torch
from torch import nn
import numpy as np
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth:bool=False,
                 cxfx=None,
                 width=None, height=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.gt_alpha_mask = None
        if image is not None:
            if image.shape[0] == 4:
                self.gt_alpha_mask = image[-1,:]
                image = image[:-1, :]

        self.original_image = image
        if self.original_image is not None:
            try:
                self.image_width = self.original_image.shape[2]
                self.image_height = self.original_image.shape[1]
            except:
                self.image_width = self.original_image.shape[1]
                self.image_height = self.original_image.shape[0]
        else:
            self.image_height = height
            self.image_width = width

        self.depth = depth
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        if cxfx is None:
            self.cx =  self.image_width / 2.0
            self.cy = self.image_height / 2.0
            fx = 0.5 * self.image_width / np.tan(self.FoVx * 0.5)
            fy = 0.5 * self.image_height / np.tan(self.FoVy * 0.5)
        else:
            self.cx = cxfx[0]
            self.cy = cxfx[1]
            fx = cxfx[2]
            fy = cxfx[3]
        self.fx = fx
        self.fy = fy

    @property
    def intrinsics(self):
        return torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0,  0,      1]
        ], dtype=torch.float32)

    @property
    def camera_center(self):
        return torch.from_numpy(self.T).float()
        
    @property
    def c2w(self):
        # Assemble directly (R,t are world->cam)
        Rt = torch.eye(4, device=self.data_device, dtype=torch.float32)
        Rt[:3, :3] = torch.from_numpy(self.R).to(self.data_device).float()
        Rt[:3, 3]  = torch.from_numpy(self.T).to(self.data_device).float()
        return Rt

    @property
    def w2c(self):
        # Invert once, get cam->world
        w2c = self.c2w
        R_wc = w2c[:3, :3]
        t_wc = w2c[:3, 3]
        Rt = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
        Rt[:3, :3] = R_wc.T
        Rt[:3, 3]  = -(R_wc.T @ t_wc)
        return Rt
    
    def generate_rays(self,device="cuda"):
        H, W = self.image_height, self.image_width
        w2c = self.w2c.to(device)             # world -> cam
        R_wc = w2c[:3, :3]                    # world->cam rotation
        t_wc = w2c[:3, 3]                     # world->cam translation (shape [3])
        
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        
        # Pixel grid 
        i = torch.arange(W, device=device, dtype=torch.float32)
        j = torch.arange(H, device=device, dtype=torch.float32)
        jj, ii = torch.meshgrid(j, i, indexing="ij")

        x = (ii - cx) / fx
        y = (jj - cy) / fy
        z = torch.ones_like(x)

        dirs_cam = torch.stack([x, y, z], dim=-1)
        dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True)

        # directions to world (row-vectors)
        dirs_world = dirs_cam @ R_wc
        dirs_world = (R_wc.T @ dirs_cam[..., None]).squeeze(-1)
        # origin in world
        cam_center_world = -(R_wc.T @ t_wc)
        origins_world = cam_center_world.view(1, 1, 3).expand_as(dirs_world)

        return origins_world, dirs_world
