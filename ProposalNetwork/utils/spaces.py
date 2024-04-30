import torch
from cubercnn import util

'''
coordinate system is assumed to have origin in the upper left
(0,0) _________________(N,0)
|  
|    
| 
|
|
(0,M)
'''

class Cube:
    '''
    3D box in the format [c1, c2, c3, w, h, l, R]

    Args:
        c1: The x coordinate of the center of the box.
        c2: The y coordinate of the center of the box.
        c3: The z coordinate of the center of the box.
        w: The width of the box in meters.
        h: The height of the box in meters.
        l: The length of the box in meters.
        R: The 3D rotation matrix of the box.
    ```

                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2,c3)    |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w             
    ```
    '''
    def __init__(self,tensor: torch.Tensor, R: torch.Tensor, score=None, label=None) -> None:
        self.tensor = tensor
        self.center = tensor[:3]
        self.dimensions = tensor[3:6]
        self.rotation = R

        # score and label are meant as auxiliary information
        self.score = score
        self.label = label

    def get_cube(self):
        color = [c/255.0 for c in util.get_color()]
        return util.mesh_cuboid(torch.cat((self.center,self.dimensions)), self.rotation, color=color)
    
    def get_all_corners(self):
        '''wrap ``util.get_cuboid_verts_faces``
        
        Returns:
            verts: the 3D vertices of the cuboid in camera space'''
        verts, _ = util.get_cuboid_verts_faces(torch.cat((self.center,self.dimensions)), self.rotation)
        return verts
    
    def get_bube_corners(self,K) -> torch.Tensor:
        cube_corners = self.get_all_corners()
        cube_corners = torch.mm(K, cube_corners.t()).t()
        return cube_corners[:,:2]/cube_corners[:,2].unsqueeze(1)
    
    def get_volume(self) -> float:
        return self.dimensions.prod().item()
    
    def get_projected_2d_area(self, K) -> float:
        def cube_to_box(cube, K):
            bube_corners = cube.get_bube_corners(K)
    
            min_x = torch.min(bube_corners[:,0])
            max_x = torch.max(bube_corners[:,0])
            min_y = torch.min(bube_corners[:,1])
            max_y = torch.max(bube_corners[:,1])
            
            return Box(torch.tensor([min_x, min_y, max_x, max_y], device=cube.tensor.device))
        area = cube_to_box(self, K).box.area()
        return area

    
    def __repr__(self) -> str:
        return f'Cube({self.center}, {self.dimensions}, {self.rotation})'
    
    def to_device(self, device):
        '''
        Move all tensors of the instantiated class to the specified device.

        Args:
            device: The device to move the tensors to (e.g., 'cuda', 'cpu').
        '''
        self.tensor = self.tensor.to(device)
        self.center = self.center.to(device)
        self.dimensions = self.dimensions.to(device)
        self.rotation = self.rotation.to(device)
        return self