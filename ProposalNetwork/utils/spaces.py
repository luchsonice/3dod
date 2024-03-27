import torch
from cubercnn import util
from detectron2.structures import Boxes

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


class Box:
    '''
    2D box with the format [x1, y1, x2, y2]
    ```
         (x1, y1)______________________   
                |                      |   
                |                      |   
                |                      |   
                |                      |   
                |        (c1,c2)       | h
                |                      |   
                |                      |  
                |                      | 
                |______________________|(x2, y2)
                            w                      
    ```
    '''
    def __init__(self, coords: torch.Tensor) -> None:
        '''
        Args:
            format is 'x1, y1, x2, y2' '''
        self.x1 = coords[0]
        self.y1 = coords[1]
        self.x2 = coords[2]
        self.y2 = coords[3]

        # check that (x1, y1) is indeed the upper left corner
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError('(x1, y1) must be the upper left corner. Did you make sure that the input is in the correct order? (upper left, bottom right)')

        self.box = Boxes(coords.clone().detach().reshape(1,4))
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center = torch.tensor([self.x1 + self.width/2, self.y1+self.height/2])
        self.area = self.width * self.height

    def get_all_corners(self) -> torch.Tensor:
        '''
        It returns the 4 corners of the box in the format [x, y]
        '''
        ul = [self.x1, self.y1]
        ur = [self.x2, self.y1]
        br = [self.x2, self.y2]
        bl = [self.x1, self.y2]

        return torch.tensor([ul, ur, br, bl])
    
    def __repr__(self) -> str:
        return f'Box({self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f})'


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
    def __init__(self,tensor: torch.Tensor, R: torch.Tensor) -> None:
        self.tensor = tensor
        self.center = tensor[:3]
        self.dimensions = tensor[3:6]
        self.rotation = R

        if self.dimensions[0] < 0:
            raise ValueError('Width must be greater than 0.')
        if self.dimensions[1] < 0:
            raise ValueError('Height must be greater than 0.')
        if self.dimensions[2] < 0:
            raise ValueError('Length must be greater than 0.')
        
        if self.rotation.shape != (3,3):
            raise ValueError('Rotation must be a 3x3 matrix.')

        color = [c/255.0 for c in util.get_color()]
        self.cube = util.mesh_cuboid(torch.cat((self.center,self.dimensions)), self.rotation, color=color)

    def get_cube(self):
        return self.cube

    def get_all_corners(self) -> torch.Tensor:
        return self.cube.verts_list()[0]
    
    def get_bube_corners(self,K) -> torch.Tensor:
        cube_corners = self.get_all_corners()
        cube_corners = torch.mm(K, cube_corners.t()).t()
        return cube_corners[:,:2]/cube_corners[:,2].unsqueeze(1)
    
    def get_cuboid_verts_faces(self):
        '''wrap ``util.get_cuboid_verts_faces``'''
        return util.get_cuboid_verts_faces(torch.cat((self.center,self.dimensions)), self.rotation)
    
    def __repr__(self) -> str:
        return f'Cube({self.center}, {self.dimensions}, {self.rotation})'