import torch
from cubercnn import util
import numpy as np

class Box:
    '''
    2D box with the format [c1, c2, w, h]

                 ______________________   
                |                      |   
                |                      |   
                |                      |   
                |                      |   
                |        (c1,c2)       | h
                |                      |   
                |                      |  
                |                      | 
                |______________________|
                            w             

    '''
    def __init__(self,tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.c1 = tensor[0]
        self.c2 = tensor[1]
        self.width = tensor[2]
        self.height = tensor[3]

        if self.width < 0:
            raise ValueError('Width must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, w, h)')
        if self.height < 0:
            raise ValueError('Height must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, w, h)')

    def get_all_corners(self) -> torch.Tensor:
        '''
        It returns the 4 corners of the box in the format [x, y]
        '''
        ul = [self.c1-self.width/2, self.c2+self.height/2]
        ur = [self.c1+self.width/2, self.c2+self.height/2]
        bl = [self.c1-self.width/2, self.c2-self.height/2]
        br = [self.c1+self.width/2, self.c2-self.height/2]

        return torch.tensor([ul, ur, bl, br])
    





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
    '''
    def __init__(self,tensor: torch.Tensor, R: torch.Tensor) -> None:
        self.tensor = tensor
        self.center = tensor[:3]
        self.dimensions = tensor[3:6]
        self.rotation = R

        if self.dimensions[0] < 0:
            raise ValueError('Width must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        if self.dimensions[1] < 0:
            raise ValueError('Height must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        if self.dimensions[2] < 0:
            raise ValueError('Length must be greater than 0. Did you make sure that the input is in the correct order? (c1, c2, c3, w, h, l, p)')
        
        if self.rotation.shape != (3,3):
            raise ValueError('Rotation must be a 3x3 matrix.')

        color = [c/255.0 for c in util.get_color()]
        self.cube = util.mesh_cuboid(torch.cat((self.center,self.dimensions)), self.rotation, color=color)

    def get_cube(self):
        return self.cube

    def get_all_corners(self) -> torch.Tensor:
        return self.cube.verts_list()

        


class Bube:
    '''
    3D box on the 2D image plane in the format [cube, K]

    Args:
        cube: A cube.
        K: The 3D camera matrix of the box.


                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2)       |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w             
    '''
    def __init__(self,cube: Cube, K: torch.Tensor) -> None:
        self.cube = cube
        self.K = K
        self.center = cube.center[:2] # This is not correct, is it?
        self.dimensions = cube.dimensions

        if K.shape != (3,3):
            raise ValueError('K must be a 3x3 matrix.')

    def get_all_corners(self) -> torch.Tensor:
        '''
        It returns the 8 corners of the bube in the format [x, y]
        '''
        corners = self.cube.get_all_corners()
        corners = torch.cat((corners, torch.ones(8,1)), dim=1)
        corners = torch.mm(self.K, corners.t()).t()
        corners = corners[:,:2]/corners[:,2].unsqueeze(1)

        return corners