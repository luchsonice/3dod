from spaces import Box, Bube, Cube
import torch

def bube_to_box(bube):
    '''
    Converts a Bube to a Box.

    Args:
        bube: A Bube

    Returns:
        A Box
    '''
    if bube is None:
        raise ValueError('bube cannot be None')
    
    bube_corner = bube.get_all_corners()
    min_x = torch.min(bube_corner[:,0])
    max_x = torch.max(bube_corner[:,0])
    min_y = torch.min(bube_corner[:,1])
    max_y = torch.max(bube_corner[:,1])
    width = max_x - min_x
    height = max_y - min_y
    
    return Box(bube.center, torch.tensor([width, height]))

def box_to_bube(box, length, rotation):
    '''
    Converts a Box to a Bube.

    Args:
        bube: A Bube.
        length: The length of the box in meters?.
        rotation: The 6D rotation of the box.

    Returns:
        A Box.
    '''
    True

def cube_to_bube(cube, K):
    '''
    Converts a Cube to a Bube.

    Args:
        cube: A Cube.
        K: The 3D camera matrix of the box.

    Returns:
        A Bube.
    '''
    if cube is None:
        raise ValueError('cube cannot be None')
    
    return Bube(cube, K)
    
    