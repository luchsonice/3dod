import torch
import numpy as np
from detectron2.structures import Boxes
    
def cube_to_box(cube,K):
    '''
    Converts a Cube to a Box.

    Args:
        cube: A Cube.
        K: The 3D camera matrix of the box.

    Returns:
        A Box.
    '''
    bube_corners = cube.get_bube_corners(K)
    
    min_x = torch.min(bube_corners[:,0])
    max_x = torch.max(bube_corners[:,0])
    min_y = torch.min(bube_corners[:,1])
    max_y = torch.max(bube_corners[:,1])
    
    return Boxes(torch.tensor([[min_x, min_y, max_x, max_y]], device=cube.tensor.device))

def cubes_to_box(cubes, K, im_shape):
    '''
    Converts a Cubes to a Boxes.

    Args:
        cubes: A Cubes.
        K: The 3D camera matrix of the box.
        im_shape: The shape of the image (width, height).

    Returns:
        A Box.
    '''
    bube_corners = cubes.get_bube_corners(K, im_shape)
    min_x, _ = torch.min(bube_corners[:, :, :, 0], 2)
    max_x, _ = torch.max(bube_corners[:, :, :, 0], 2)
    min_y, _ = torch.min(bube_corners[:, :, :, 1], 2)
    max_y, _ = torch.max(bube_corners[:, :, :, 1], 2)

    values = torch.stack((min_x, min_y, max_x, max_y),dim=2)
    box_list = []
    for i in range(cubes.num_instances):
        box_list.append(Boxes(values[i]))

    return box_list

def pixel_to_normalised_space(pixel_coord, im_shape, norm_shape):
    '''
    pixel_coord: List of length N
    im_shape: List of length N
    norm_shape: List of length N
    '''
    pixel_coord = torch.stack(pixel_coord,dim=1)

    new_coords = pixel_coord.to(torch.float32)

    for i in range(pixel_coord.size(1)):
        old_dim = im_shape[i]
        new_dim = norm_shape[i]

        new_coords[:,i] -= 0.5 * old_dim
        new_coords[:,i] *= new_dim / old_dim
    
    return new_coords # TODO feel like its missing a line, something if normshape is not 2. Where did we take inspiration from? A library?

def normalised_space_to_pixel(coords, im_shape, norm_shape):
    new_coords = np.array(coords).astype(np.float32)

    for i in range(len(new_coords)):
        new_dim = im_shape[i]
        old_dim = norm_shape[i]
        new_coords[i] *= new_dim / old_dim
        new_coords[i] += 0.5 * new_dim

    return new_coords
    