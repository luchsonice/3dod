from ProposalNetwork.utils.spaces import Box
import torch
import numpy as np
    
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
    
    return Box(torch.tensor([min_x, min_y, max_x, max_y], device=cube.tensor.device))

def Boxes_to_list_of_Box(Boxes):
    '''
    Boxes: detectron2 Boxes
    '''
    detectron_boxes = Boxes.tensor
    return [Box(detectron_boxes[i,:]) for i in range(detectron_boxes.shape[1])]

def pixel_to_normalised_space(pixel_coord, im_shape, norm_shape):
    '''
    pixel_coord: List of length N
    im_shape: List of length N
    norm_shape: List of length N
    '''
    if not torch.is_tensor(pixel_coord):
        pixel_coord = torch.tensor(pixel_coord)

    new_coords = pixel_coord.to(torch.float32)

    for i in range(len(pixel_coord)):
        old_dim = im_shape[i]
        new_dim = norm_shape[i]

        new_coords[i] -= 0.5 * old_dim
        new_coords[i] *= new_dim / old_dim
    
    return new_coords # TODO feel like its missing a line, something if normshape is not 2. Where did we take inspiration from? A library?

def normalised_space_to_pixel(coords, im_shape, norm_shape):
    new_coords = np.array(coords).astype(np.float32)

    for i in range(len(new_coords)):
        new_dim = im_shape[i]
        old_dim = norm_shape[i]
        new_coords[i] *= new_dim / old_dim
        new_coords[i] += 0.5 * new_dim

    return new_coords
    