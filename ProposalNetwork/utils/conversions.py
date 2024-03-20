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
    
    return Box(torch.tensor([min_x, min_y, max_x, max_y]))

def Boxes_to_list_of_Box(Boxes):
    '''
    Boxes: detectron2 Boxes
    '''
    detectron_boxes = Boxes.tensor
    return [Box(detectron_boxes[i,:]) for i in range(detectron_boxes.shape[1])]

def pixel_to_normalised_space(pixel_coord, im_shape):
    '''
    pixel_coord: Nx2
    '''
    pixel_coord = np.array(pixel_coord)
    if np.shape(pixel_coord) == (2,):
        pixel_coord = pixel_coord.reshape(1,2) # TODO should be for general N

    new_height, new_width = 2,2
    old_width = im_shape[0]
    old_height = im_shape[1]
    new_coords = pixel_coord.astype(np.float32)
    new_coords[:, 0] -= 0.5 * old_width
    new_coords[:, 1] -= 0.5 * old_height
    new_coords[:, 0] *= new_width / old_width
    new_coords[:, 1] *= new_height / old_height
    
    return new_coords

def normalised_space_to_pixel(coords,im_shape):
    coords = np.array(coords)
    if np.shape(coords) == (2,):
        coords = coords.reshape(1,2) # TODO should be for general N

    new_height, new_width = im_shape[1],im_shape[0]
    old_width = 2
    old_height = 2
    new_coords = coords.astype(np.float32)
    new_coords[:, 0] *= new_width / old_width
    new_coords[:, 1] *= new_height / old_height
    new_coords[:, 0] += 0.5 * new_width
    new_coords[:, 1] += 0.5 * new_height

    return [[int(entry) for entry in sublist] for sublist in new_coords][0]
    