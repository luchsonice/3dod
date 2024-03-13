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
    
    bube_corners = bube.get_all_corners()
    min_x = torch.min(bube_corners[:,0])
    max_x = torch.max(bube_corners[:,0])
    min_y = torch.min(bube_corners[:,1])
    max_y = torch.max(bube_corners[:,1])
    width = max_x - min_x
    height = max_y - min_y
    center = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2])
    
    return Box(torch.tensor([center[0], center[1], width, height]))

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
    
def cube_to_box(cube,K):
    '''
    Converts a Cube to a Box.

    Args:
        cube: A Cube.
        K: The 3D camera matrix of the box.

    Returns:
        A Box.
    '''
    return bube_to_box(cube_to_bube(cube,K))
    
def Boxes_to_Box(boxes: detectron2.structures.boxes.Boxes) -> Box:
    '''
    Converts a detectron2 Boxes object to a spaces.Box.
    if the input is many boxes, it will return a list of spaces.Boxes

    Args:
        boxes: A Boxes.

    Returns:
        A Box. or list of Boxes
    '''
    if boxes is None:
        raise ValueError('boxes cannot be None')
    if boxes.tensor.shape[0] == 1:
        return Box(boxes.tensor.squeeze(),format='x1, y1, x2, y2')
    else:
        return [Box(i, format='x1, y1, x2, y2') for i in boxes.tensor]
    