from spaces import Box
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
    
    return Box(torch.tensor([bube.x1, bube.y3, bube.x3, bube.y2]))

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

    