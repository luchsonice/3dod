from spaces import Box
import torch
    
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
    return [Box(detectron_boxes[i,:], format='x1, y1, x2, y2') for i in range(detectron_boxes.shape[1])]

def pixel_to_normalised_space(pixel_coord, im_shape):
    '''
    pixel_coord: Nx2
    '''
    pixel_coord = torch.tensor(pixel_coord, dtype=torch.float)
    im_shape = torch.tensor(im_shape, dtype=torch.float)

    if pixel_coord.dim() == 1:
        pixel_coord = pixel_coord.reshape(1,2)

    pixel_coord[:,0] = 2 * pixel_coord[:,0] / im_shape[0] - 1
    pixel_coord[:,1] = 2 * pixel_coord[:,1] / im_shape[1] - 1
    return pixel_coord
    