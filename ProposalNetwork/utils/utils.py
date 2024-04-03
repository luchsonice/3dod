import torch
from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space, normalised_space_to_pixel
import numpy as np
from cubercnn import util

from detectron2.structures import pairwise_iou
from pytorch3d.ops import box3d_overlap

##### Proposal
def normalize_vector(v):
    v_mag = torch.sqrt(v.pow(2).sum())
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(1,1).expand(1,v.shape[0])
    v = v/v_mag

    return v[0]
    
def cross_product(u, v):
    i = u[1]*v[2] - u[2]*v[1]
    j = u[2]*v[0] - u[0]*v[2]
    k = u[0]*v[1] - u[1]*v[0]
    out = torch.cat((i.view(1,1), j.view(1,1), k.view(1,1)),1)
        
    return out[0]

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[0:3]
    y_raw = poses[3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x,y_raw)
    z = normalize_vector(z)
    y = cross_product(z,x)
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2)[0]

    return matrix

def sample_normal_greater_than(mean, std, threshold):
    sample = np.random.normal(mean, std)
    while sample < threshold:
        sample = np.random.normal(mean, std)
    return sample

def make_cube(x_range, y_range, z, w_prior, h_prior, l_prior):
    '''
    need xyz, whl, and pose (R)
    '''
    # xyz
    x = (x_range[0]-x_range[1]) * torch.rand(1) + x_range[1]
    y = (y_range[0]-y_range[1]) * torch.rand(1) + y_range[1]
    xyz = torch.tensor([x, y, z])

    # whl
    w = sample_normal_greater_than(w_prior[0],w_prior[1],0.1)
    h = sample_normal_greater_than(h_prior[0],h_prior[1],0.1)
    l = sample_normal_greater_than(l_prior[0],l_prior[1],0.05)
    whl = torch.tensor([w, h, l])

    # R
    #rotation_matrix = compute_rotation_matrix_from_ortho6d(torch.rand(6)) # Use this when learnable
    rx = np.random.rand(1) * np.pi - np.pi/2
    ry = np.random.rand(1) * np.pi - np.pi/2
    rz = np.random.rand(1) * np.pi - np.pi/2
    rotation_matrix = torch.from_numpy(util.euler2mat([rx,ry,rz]))
    
    return xyz, whl, rotation_matrix

def is_box_included_in_other_box(reference_box, proposed_box):
    reference_corners = reference_box.get_all_corners()
    proposed_corners = proposed_box.get_all_corners()

    reference_min_x = torch.min(reference_corners[:,0])
    reference_max_x = torch.max(reference_corners[:,0])
    reference_min_y = torch.min(reference_corners[:,1])
    reference_max_y = torch.max(reference_corners[:,1])

    proposed_min_x = torch.min(proposed_corners[:,0])
    proposed_max_x = torch.max(proposed_corners[:,0])
    proposed_min_y = torch.min(proposed_corners[:,1])
    proposed_max_y = torch.max(proposed_corners[:,1])

    return (reference_min_x <= proposed_min_x <= proposed_max_x <= reference_max_x and reference_min_y <= proposed_min_y <= proposed_max_y <= reference_max_y)







##### Scoring
def iou_2d(gt_box, proposal_boxes):
    '''
    gt_box: Box
    proposal_box: list of Box
    '''
    IoU = []
    for i in range(len(proposal_boxes)):
        proposal_box = proposal_boxes[i]
        IoU.append(pairwise_iou(gt_box.box,proposal_box.box)[0][0].item())
    return IoU

def iou_3d(gt_cube, proposal_cubes):
    """
    Compute the Intersection over Union (IoU) of two 3D cubes.

    Parameters:
    - gt_cube: GT Cube.
    - proposal_cube: List of Proposal Cubes.

    Returns:
    - iou: Intersection over Union (IoU) value.
    """
    gt_corners = torch.stack([gt_cube.get_all_corners()])
    proposal_corners = torch.stack([cube.get_all_corners() for cube in proposal_cubes]).to(gt_corners.device)

    # TODO check if corners in correct order; Should be
    vol, iou = box3d_overlap(gt_corners,proposal_corners)
    iou = iou[0]

    return iou

def custom_mapping(x,beta=1.7):
    '''
    maps the input curve to be S shaped instead of linear
    
    Args:
    beta: number > 1, higher beta is more aggressive
    x: list of floats betweeen and including 0 and 1
    beta: number > 1 higher beta is more aggressive
    '''
    mapped_list = []
    for i in range(len(x)):
        if x[i] <= 0:
            mapped_list.append(0.0)
        else:
            mapped_list.append((1 / (1 + (x[i] / (1 - x[i])) ** (-beta))))
    
    return mapped_list

def Boxes_to_list_of_Box(Boxes):
    '''
    Boxes: detectron2 Boxes
    '''
    detectron_boxes = Boxes.tensor
    return [Box(detectron_boxes[i,:]) for i in range(detectron_boxes.shape[1])]

def mask_iou(segmentation_mask, bube_mask):
    '''
    Area is of segmentation_mask
    '''
    # Compute intersection mask
    intersection_mask = np.logical_and(segmentation_mask, bube_mask).astype(np.uint8)
    # Count pixels in intersection
    intersection_area = np.sum(intersection_mask)

    # Compute union mask
    union_mask = np.logical_or(segmentation_mask, bube_mask).astype(np.uint8)
    # Count pixels in union mask
    union_area = np.sum(union_mask)

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def is_gt_included(gt_cube,x_range,y_range,z_range, w_prior, h_prior, l_prior):
    # Define how far away dimensions need to be to be counted as unachievable
    stds_away = 3
    # Center
    if not (x_range[0] < gt_cube.center[0] < x_range[1]):
        print('Because of x')
        return False
    elif not (y_range[0] < gt_cube.center[1] < y_range[1]):
        print('Because of y')
        return False
    # Depth
    elif not (z_range[0] < gt_cube.center[2] < z_range[1]):
        print('Because of z')
        return False
    # Dimensions
    elif (gt_cube.dimensions[0] < w_prior[0]-stds_away*w_prior[1]):
        print('Because of w-')
        return False
    elif (gt_cube.dimensions[0] > w_prior[0]+stds_away*w_prior[1]):
        print('Because of w+')
        return False
    elif (gt_cube.dimensions[1] < h_prior[0]-stds_away*h_prior[1]):
        print('Because of h-')
        return False
    elif (gt_cube.dimensions[1] > h_prior[0]+stds_away*h_prior[1]):
        print('Because of h+')
        return False
    elif (gt_cube.dimensions[2] < l_prior[0]-stds_away*l_prior[1]):
        print('Because of l-')
        return False
    elif (gt_cube.dimensions[2] > l_prior[0]+stds_away*l_prior[1]):
        print('Because of l+')
        return False
    else:
        return True

    # rotation nothing yet

def euler_to_unit_vector(eulers):
    """
    Convert Euler angles to a unit vector.
    """
    yaw, pitch, roll = eulers
    
    # Calculate the components of the unit vector
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    
    # Normalize the vector
    length = np.sqrt(x**2 + y**2 + z**2)
    unit_vector = np.array([x, y, z]) / length
    
    return unit_vector