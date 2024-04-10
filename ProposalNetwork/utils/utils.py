import torch
from ProposalNetwork.utils.spaces import Box
import numpy as np
from cubercnn import util
import matplotlib.pyplot as plt

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

def sample_normal_greater_than_para(mean, std, threshold, count):
    device = mean.device
    mean = mean.item()
    std = std.item()
    samples = torch.normal(mean, std, size=(count,))
    while torch.any(samples < threshold):
        samples[samples < threshold] = torch.normal(mean, std, size=((samples < threshold).sum(),))
    return samples.to(device)

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
    rx = np.random.rand(1) * 2 * np.pi - np.pi
    ry = np.random.rand(1) * 2 * np.pi - np.pi
    rz = np.random.rand(1) * 2 * np.pi - np.pi
    rotation_matrix = torch.from_numpy(util.euler2mat([rx,ry,rz]))
    
    return xyz, whl, rotation_matrix

def make_cubes_parallel(x_range, y_range, z, w_prior, h_prior, l_prior, number_of_proposals=1):
    '''
    need xyz, whl, and pose (R)
    it does not run faster on cuda.
    '''
    # xyz
    x_range = x_range.repeat(number_of_proposals,1)
    y_range = y_range.repeat(number_of_proposals,1)
    x = (x_range[:, 0]-x_range[:, 1]).t() * torch.rand(number_of_proposals) + x_range[:, 1]
    y = (y_range[:, 0]-y_range[:, 1]).t() * torch.rand(number_of_proposals) + y_range[:, 1]
    xyz = torch.stack([x, y, z], 1)

    # whl
    w = sample_normal_greater_than_para(w_prior[0], w_prior[1], 0.1, number_of_proposals)
    h = sample_normal_greater_than_para(h_prior[0], h_prior[1], 0.1, number_of_proposals)
    l = sample_normal_greater_than_para(l_prior[0], l_prior[1], 0.05, number_of_proposals)
    whl = torch.stack([w, h, l], 1)

    # R
    rotation_matrix = randn_orthobasis_torch(number_of_proposals) 
    
    return xyz, whl, rotation_matrix

def randn_orthobasis_torch(num_samples=1):
    z = torch.randn(num_samples, 3, 3)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    z[:, 0] = torch.cross(z[:, 1], z[:, 2], dim=-1)
    z[:, 0] = z[:, 0] / torch.norm(z[:, 0], dim=-1, keepdim=True)
    z[:, 1] = torch.cross(z[:, 2], z[:, 0], dim=-1)
    z[:, 1] = z[:, 1] / torch.norm(z[:, 1], dim=-1, keepdim=True)
    return z

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
    stds_away = 1.5
    # Center
    because_of = []
    if not (x_range[0] < gt_cube.center[0] < x_range[1]):
        if (gt_cube.center[0] < x_range[0]):
            val = abs(x_range[0] - gt_cube.center[0])
        else:
            val = abs(gt_cube.center[0] - x_range[1])
        because_of.append(f'x by {val:.1f}')
    if not (y_range[0] < gt_cube.center[1] < y_range[1]):
        if (gt_cube.center[1] < y_range[0]):
            val = abs(y_range[0] - gt_cube.center[1])
        else:
            val = abs(gt_cube.center[1] - y_range[1])
        because_of.append(f'y by {val:.1f}')
    # Depth
    if not (z_range[0] < gt_cube.center[2] < z_range[1]):
        if (gt_cube.center[2] < z_range[0]):
            val = abs(z_range[0] - gt_cube.center[2])
        else:
            val = abs(gt_cube.center[2] - z_range[1])
        because_of.append(f'z by {val:.1f}')
    # Dimensions
    if (gt_cube.dimensions[0] < w_prior[0]-stds_away*w_prior[1]):
        because_of.append('w-')
    if (gt_cube.dimensions[0] > w_prior[0]+stds_away*w_prior[1]):
        because_of.append('w+')
    if (gt_cube.dimensions[1] < h_prior[0]-stds_away*h_prior[1]):
        because_of.append('h-')
    if (gt_cube.dimensions[1] > h_prior[0]+stds_away*h_prior[1]):
        because_of.append('h+')
    if (gt_cube.dimensions[2] < l_prior[0]-stds_away*l_prior[1]):
        because_of.append('l-')
    if (gt_cube.dimensions[2] > l_prior[0]+stds_away*l_prior[1]):
        because_of.append('l+')
    if because_of == []:
        return True
    else:
        print('GT cannot be found due to',because_of)
        return False

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


# helper functions for plotting segmentation masks
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask2(masks:np.array, im:np.array, random_color=False):
    """
    Display the masks on top of the image.

    Args:
        masks (np.array): Array of masks with shape (h, w, 4).
        im (np.array): Image with shape (h, w, 3).
        random_color (bool, optional): Whether to use random colors for the masks. Defaults to False.

    Returns:
        np.array: Image with masks displayed on top.
    """
    im_expanded = np.concatenate((im, np.ones((im.shape[0],im.shape[1],1))*255), axis=-1)/255

    mask_image = np.zeros((im.shape[0],im.shape[1],4))
    for i, mask in enumerate(masks):
        if isinstance(random_color, list):
            color = random_color[i]
        else:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_sub = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image + mask_sub
    mask_binary = (mask_image > 0).astype(bool)
    im_out = im_expanded * ~mask_binary + (0.5* mask_image + 0.5 * (im_expanded * mask_binary))
    im_out = im_out.clip(0,1)
    return im_out
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
