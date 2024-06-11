import torch
import numpy as np
import matplotlib.pyplot as plt
#import open3d as o3d

from detectron2.structures import pairwise_iou
from pytorch3d.ops import box3d_overlap

##### Proposal
def normalize_vector(v):
    v_mag = torch.sqrt(v.pow(2).sum())
    v_mag = torch.max(v_mag, torch.tensor([1e-8], device=v.device))
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

def sample_normal_in_range(means, stds, count, threshold_low=None, threshold_high=None):
    device = means.device
    # Generate samples from a normal distribution
    samples = torch.normal(means.unsqueeze(1).expand(-1,count), stds.unsqueeze(1).expand(-1,count))

    # Ensure that all samples are greater than threshold_low and less than threshold_high
    if threshold_high is not None and threshold_low is not None:
        tries = 0
        threshold_high = threshold_high.unsqueeze(1).expand_as(samples)
        while torch.any((samples < threshold_low) | (samples > threshold_high)):
            invalid_mask = (samples < threshold_low) | (samples > threshold_high)
            # Replace invalid samples with new samples drawn from the normal distribution, could be done more optimal by sampling only sum(invalid mask) new samples, but matching of correct means is difficult then
            samples[invalid_mask] = torch.normal(means.unsqueeze(1).expand(-1,count), stds.unsqueeze(1).expand(-1,count))[invalid_mask]
            
            tries += 1
            if tries == 10000:
                break

    return samples.to(device)

def randn_orthobasis_torch(num_samples=1,num_instances=1):
    z = torch.randn(num_instances, num_samples, 3, 3)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    z[:, :, 0] = torch.cross(z[:, :, 1], z[:, :, 2], dim=-1)
    z[:, :, 0] = z[:, :, 0] / torch.norm(z[:, :, 0], dim=-1, keepdim=True)
    z[:, :, 1] = torch.cross(z[:, :, 2], z[:, :, 0], dim=-1)
    z[:, :, 1] = z[:, :, 1] / torch.norm(z[:, :, 1], dim=-1, keepdim=True)
    return z

def randn_orthobasis(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    z[:, 0] = np.cross(z[:, 1], z[:, 2], axis=-1)
    z[:, 0] = z[:, 0] / np.linalg.norm(z[:, 0], axis=-1, keepdims=True)
    z[:, 1] = np.cross(z[:, 2], z[:, 0], axis=-1)
    z[:, 1] = z[:, 1] / np.linalg.norm(z[:, 1], axis=-1, keepdims=True)
    return z

# ##things for making rotations
def vec_perp(vec):
    '''generate a vector perpendicular to vec in 3d'''
    # https://math.stackexchange.com/a/2450825
    a, b, c = vec
    if a == 0:
        return np.array([0,c,-b])
    return np.array(normalize_vector(torch.tensor([b,-a,0])))

def orthobasis_from_normal(normal, yaw_angle=0):
    '''generate an orthonormal/Rotation matrix basis from a normal vector in 3d
     
       returns a 3x3 matrix with the basis vectors as columns, 3rd column is the original normal vector
    '''
    x = rotate_vector(vec_perp(normal), normal, yaw_angle)
    x = x / np.linalg.norm(x, ord=2)
    y = np.cross(normal, x)
    return np.array([x, normal, y]).T # the vectors should be as columns

def rotate_vector(v, k, theta):
    '''rotate a vector v around an axis k by an angle theta
    it is assumed that k is a unit vector (p2 norm = 1)'''
    # https://medium.com/@sim30217/rodrigues-rotation-formula-47489db49050
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1 - cos_theta)
    
    return term1 + term2 + term3

def vec_perp_t(vec):
    '''generate a vector perpendicular to vec in 3d'''
    # https://math.stackexchange.com/a/2450825
    a, b, c = vec
    if a == 0:
        return torch.tensor([0,c,-b], device=vec.device)
    return normalize_vector(torch.tensor([b,-a,0], device=vec.device))

def orthobasis_from_normal_t(normal:torch.Tensor, yaw_angles:torch.Tensor=0):
    '''generate an orthonormal/Rotation matrix basis from a normal vector in 3d

        normal is assumed to be normalised 
     
       returns a (no. of yaw_angles)x3x3 matrix with the basis vectors as columns, 3rd column is the original normal vector
    '''
    n = len(yaw_angles)
    x = rotate_vector_t(vec_perp_t(normal), normal, yaw_angles)
    # x = x / torch.norm(x, p=2)
    y = torch.cross(normal.view(-1,1), x)
    # y = y / torch.norm(y, p=2, dim=1)
    return torch.cat([x.t(), normal.unsqueeze(0).repeat(n, 1), y.t()],dim=1).reshape(n,3,3).transpose(2,1) # the vectors should be as columns

def rotate_vector_t(v, k, theta):
    '''rotate a vector v around an axis k by an angle theta
    it is assumed that k is a unit vector (p2 norm = 1)'''
    # https://medium.com/@sim30217/rodrigues-rotation-formula-47489db49050
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    v2 = v.view(-1,1)

    term1 = v2 * cos_theta
    term2 = torch.cross(k, v).view(-1, 1) * sin_theta
    term3 = (k * (k @ v)).view(-1, 1) * (1 - cos_theta)
    
    return (term1 + term2 + term3)

# ########### End rotations
def gt_in_norm_range(range,gt):
    tmp = gt-range[0]
    res = tmp / abs(range[1] - range[0])

    return res

    if range[0] > 0: # both positive
        tmp = gt-range[0]
        res = tmp / abs(range[1] - range[0])
    elif range[1] > 0: # lower negative upper positive
        if gt > 0:
            tmp = gt-range[0]
        else:
            tmp = range[1]-gt
        res = tmp / abs(range[1] - range[0])
    else: # both negative
        tmp = range[1]-gt
        res = tmp / abs(range[1] - range[0])

    return res

def vectorized_linspace(start_tensor, end_tensor, number_of_steps):
    # Calculate spacing
    spacing = (end_tensor - start_tensor) / (number_of_steps - 1)
    # Create linear spaces with arange
    linear_spaces = torch.arange(start=0, end=number_of_steps, dtype=start_tensor.dtype, device=start_tensor.device)
    linear_spaces = linear_spaces.repeat(start_tensor.size(0),1)
    linear_spaces = linear_spaces * spacing[:,None] + start_tensor[:,None]
    return linear_spaces







##### Scoring
def iou_2d(gt_box, proposal_boxes):
    '''
    gt_box: Boxes
    proposal_box: Boxes
    '''
    IoU = pairwise_iou(gt_box,proposal_boxes).flatten()
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
    gt_corners = gt_cube.get_all_corners()[0]
    proposal_corners = proposal_cubes.get_all_corners()[0]
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

def mask_iou(segmentation_mask, bube_mask):
    '''
    Area is of segmentation_mask
    '''
    bube_mask = torch.tensor(bube_mask, device=segmentation_mask.device)
    intersection = (segmentation_mask * bube_mask).sum()
    if intersection == 0:
        return torch.tensor(0.0)
    union = torch.logical_or(segmentation_mask, bube_mask).to(torch.int).sum()
    return intersection / union

def mod_mask_iou(segmentation_mask, bube_mask):
    '''
    Area is of segmentation_mask
    '''
    bube_mask = torch.tensor(bube_mask, device=segmentation_mask.device)
    intersection = (segmentation_mask * bube_mask).sum()
    if intersection == 0:
        return torch.tensor(0.0)
    union = torch.logical_or(segmentation_mask, bube_mask).to(torch.int).sum()
    return intersection**5 / union # NOTE not standard IoU

def mask_iou_loss(segmentation_mask, bube_mask):
    '''
    Area is of segmentation_mask
    '''
    intersection = (segmentation_mask * bube_mask).sum()
    if intersection == 0:
        return torch.tensor(0.0)
    union = torch.logical_or(segmentation_mask, bube_mask).to(torch.int).sum()
    return intersection / union

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















# Convex Hull
import torch

def jarvis_march(points):
    # Number of points
    n = points.size(0)
    # List to store the convex hull vertices
    hull = []
    
    # Find the leftmost point
    leftmost = points[torch.argmin(points[:, 0])]
    point_on_hull = leftmost
    
    while True:
        hull.append(point_on_hull)
        endpoint = points[0]
        
        for j in range(1, n):
            if torch.equal(endpoint, point_on_hull) or is_left_turn(point_on_hull, endpoint, points[j]):
                endpoint = points[j]
        
        point_on_hull = endpoint
        
        if torch.equal(endpoint, leftmost):
            break
    
    return torch.stack(hull)

def is_left_turn(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0

def fill_polygon(mask, polygon):
    h, w = mask.shape
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid_coords = torch.stack([X.flatten(), Y.flatten()], dim=1).float().to(mask.device)
    
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    
    for i in range(len(polygon)):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % len(polygon)]
        
        # Determine the direction of the edge
        edge_direction = v2 - v1
        
        # Check if the point is to the left of the edge
        is_left = (grid_coords[:, 0] - v1[0]) * edge_direction[1] - (grid_coords[:, 1] - v1[1]) * edge_direction[0] > 0
        
        # Count the number of times the ray from the point intersects the edge
        num_intersections = is_left.view(h, w).sum(dim=0) % 2
        
        # If the number of intersections is odd, the point is inside the polygon
        new_mask |= num_intersections.bool()
    
    return new_mask

def convex_hull(mask):
    coords = torch.nonzero(mask).float()
    hull = jarvis_march(coords)
    new_mask = fill_polygon(mask, hull)

    return new_mask
