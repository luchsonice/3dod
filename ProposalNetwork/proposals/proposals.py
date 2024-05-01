from ProposalNetwork.utils import utils
from ProposalNetwork.utils.spaces import Cube, Cubes
from ProposalNetwork.utils.utils import gt_in_norm_range, sample_normal_in_range, vectorized_linspace
from ProposalNetwork.utils.conversions import pixel_to_normalised_space
import torch
import numpy as np
from cubercnn import util
import sys

def propose(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cube=None, ground_normal:np.ndarray=None):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.

    im_shape = [x,y]
    priors = [prior_mean, prior_std] 2x3

    Output:
    cubes : Cubes with (number of proposals) cubes
    stats         : tensor N x number_of_proposals
    '''
    number_of_instances = len(reference_box)

    ####### Center
    # Removing the outer % on each side of range for center point
    m = 4
    widths = reference_box.tensor[:,2] - reference_box.tensor[:,0]
    heights = reference_box.tensor[:,3] - reference_box.tensor[:,1]
    x_range_px = torch.stack((reference_box.tensor[:,0]+widths/m,reference_box.tensor[:,2]-widths/m),dim=1)
    y_range_px = torch.stack((reference_box.tensor[:,1]+heights/m,reference_box.tensor[:,3]-heights/m),dim=1)
    # Find depths
    x_grid_px = vectorized_linspace(x_range_px[:,0],x_range_px[:,1],number_of_proposals).long()
    y_grid_px = vectorized_linspace(y_range_px[:,0],y_range_px[:,1],number_of_proposals).long()
    x_indices = x_grid_px.round()
    y_indices = y_grid_px.round()
    d = depth_image[y_indices, x_indices]
    # Calculate x and y and temporary z
    opposite_side_x = x_grid_px-K[0,2].repeat(number_of_proposals) # x-directional distance in px between image center and object center
    opposite_side_y = y_grid_px-K[1,2].repeat(number_of_proposals) # y-directional distance in px between image center and object center
    adjacent_side = K[0,0].repeat(number_of_proposals) # depth in px to image plane
    angle_x = torch.atan2(opposite_side_x,adjacent_side)
    dx_inside_camera = torch.sqrt(opposite_side_x**2 + adjacent_side**2)
    angle_d = torch.atan2(opposite_side_y,dx_inside_camera)
    y = d * torch.sin(angle_d)
    dx = torch.sqrt(d**2 - y**2)
    x = dx * torch.sin(angle_x)
    z_tmp = torch.sqrt(dx**2 - x**2)

    # Dimensions
    w_prior = [priors[0][:,0], priors[1][:,0]]
    h_prior = [priors[0][:,1], priors[1][:,1]]
    l_prior = [priors[0][:,2], priors[1][:,2]]
    w = sample_normal_in_range(w_prior[0], w_prior[1], number_of_proposals, 0.05, w_prior[0] + 2 * w_prior[1])
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, number_of_proposals, 0.05, h_prior[0] + 2.2 * h_prior[1])
    l = sample_normal_in_range(l_prior[0], l_prior[1], number_of_proposals, 0.05, l_prior[0] + 2 * l_prior[1])

    # Finish center
    def fun(x,coef):
        return coef[0] * x + coef[1]
    # x
    x_coefficients = torch.tensor([1.15, 0])
    x = sample_normal_in_range(fun(torch.median(x,dim=1).values,x_coefficients), torch.std(x,dim=1)*0.7, torch.tensor(number_of_proposals))
    
    # y
    y_coefficients  = torch.tensor([1.1, 0])
    y = sample_normal_in_range(fun(torch.median(y,dim=1).values,y_coefficients), torch.std(y,dim=1)*0.7, number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = torch.tensor([0.85, 0.35])
    z = sample_normal_in_range(fun(torch.median(z,dim=1).values,z_coefficients), torch.std(z,dim=1) * 1.2, number_of_proposals)

    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    if ground_normal is None:
        rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances)
    else:
        angles = torch.linspace(0, np.pi, 36)
        rotation_matrices_inner = utils.orthobasis_from_normal_t(torch.from_numpy(ground_normal), angles)
        rotation_matrices = rotation_matrices_inner[torch.randint(len(rotation_matrices_inner), (number_of_instances,number_of_proposals))]  
    
    # Check whether it is possible to find gt
    # if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
    #    pass

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cube is None:
        return cubes, None, None
    """
    stat_x = gt_in_norm_range([torch.min(x,dim=1),torch.max(x,dim=1)],gt_cube.center[0])
    stat_y = gt_in_norm_range([torch.min(y,dim=1),torch.max(y,dim=1)],gt_cube.center[1])
    stat_z = gt_in_norm_range([torch.min(z,dim=1),torch.max(z,dim=1)],gt_cube.center[2])
    stat_w = gt_in_norm_range([torch.min(w,dim=1),torch.max(w,dim=1)],gt_cube.dimensions[0])
    stat_h = gt_in_norm_range([torch.min(h,dim=1),torch.max(h,dim=1)],gt_cube.dimensions[1])
    stat_l = gt_in_norm_range([torch.min(l,dim=1),torch.max(l,dim=1)],gt_cube.dimensions[2])
    angles = util.mat2euler(gt_cube.rotation)
    stat_rx = gt_in_norm_range(torch.tensor([0,np.pi]),torch.tensor(angles[0]))
    stat_ry = gt_in_norm_range(torch.tensor([0,np.pi/2]),torch.tensor(angles[1]))
    stat_rz = gt_in_norm_range(torch.tensor([0,np.pi]),torch.tensor(angles[2]))

    stats = torch.tensor([stat_x,stat_y,stat_z,stat_w,stat_h,stat_l,stat_rx,stat_ry,stat_rz])
    
    ranges = np.array([torch.std(x,dim=1)*0.8, torch.std(y,dim=1)*0.8, torch.std(z,dim=1)*1.2, w_prior[1], h_prior[1]*1.1, l_prior[1], np.pi,np.pi,np.pi])

    return cubes, stats, ranges
    """
    return cubes, torch.zeros(9), np.ones(9)

"""

#### Other versions that were worse
    
def propose_random_xy(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cube=None, ground_normal=None):
    ####### Center
    # Removing the outer % on each side of range for center point
    m = 4
    x_range_px = torch.tensor([reference_box.tensor[:,0]+reference_box.width/m,reference_box.tensor[:,2]-reference_box.width/m],device=depth_image.device)
    y_range_px = torch.tensor([reference_box.tensor[:,1]+reference_box.height/m,reference_box.tensor[:,3]-reference_box.height/m],device=depth_image.device)
    # Find depths
    x_grid_px = torch.linspace(x_range_px[0],x_range_px[1],number_of_proposals, device=depth_image.device).long()
    y_grid_px = torch.linspace(y_range_px[0],y_range_px[1],number_of_proposals, device=depth_image.device).long()
    x_indices = x_grid_px.round()
    y_indices = y_grid_px.round()
    d = depth_image[y_indices, x_indices]
    # Calculate x and y and temporary z
    opposite_side_x = x_grid_px-K[0,2].repeat(number_of_proposals) # x-directional distance in px between image center and object center
    opposite_side_y = y_grid_px-K[1,2].repeat(number_of_proposals) # y-directional distance in px between image center and object center
    adjacent_side = K[0,0].repeat(number_of_proposals) # depth in px to image plane
    angle_x = torch.atan2(opposite_side_x,adjacent_side)
    dx_inside_camera = torch.sqrt(opposite_side_x**2 + adjacent_side**2)
    angle_d = torch.atan2(opposite_side_y,dx_inside_camera)
    y = d * torch.sin(angle_d)
    dx = torch.sqrt(d**2 - y**2)
    x = dx * torch.sin(angle_x)
    z_tmp = torch.sqrt(dx**2 - x**2)

    # Dimensions
    w_prior = torch.tensor([priors[0][0], priors[1][0]], device=depth_image.device)
    h_prior = torch.tensor([priors[0][1], priors[1][1]], device=depth_image.device)
    l_prior = torch.tensor([priors[0][2], priors[1][2]], device=depth_image.device)
    w = sample_normal_in_range(w_prior[0], w_prior[1], torch.tensor(0.05), w_prior[0] + 2 * w_prior[1], number_of_proposals)
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, torch.tensor(0.05), h_prior[0] + 2.2 * h_prior[1], number_of_proposals)
    l = sample_normal_in_range(l_prior[0], l_prior[1], torch.tensor(0.05), l_prior[0] + 2 * l_prior[1], number_of_proposals)
    whl = torch.stack([w, h, l], 1)

    # Finish center
    def fun(x,coef):
        return coef[0] * x + coef[1]
    
    # xy
    x = torch.rand(1000) * 2 - 1
    y = torch.rand(1000) * 2 - 1
    # z
    z = z_tmp+l/2
    z_coefficients = np.array([0.85, 0.35])
    z = sample_normal_in_range(fun(torch.median(z),z_coefficients), torch.std(z) * 1.2, torch.tensor(-0.5),torch.tensor(100), number_of_proposals)

    xyz = torch.stack([x, y, z], 1)
    
    # Pose
    rotation_matrix = []
    if ground_normal is None:
        rotation_matrix = utils.randn_orthobasis_torch(number_of_proposals).squeeze(0)
    else:
        angles = np.linspace(0, np.pi, 36) # 5 degree steps
        for i in range(number_of_proposals):
            rotation_matrix.append(torch.from_numpy(utils.orthobasis_from_normal(ground_normal, np.random.choice(angles)).astype(np.float32)))

    list_of_cubes = []
    for i in range(number_of_proposals):
        pred_cube = Cube(torch.cat((xyz[i], whl[i]), dim=0),rotation_matrix[i])
        list_of_cubes.append(pred_cube)

    return list_of_cubes, torch.zeros(9), np.ones(9)



def propose_random_xy_patch(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cube=None, ground_normal=None):
    ####### Center
    # Removing the outer % on each side of range for center point
    m = 4
    x_range_px = torch.tensor([reference_box.tensor[:,0]+reference_box.width/m,reference_box.tensor[:,2]-reference_box.width/m],device=depth_image.device)
    y_range_px = torch.tensor([reference_box.tensor[:,1]+reference_box.height/m,reference_box.tensor[:,3]-reference_box.height/m],device=depth_image.device)
    # Find depths
    x_grid_px = torch.linspace(x_range_px[0],x_range_px[1],number_of_proposals, device=depth_image.device).long()
    y_grid_px = torch.linspace(y_range_px[0],y_range_px[1],number_of_proposals, device=depth_image.device).long()
    x_indices = x_grid_px.round()
    y_indices = y_grid_px.round()
    d = depth_image[y_indices, x_indices]
    # Calculate x and y and temporary z
    opposite_side_x = x_grid_px-K[0,2].repeat(number_of_proposals) # x-directional distance in px between image center and object center
    opposite_side_y = y_grid_px-K[1,2].repeat(number_of_proposals) # y-directional distance in px between image center and object center
    adjacent_side = K[0,0].repeat(number_of_proposals) # depth in px to image plane
    angle_x = torch.atan2(opposite_side_x,adjacent_side)
    dx_inside_camera = torch.sqrt(opposite_side_x**2 + adjacent_side**2)
    angle_d = torch.atan2(opposite_side_y,dx_inside_camera)
    y = d * torch.sin(angle_d)
    dx = torch.sqrt(d**2 - y**2)
    x = dx * torch.sin(angle_x)
    z_tmp = torch.sqrt(dx**2 - x**2)

    # Dimensions
    w_prior = torch.tensor([priors[0][0], priors[1][0]], device=depth_image.device)
    h_prior = torch.tensor([priors[0][1], priors[1][1]], device=depth_image.device)
    l_prior = torch.tensor([priors[0][2], priors[1][2]], device=depth_image.device)
    w = sample_normal_in_range(w_prior[0], w_prior[1], torch.tensor(0.05), w_prior[0] + 2 * w_prior[1], number_of_proposals)
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, torch.tensor(0.05), h_prior[0] + 2.2 * h_prior[1], number_of_proposals)
    l = sample_normal_in_range(l_prior[0], l_prior[1], torch.tensor(0.05), l_prior[0] + 2 * l_prior[1], number_of_proposals)
    whl = torch.stack([w, h, l], 1)

    # Finish center
    def fun(x,coef):
        return coef[0] * x + coef[1]
    
    # xy
    [l_x,h_x] = pixel_to_normalised_space([reference_box.tensor[:,0],reference_box.tensor[:,2]],[im_shape[0],im_shape[0]],[2,2])
    [l_y,h_y] = pixel_to_normalised_space([reference_box.tensor[:,1],reference_box.tensor[:,3],],[im_shape[1],im_shape[0]],[1.5,1.5])
    
    x = torch.rand(1000) * (h_x - l_x) + l_x
    y = torch.rand(1000) * (h_y - l_y) + l_y
    # z
    z = z_tmp+l/2
    z_coefficients = np.array([0.85, 0.35])
    z = sample_normal_in_range(fun(torch.median(z),z_coefficients), torch.std(z) * 1.2, torch.tensor(-0.5),torch.tensor(100), number_of_proposals)

    xyz = torch.stack([x, y, z], 1)
    
    # Pose
    rotation_matrix = []
    if ground_normal is None:
        rotation_matrix = utils.randn_orthobasis_torch(number_of_proposals).squeeze(0)
    else:
        angles = np.linspace(0, np.pi, 36) # 5 degree steps
        for i in range(number_of_proposals):
            rotation_matrix.append(torch.from_numpy(utils.orthobasis_from_normal(ground_normal, np.random.choice(angles)).astype(np.float32)))

    list_of_cubes = []
    for i in range(number_of_proposals):
        pred_cube = Cube(torch.cat((xyz[i], whl[i]), dim=0),rotation_matrix[i])
        list_of_cubes.append(pred_cube)

    return list_of_cubes, torch.zeros(9), np.ones(9)

def propose_rand_rotation(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cube=None, ground_normal=None):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.

    im_shape = [x,y]
    priors = [prior_mean, prior_std] 2x3

    Output:
    list_of_cubes : List of Cube
    stats         : tensor N x number_of_proposals
    '''
    ####### Center
    # Removing the outer % on each side of range for center point
    m = 4
    x_range_px = torch.tensor([reference_box.tensor[:,0]+reference_box.width/m,reference_box.tensor[:,2]-reference_box.width/m],device=depth_image.device)
    y_range_px = torch.tensor([reference_box.tensor[:,1]+reference_box.height/m,reference_box.tensor[:,3]-reference_box.height/m],device=depth_image.device)
    # Find depths
    x_grid_px = torch.linspace(x_range_px[0],x_range_px[1],number_of_proposals, device=depth_image.device).long()
    y_grid_px = torch.linspace(y_range_px[0],y_range_px[1],number_of_proposals, device=depth_image.device).long()
    x_indices = x_grid_px.round()
    y_indices = y_grid_px.round()
    d = depth_image[y_indices, x_indices]
    # Calculate x and y and temporary z
    opposite_side_x = x_grid_px-K[0,2].repeat(number_of_proposals) # x-directional distance in px between image center and object center
    opposite_side_y = y_grid_px-K[1,2].repeat(number_of_proposals) # y-directional distance in px between image center and object center
    adjacent_side = K[0,0].repeat(number_of_proposals) # depth in px to image plane
    angle_x = torch.atan2(opposite_side_x,adjacent_side)
    dx_inside_camera = torch.sqrt(opposite_side_x**2 + adjacent_side**2)
    angle_d = torch.atan2(opposite_side_y,dx_inside_camera)
    y = d * torch.sin(angle_d)
    dx = torch.sqrt(d**2 - y**2)
    x = dx * torch.sin(angle_x)
    z_tmp = torch.sqrt(dx**2 - x**2)

    # Dimensions
    w_prior = torch.tensor([priors[0][0], priors[1][0]], device=depth_image.device)
    h_prior = torch.tensor([priors[0][1], priors[1][1]], device=depth_image.device)
    l_prior = torch.tensor([priors[0][2], priors[1][2]], device=depth_image.device)
    w = sample_normal_in_range(w_prior[0], w_prior[1], torch.tensor(0.05), w_prior[0] + 2 * w_prior[1], number_of_proposals)
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, torch.tensor(0.05), h_prior[0] + 2.2 * h_prior[1], number_of_proposals)
    l = sample_normal_in_range(l_prior[0], l_prior[1], torch.tensor(0.05), l_prior[0] + 2 * l_prior[1], number_of_proposals)
    whl = torch.stack([w, h, l], 1)

    # Finish center
    def fun(x,coef):
        return coef[0] * x + coef[1]
    
    # x
    x_coefficients = np.array([1.15, 0])
    x = sample_normal_in_range(fun(torch.median(x),x_coefficients), torch.std(x)*0.7, torch.tensor(-8),torch.tensor(8), number_of_proposals) # TODO Run without limits
    
    # y
    y_coefficients  = np.array([1.1, 0])
    y = sample_normal_in_range(fun(torch.median(y),y_coefficients), torch.std(y)*0.7, torch.tensor(-3),torch.tensor(3), number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = np.array([0.85, 0.35])
    z = sample_normal_in_range(fun(torch.median(z),z_coefficients), torch.std(z) * 1.2, torch.tensor(-0.5),torch.tensor(100), number_of_proposals)

    xyz = torch.stack([x, y, z], 1)
    
    # Pose
    rotation_matrix = utils.randn_orthobasis_torch(number_of_proposals).squeeze(0)
    
    list_of_cubes = []
    for i in range(number_of_proposals):
        pred_cube = Cube(torch.cat((xyz[i], whl[i]), dim=0),rotation_matrix[i])
        list_of_cubes.append(pred_cube)

    return list_of_cubes, torch.zeros(9), np.ones(9)
"""