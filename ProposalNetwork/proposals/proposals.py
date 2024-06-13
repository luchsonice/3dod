from ProposalNetwork.utils import utils
from ProposalNetwork.utils.spaces import Cubes
from ProposalNetwork.utils.utils import gt_in_norm_range, sample_normal_in_range, vectorized_linspace
from ProposalNetwork.utils.conversions import pixel_to_normalised_space
import torch
import numpy as np
from cubercnn import util

# 0.0x meters is the minimum edge length
MIN_PROP_S = 0.05

def rescale_interval(x, min, max):
    '''operation  (min - max) * x + max'''
    return (min - max) * x + max

def lin_fun(x,coef):
    '''used for finishing the center of the cube proposal. The center is calculated as a linear function (typically of the depth image).'''
    return coef[0] * x + coef[1]

def propose_random(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
    number_of_instances = len(reference_box)
    # Center
    x = torch.rand(number_of_instances,number_of_proposals, device=reference_box.device) * 4 - 2
    y = torch.rand(number_of_instances,number_of_proposals, device=reference_box.device) * 2 - 1
    z = torch.rand(number_of_instances,number_of_proposals, device=reference_box.device) * 4 + 1

    # Dimensions
    w = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)
    h = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)
    l = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)

    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(reference_box.device)

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,x,y,z,w,h,l)

    return cubes, stats, torch.ones(cubes.num_instances,9)

def propose_xy_patch(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
    '''
    only propose x and y values that are within the reference box'''
    number_of_instances = len(reference_box)

    # Center
    m = 4
    widths = reference_box.tensor[:,2] - reference_box.tensor[:,0]
    heights = reference_box.tensor[:,3] - reference_box.tensor[:,1]
    x_min, x_max = reference_box.tensor[:,0]+widths/m, reference_box.tensor[:,2]-widths/m
    y_min, y_max = reference_box.tensor[:,1]+heights/m, reference_box.tensor[:,3]-heights/m
    x_tensor = pixel_to_normalised_space([x_min, x_max],[im_shape[0],im_shape[0]],[2,2])
    y_tensor = pixel_to_normalised_space([y_min, y_max],[im_shape[1],im_shape[0]],[1.5,1.5])
    l_x = x_tensor[:,0]
    h_x = x_tensor[:,1]
    l_y = y_tensor[:,0]
    h_y = y_tensor[:,1]

    x = torch.rand(number_of_instances,number_of_proposals, device=h_x.device)
    y = torch.rand(number_of_instances,number_of_proposals, device=h_x.device)
    # rescale the numbers to the "normalised space"
    xt = rescale_interval(x, (h_x).view(-1, 1), l_x.view(-1, 1))
    yt = rescale_interval(y, (h_y).view(-1, 1), l_y.view(-1, 1))
    z = torch.rand(number_of_instances,number_of_proposals, device=reference_box.device) * 4 + 1

    # Dimensions
    # constrain to interval [MIN_PROP_S, 2] meters 
    w = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=h_x.device), MIN_PROP_S, 2)
    h = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=h_x.device), MIN_PROP_S, 2)
    l = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=h_x.device), MIN_PROP_S, 2)

    xyzwhl = torch.stack([xt, yt, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=h_x.device)

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,xt,yt,z,w,h,l)

    return cubes, stats, torch.ones(cubes.num_instances,9)

def propose_z(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
    '''
    picke a random x and y spot anywhere on the image and grab the z-value from that spot'''
    number_of_instances = len(reference_box)

    # Center
    m = 4
    widths = reference_box.tensor[:,2] - reference_box.tensor[:,0]
    heights = reference_box.tensor[:,3] - reference_box.tensor[:,1]
    x_min, x_max = reference_box.tensor[:,0]+widths/m, reference_box.tensor[:,2]-widths/m
    y_min, y_max = reference_box.tensor[:,1]+heights/m, reference_box.tensor[:,3]-heights/m
    x_tensor = pixel_to_normalised_space([x_min, x_max],[im_shape[0],im_shape[0]],[2,2])
    y_tensor = pixel_to_normalised_space([y_min, y_max],[im_shape[1],im_shape[0]],[1.5,1.5])
    l_x = x_tensor[:,0]
    h_x = x_tensor[:,1]
    l_y = y_tensor[:,0]
    h_y = y_tensor[:,1]

    x = torch.rand(number_of_instances,number_of_proposals, device=h_x.device)
    y = torch.rand(number_of_instances,number_of_proposals, device=h_x.device)
    # rescale the numbers to the "normalised space"
    xt = rescale_interval(x, (h_x).view(-1, 1), l_x.view(-1, 1))
    yt = rescale_interval(y, (h_y).view(-1, 1), l_y.view(-1, 1))
    z = torch.zeros_like(x)
    for i in range(number_of_instances):
        z_depth_patch = depth_image[int(reference_box.tensor[i,1]):int(reference_box.tensor[i,3]), int(reference_box.tensor[i,0]):int(reference_box.tensor[i,2])]
        quantiles = torch.quantile(z_depth_patch, torch.tensor([0.1, 0.9],device=z_depth_patch.device), dim=None)
        z[i] = torch.linspace(quantiles[0],quantiles[1],number_of_proposals)

    # Dimensions
    w = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=x.device), MIN_PROP_S, 2)
    h = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=x.device), MIN_PROP_S, 2)
    l = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=x.device), MIN_PROP_S, 2)

    xyzwhl = torch.stack([xt, yt, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=x.device)

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,xt,yt,z,w,h,l)

    return cubes, stats, torch.ones(cubes.num_instances,9)

def propose_random_dim(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
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
    w = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)
    h = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)
    l = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)

    # Finish center
    # x
    x_coefficients = torch.tensor([1.15, 0])
    x = sample_normal_in_range(lin_fun(torch.median(x,dim=1).values,x_coefficients), torch.std(x,dim=1)*0.7, torch.tensor(number_of_proposals))
    
    # y
    y_coefficients  = torch.tensor([1.1, 0])
    y = sample_normal_in_range(lin_fun(torch.median(y,dim=1).values,y_coefficients), torch.std(y,dim=1)*0.7, number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = torch.tensor([0.85, 0.35])
    z = sample_normal_in_range(lin_fun(torch.median(z,dim=1).values,z_coefficients), torch.std(z,dim=1) * 1.2, number_of_proposals)

    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=reference_box.device)

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,x,y,z,w,h,l)

    return cubes, stats, torch.ones(cubes.num_instances,9)

def propose_aspect_ratio(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
    '''    
    sample width from the prior and then apply a set of ratios on h. Then take a random shuffled version of the set and apply it to L.
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
    w = rescale_interval(torch.rand(number_of_instances,number_of_proposals, device=reference_box.device), MIN_PROP_S, 2)
    #
    ratios = [0.33, 0.66, 1, 1.33, 1.67, 2, 3]
    h = torch.zeros_like(w)
    l = torch.zeros_like(w)
    for i in range(number_of_instances):
        # must
        ratio1, ratio2 = torch.randperm(len(ratios))[0], torch.randperm(len(ratios))[0]
        h[i] = w[i] * ratios[ratio1]
        l[i] = w[i] * ratios[ratio2]

    # Finish center
    # x
    x_coefficients = torch.tensor([1.15, 0])
    x = sample_normal_in_range(lin_fun(torch.median(x,dim=1).values,x_coefficients), torch.std(x,dim=1)*0.7, torch.tensor(number_of_proposals))
    
    # y
    y_coefficients  = torch.tensor([1.1, 0])
    y = sample_normal_in_range(lin_fun(torch.median(y,dim=1).values,y_coefficients), torch.std(y,dim=1)*0.7, number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = torch.tensor([0.85, 0.35])
    z = sample_normal_in_range(lin_fun(torch.median(z,dim=1).values,z_coefficients), torch.std(z,dim=1) * 1.2, number_of_proposals)

    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=reference_box.device)

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,x,y,z,w,h,l)

    return cubes, stats, torch.ones(cubes.num_instances,9)


def propose_random_rotation(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
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
    w = sample_normal_in_range(w_prior[0], w_prior[1], number_of_proposals, MIN_PROP_S, w_prior[0] + 2 * w_prior[1])
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, number_of_proposals, MIN_PROP_S, h_prior[0] + 2.2 * h_prior[1])
    l = sample_normal_in_range(l_prior[0], l_prior[1], number_of_proposals, MIN_PROP_S, l_prior[0] + 2 * l_prior[1])

    # x
    x_coefficients = torch.tensor([1.15, 0], device=reference_box.device)
    x = sample_normal_in_range(lin_fun(torch.median(x,dim=1).values,x_coefficients), torch.std(x,dim=1)*0.7, torch.tensor(number_of_proposals, device=reference_box.device))
    
    # y
    y_coefficients  = torch.tensor([1.1, 0], device=reference_box.device)
    y = sample_normal_in_range(lin_fun(torch.median(y,dim=1).values,y_coefficients), torch.std(y,dim=1)*0.7, number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = torch.tensor([0.85, 0.35], device=reference_box.device)
    z = sample_normal_in_range(lin_fun(torch.median(z,dim=1).values,z_coefficients), torch.std(z,dim=1) * 1.2, number_of_proposals)

    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=reference_box.device)
    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,x,y,z,w,h,l)

    n = gt_cubes.num_instances
    ranges = torch.stack([torch.std(x,dim=1)*0.8, torch.std(y,dim=1)*0.8, torch.std(z,dim=1)*1.2, w_prior[1], h_prior[1]*1.1, l_prior[1], torch.tensor(torch.pi,device=reference_box.device).repeat(n),torch.tensor(torch.pi,device=reference_box.device).repeat(n),torch.tensor(torch.pi,device=reference_box.device).repeat(n)],dim=1).cpu().numpy()

    return cubes, stats, ranges

def propose(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cubes=None, ground_normal:torch.Tensor=None):
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
    w = sample_normal_in_range(w_prior[0], w_prior[1], number_of_proposals, MIN_PROP_S, w_prior[0] + 2 * w_prior[1])
    h = sample_normal_in_range(h_prior[0], h_prior[1]*1.1, number_of_proposals, MIN_PROP_S, h_prior[0] + 2.2 * h_prior[1])
    l = sample_normal_in_range(l_prior[0], l_prior[1], number_of_proposals, MIN_PROP_S, l_prior[0] + 2 * l_prior[1])

    # x
    x_coefficients = torch.tensor([1.15, 0], device=reference_box.device)
    x = sample_normal_in_range(lin_fun(torch.median(x,dim=1).values,x_coefficients), torch.std(x,dim=1)*0.7, torch.tensor(number_of_proposals, device=reference_box.device))
    
    # y
    y_coefficients  = torch.tensor([1.1, 0], device=reference_box.device)
    y = sample_normal_in_range(lin_fun(torch.median(y,dim=1).values,y_coefficients), torch.std(y,dim=1)*0.7, number_of_proposals)
    
    # z
    z = z_tmp+l/2
    z_coefficients = torch.tensor([0.85, 0.35], device=reference_box.device)
    z = sample_normal_in_range(lin_fun(torch.median(z,dim=1).values,z_coefficients), torch.std(z,dim=1) * 1.2, number_of_proposals)
    #z = gt_cubes.tensor[:,0,2].view(-1,1).repeat(1,number_of_proposals)
    xyzwhl = torch.stack([x, y, z, w, h, l], 2)
    
    # Pose
    if ground_normal is None:
        rotation_matrices = utils.randn_orthobasis_torch(number_of_proposals, number_of_instances).to(device=reference_box.device)
    else:
        ground_normal = ground_normal.to(device=reference_box.device)
        angles = torch.linspace(0, np.pi, 36, device=ground_normal.device)
        rotation_matrices_inner = utils.orthobasis_from_normal_t(ground_normal, angles)
        rotation_matrices = rotation_matrices_inner[torch.randint(len(rotation_matrices_inner), (number_of_instances,number_of_proposals))]  
    
    # Check whether it is possible to find gt
    # if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
    #    pass

    cubes = Cubes(torch.cat((xyzwhl, rotation_matrices.flatten(start_dim=2)), dim=2))

    # Statistics
    if gt_cubes is None:
        return cubes, None, None
    
    stats = statistics(gt_cubes,x,y,z,w,h,l)

    n = gt_cubes.num_instances
    ranges = torch.stack([torch.std(x,dim=1)*0.8, torch.std(y,dim=1)*0.8, torch.std(z,dim=1)*1.2, w_prior[1], h_prior[1]*1.1, l_prior[1], torch.tensor(torch.pi,device=reference_box.device).repeat(n),torch.tensor(torch.pi,device=reference_box.device).repeat(n),torch.tensor(torch.pi,device=reference_box.device).repeat(n)],dim=1).cpu().numpy()

    return cubes, stats, ranges


def statistics(gt_cubes,x,y,z,w,h,l):    
    n = gt_cubes.num_instances
    stats = torch.zeros((n,9))
    for i in range(n):
        gt_cube = gt_cubes[i].tensor[0,0]
        stat_x = gt_in_norm_range([torch.min(x[i]),torch.max(x[i])],gt_cube[0])
        stat_y = gt_in_norm_range([torch.min(y[i]),torch.max(y[i])],gt_cube[1])
        stat_z = gt_in_norm_range([torch.min(z[i]),torch.max(z[i])],gt_cube[2])
        stat_w = gt_in_norm_range([torch.min(w[i]),torch.max(w[i])],gt_cube[3])
        stat_h = gt_in_norm_range([torch.min(h[i]),torch.max(h[i])],gt_cube[4])
        stat_l = gt_in_norm_range([torch.min(l[i]),torch.max(l[i])],gt_cube[5])
        angles = util.mat2euler(gt_cube[-9:].reshape((3,3)))
        stat_rx = gt_in_norm_range(torch.tensor([0,np.pi]),torch.tensor(angles[0]))
        stat_ry = gt_in_norm_range(torch.tensor([0,np.pi/2]),torch.tensor(angles[1]))
        stat_rz = gt_in_norm_range(torch.tensor([0,np.pi]),torch.tensor(angles[2]))

        stats[i] = torch.tensor([stat_x,stat_y,stat_z,stat_w,stat_h,stat_l,stat_rx,stat_ry,stat_rz])

    return stats