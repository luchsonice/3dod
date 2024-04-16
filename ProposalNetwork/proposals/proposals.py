from ProposalNetwork.utils.spaces import Cube
from ProposalNetwork.utils.utils import gt_in_norm_range, sample_normal_greater_than_para
import torch
import numpy as np
from cubercnn import util

def propose(reference_box, depth_image, priors, im_shape, K, number_of_proposals=1, gt_cube=None):
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
    x_range_px = torch.tensor([reference_box.x1+reference_box.width/m,reference_box.x2-reference_box.width/m],device=depth_image.device)
    y_range_px = torch.tensor([reference_box.y1+reference_box.height/m,reference_box.y2-reference_box.height/m],device=depth_image.device)
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
    w = sample_normal_greater_than_para(w_prior[0], w_prior[1], torch.tensor(0.05), w_prior[0] + 2 * w_prior[1], number_of_proposals)
    h = sample_normal_greater_than_para(h_prior[0], h_prior[1], torch.tensor(0.05), h_prior[0] + 2 * h_prior[1], number_of_proposals)
    l = sample_normal_greater_than_para(l_prior[0], l_prior[1], torch.tensor(0.05), l_prior[0] + 2 * l_prior[1], number_of_proposals)
    whl = torch.stack([w, h, l], 1)

    # Finish z
    z = z_tmp+l
    z = sample_normal_greater_than_para(torch.median(z), torch.std(z) * 1.2, torch.tensor(0),torch.tensor(100), number_of_proposals)

    x *= 1.3
    y *= 1.4
    x_width = torch.max(x) - torch.min(x)
    x = sample_normal_greater_than_para(torch.min(x) + x_width/2, torch.std(x), torch.min(x),torch.max(x), number_of_proposals)
    y_width = torch.max(y) - torch.min(y)
    y = sample_normal_greater_than_para(torch.min(y) + y_width/2, torch.std(y), torch.min(y),torch.max(y), number_of_proposals)
    xyz = torch.stack([x, y, z], 1)
    
    """
    print(f"{torch.min(x).item():.2f},{torch.max(x).item():.2f},{torch.median(x).item():.2f}\033[1m,{gt_cube.center[0].item():.2f}\033[0m,{torch.mean(x).item():.2f}")
    import matplotlib.pyplot as plt
    import os
    num_bins = 40
    plt.figure(figsize=(15, 15))
    plt.hist(x.numpy(), bins=num_bins, color='darkslategrey')
    plt.axvline(x=gt_cube.center[0].item(), color='red')
    plt.savefig(os.path.join('ProposalNetwork/output/MABO', 'tmp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    """
    
    # Pose
    rotation_matrix = torch.tensor([[0.5,0,0.8660254],[0,1,0],[-0.8660254,0,0.5]])

    # Check whether it is possible to find gt
    # if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
    #    pass


    list_of_cubes = []

    for i in range(number_of_proposals):
        pred_cube = Cube(torch.cat((xyz[i], whl[i]), dim=0),rotation_matrix)
        list_of_cubes.append(pred_cube)
    
    # Statistics
    stds = 2
    stat_x = gt_in_norm_range([torch.min(x),torch.max(x)],gt_cube.center[0])
    stat_y = gt_in_norm_range([torch.min(y),torch.max(y)],gt_cube.center[1])
    stat_z = gt_in_norm_range([torch.min(z),torch.max(z)],gt_cube.center[2])
    stat_w = gt_in_norm_range([torch.max(w_prior[0]-stds*w_prior[1],torch.tensor(0.2)),torch.min(w_prior[0]+stds*w_prior[1],w_prior[0]+2*w_prior[1])],gt_cube.dimensions[0])
    stat_h = gt_in_norm_range([torch.max(h_prior[0]-stds*h_prior[1],torch.tensor(0.05)),torch.min(h_prior[0]+stds*h_prior[1],h_prior[0]+2*h_prior[1])],gt_cube.dimensions[1])
    stat_l = gt_in_norm_range([torch.max(l_prior[0]-stds*l_prior[1],torch.tensor(0.05)),torch.min(l_prior[0]+stds*l_prior[1],l_prior[0]+2*l_prior[1])],gt_cube.dimensions[2])
    angles = util.mat2euler(gt_cube.rotation)
    stat_rx = gt_in_norm_range(torch.tensor([-np.pi,np.pi]),torch.tensor(angles[0]))
    stat_ry = gt_in_norm_range(torch.tensor([-np.pi,np.pi]),torch.tensor(angles[1]))
    stat_rz = gt_in_norm_range(torch.tensor([-np.pi,np.pi]),torch.tensor(angles[2]))

    stats = torch.tensor([stat_x,stat_y,stat_z,stat_w,stat_h,stat_l,stat_rx,stat_ry,stat_rz])

    return list_of_cubes, stats