from ProposalNetwork.utils.spaces import Cube
from ProposalNetwork.utils.conversions import pixel_to_normalised_space
from ProposalNetwork.utils.utils import make_cube, is_gt_included, make_cubes_parallel
import torch
import numpy as np

def propose_old(reference_box, depth_image, priors, im_shape, number_of_proposals=1, gt_cube=None):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.

    im_shape = [x,y]
    priors = [prior_mean, prior_std] 2x3
    '''
    # Removing the outer 25% on each side of range for center point
    x_stretch = 1.2
    y_stretch = x_stretch * im_shape[1]/im_shape[0]
    d = 4
    x_range_px = [reference_box.x1+reference_box.width/d,reference_box.x2-reference_box.width/d]
    y_range_px = [reference_box.y1+reference_box.height/d,reference_box.y2-reference_box.height/d]

    # Depth grid
    flattened_depth_image = depth_image.flatten()
    percentile_lower = torch.kthvalue(flattened_depth_image, int(0.1 * flattened_depth_image.numel())).values.item()
    percentile_higher = torch.kthvalue(flattened_depth_image, int(0.8 * flattened_depth_image.numel())).values.item()
    z_range = [percentile_lower,percentile_higher]
    z_grid = np.linspace(z_range[0],z_range[1],number_of_proposals) #TODO Scale depth (maybe with focal length)

    # Should also have min and max
    w_prior = torch.tensor([priors[0][0], priors[1][0]])
    h_prior = torch.tensor([priors[0][1], priors[1][1]])
    l_prior = torch.tensor([priors[0][2], priors[1][2]])

    # Check whether it is possible to find gt
    x_range = pixel_to_normalised_space(x_range_px,[im_shape[0],im_shape[0]],[x_stretch * np.mean(z_grid),x_stretch * np.mean(z_grid)])
    y_range = pixel_to_normalised_space(y_range_px,[im_shape[1],im_shape[1]],[y_stretch * np.mean(z_grid),y_stretch * np.mean(z_grid)])
    if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
        pass

    #print('x',x_range,gt_cube.center[0].numpy())
    #print('y',y_range,gt_cube.center[1].numpy())
    print('z',z_range,gt_cube.center[2].numpy())
    list_of_cubes = []
    for i in range(number_of_proposals):
        # Transform center
        x_range = pixel_to_normalised_space(x_range_px,[im_shape[0],im_shape[0]],[x_stretch * z_grid[i],x_stretch * z_grid[i]])
        y_range = pixel_to_normalised_space(y_range_px,[im_shape[1],im_shape[1]],[y_stretch * z_grid[i],y_stretch * z_grid[i]])

        # Predict cube
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,z_grid[i],w_prior,h_prior,l_prior)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)

    # TODO proposal should be different enough from each other, grid search?
    # TODO normals for rotations
    
    return list_of_cubes

def propose(reference_box, depth_image, priors, im_shape, number_of_proposals=1, gt_cube=None):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.

    im_shape = [x,y]
    priors = [prior_mean, prior_std] 2x3
    '''
    # Removing the outer 25% on each side of range for center point
    x_stretch = 1.2
    y_stretch = x_stretch * im_shape[1]/im_shape[0]
    d = 4
    x_range_px = torch.tensor([reference_box.x1+reference_box.width/d,reference_box.x2-reference_box.width/d],device=depth_image.device)
    y_range_px = torch.tensor([reference_box.y1+reference_box.height/d,reference_box.y2-reference_box.height/d],device=depth_image.device)

    # Depth grid
    #z_range = [torch.min(depth_image), torch.max(depth_image)]
    flattened_depth_image = depth_image.flatten()
    percentile_lower = torch.kthvalue(flattened_depth_image, int(0.1 * flattened_depth_image.numel())).values.item()
    percentile_higher = torch.kthvalue(flattened_depth_image, int(0.8 * flattened_depth_image.numel())).values.item()
    z_range = [percentile_lower,percentile_higher]
    z_grid = torch.linspace(z_range[0],z_range[1],number_of_proposals, device=depth_image.device)

    # Should also have min and max
    w_prior = torch.tensor([priors[0][0], priors[1][0]], device=depth_image.device)
    h_prior = torch.tensor([priors[0][1], priors[1][1]], device=depth_image.device)
    l_prior = torch.tensor([priors[0][2], priors[1][2]], device=depth_image.device)

    # Check whether it is possible to find gt
    x_range = pixel_to_normalised_space(x_range_px,[im_shape[0],im_shape[0]],[x_stretch * torch.mean(z_grid),x_stretch * torch.mean(z_grid)])
    y_range = pixel_to_normalised_space(y_range_px,[im_shape[1],im_shape[1]],[y_stretch * torch.mean(z_grid),y_stretch * torch.mean(z_grid)])
    if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
        pass

    list_of_cubes = []

    # Transform center
    #x_range = pixel_to_normalised_space(x_range_px,[im_shape[0],im_shape[0]],[x_stretch * z_grid[i],x_stretch * z_grid[i]])
    #y_range = pixel_to_normalised_space(y_range_px,[im_shape[1],im_shape[1]],[y_stretch * z_grid[i],y_stretch * z_grid[i]])

    # Predict cubes
    pred_xyz, pred_whl, pred_pose = make_cubes_parallel(x_range,y_range,z_grid,w_prior,h_prior,l_prior, number_of_proposals)
    for i in range(number_of_proposals):
        pred_cube = Cube(torch.cat((pred_xyz[i], pred_whl[i]), dim=0),pred_pose[i])
        list_of_cubes.append(pred_cube)

    # TODO proposal should be different enough from each other, grid search?
    # TODO normals for rotations
    
    return list_of_cubes