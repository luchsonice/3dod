from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import pixel_to_normalised_space, normalised_space_to_pixel
from ProposalNetwork.utils.utils import make_cube, is_gt_included
import torch
import numpy as np

def propose_random(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height of at least 0.1 in the normalizes space.
    '''
    x_range = pixel_to_normalised_space([reference_box.x1,reference_box.x2],[im_shape[0],im_shape[0]])[0]
    y_range = pixel_to_normalised_space([reference_box.y1,reference_box.y2],[im_shape[1],im_shape[1]])[0]
    w_range = torch.tensor([0.1,2])
    h_range = torch.tensor([0.1,2])
    l_range = torch.tensor([0.1,2])

    list_of_cubes = []
    for _ in range(number_of_proposals):
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)
    
    return list_of_cubes

def propose(reference_box, depth_image, priors, im_shape, number_of_proposals=1, gt_cube=None):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.

    im_shape = [x,y]
    priors = [prior_mean, prior_std] 2x3
    '''
    # Removing the outer 25% on each side of range for center point
    n = 4
    x_range_px = [reference_box.x1+reference_box.width/n,reference_box.x2-reference_box.width/n]
    x_range = pixel_to_normalised_space(x_range_px,[im_shape[0],im_shape[0]])[0]
    y_range_px = [reference_box.y1+reference_box.height/n,reference_box.y2-reference_box.height/n]
    y_range = pixel_to_normalised_space(y_range_px,[im_shape[1],im_shape[1]])[0]

    # Depth grid
    z_range = [depth_image.min(), depth_image.max()]
    z_grid = np.linspace(z_range[0],z_range[1],number_of_proposals)

    
    # Should also have min and max
    w_prior = torch.tensor([priors[0][0], priors[1][0]])
    h_prior = torch.tensor([priors[0][1], priors[1][1]])
    l_prior = torch.tensor([priors[0][2], priors[1][2]])

    #print(x_range,y_range,z_range,w_prior,h_prior,l_prior)

    # Check whether it is possible to find gt
    if not (gt_cube == None) and not is_gt_included(gt_cube,x_range, y_range, z_range, w_prior, h_prior, l_prior):
        print('GT cannot be found!')

    list_of_cubes = []
    for i in range(number_of_proposals):
        # Predict cube
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,z_grid[i],w_prior,h_prior,l_prior)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)

    # TODO proposal should be different enough from each other, grid search?
    # TODO normals for rotations
    
    return list_of_cubes