from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space
from ProposalNetwork.utils.utils import make_cube, is_box_included_in_other_box, iou_2d
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

def propose_tigther_center(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    x1 = reference_box.x1+reference_box.width/5
    x2 = reference_box.x2-reference_box.width/5
    y1 = reference_box.y1+reference_box.height/5
    y2 = reference_box.y2-reference_box.height/5
    x_range = pixel_to_normalised_space([x1,x2],[im_shape[0],im_shape[0]])[0]
    y_range = pixel_to_normalised_space([y1,y2],[im_shape[1],im_shape[1]])[0]
    w_range = torch.tensor([0.2,1])
    h_range = torch.tensor([0.2,1])
    l_range = torch.tensor([0.2,1])

    list_of_cubes = []
    for _ in range(number_of_proposals):
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)
    
    return list_of_cubes

def propose_tighter_dimensions(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.
    '''
    x_range = pixel_to_normalised_space([reference_box.x1,reference_box.x2],[im_shape[0],im_shape[0]])[0]
    y_range = pixel_to_normalised_space([reference_box.y1,reference_box.y2],[im_shape[1],im_shape[1]])[0]
    # TODO Should be priors in some way. This is still too random for a realistic approach
    w_range = torch.tensor([0.1,reference_box.width])
    h_range = torch.tensor([0.1,reference_box.height])
    l_range = torch.tensor([0.1,torch.min(2, reference_box.width + reference_box.height)])

    list_of_cubes = []
    for _ in range(number_of_proposals):
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)
    
    return list_of_cubes

def propose(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    '''
    Proposes a cube. The ranges are largely random, except for that the center needs to be inside the reference box.
    Also, objects have a length, width and height according to priors.
    '''
    x_range = pixel_to_normalised_space([reference_box.x1,reference_box.x2],[im_shape[0],im_shape[0]])[0]
    y_range = pixel_to_normalised_space([reference_box.y1,reference_box.y2],[im_shape[1],im_shape[1]])[0]

    # TODO Should be priors in some way. This is still too random for a realistic approach
    width = x_range[1]-x_range[0]
    height = y_range[1]-y_range[0]
    w_range = torch.tensor([0.1,width])
    h_range = torch.tensor([0.1,height])
    l_range = torch.tensor([0.1,height])

    # Don't know if that is a good idea, i.e. removing the outer 20% on each side of the center proposals
    x_range = [x_range[0]+width/5,x_range[1]-width/5]
    y_range = [y_range[0]+height/5,y_range[1]-height/5]

    list_of_cubes = []
    for _ in range(number_of_proposals):
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)
    
    return list_of_cubes