from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space
from ProposalNetwork.utils.utils import make_cube, is_box_included_in_other_box, iou_2d
import torch

def propose_random(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    x_range = torch.tensor([-0.8,0.8])
    y_range = torch.tensor([-0.8,0.8])
    w_range = torch.tensor([0.2,2])
    h_range = torch.tensor([0.2,2])
    l_range = torch.tensor([0.2,2])

    list_of_cubes = []
    for _ in range(number_of_proposals):
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        list_of_cubes.append(pred_cube)
    
    return list_of_cubes





# TODO changes need to be applied
def propose(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    x_range = pixel_to_normalised_space([reference_box.x1,reference_box.x2],im_shape)[0]
    y_range = pixel_to_normalised_space([reference_box.y1,reference_box.y2],im_shape)[0]
    w_range = torch.tensor([0.2,1])
    h_range = torch.tensor([0.2,1])
    l_range = torch.tensor([0.2,1])

    list_of_cubes = []
    c = 0
    while len(list_of_cubes) < number_of_proposals:
        c += 1
        pred_xyz, pred_whl, pred_pose = make_cube(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        pred_box = cube_to_box(pred_cube,K_scaled)
        if is_box_included_in_other_box(reference_box,pred_box):
            list_of_cubes.append(pred_cube)
    
    print('It took',c,'tries to find',number_of_proposals,'boxes.')
    return list_of_cubes