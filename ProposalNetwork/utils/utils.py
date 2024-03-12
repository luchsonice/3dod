import torch
from spaces import Box, Bube, Cube
from conversions import bube_to_box, cube_to_bube, cube_to_box

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

def make_random_box(x_range, y_range, depth_image, w_range, h_range, l_range, im_shape):
    '''
    need xyz, whl, and pose (R)
    '''
    # xyz
    x = (x_range[0]-x_range[1]) * torch.rand(1) + x_range[1]
    y = (y_range[0]-y_range[1]) * torch.rand(1) + y_range[1]
    z = depth_image[int((x+1)*im_shape[0]/2),int((y+1)*im_shape[1]/2)]
    xyz = torch.tensor([x, y, z])

    # whl
    w = (w_range[0]-w_range[1]) * torch.rand(1) + w_range[1]
    h = (h_range[0]-h_range[1]) * torch.rand(1) + h_range[1]
    l = (l_range[0]-l_range[1]) * torch.rand(1) + l_range[1]
    whl = torch.tensor([w, h, l])

    # R
    rotation_matrix = compute_rotation_matrix_from_ortho6d(torch.rand(6))

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

def propose(reference_box, depth_image, K_scaled, im_shape, number_of_proposals=1):
    x_range = torch.tensor([-0.8,0])
    y_range = torch.tensor([-0.8,0.4])
    w_range = torch.tensor([0.2,1])
    h_range = torch.tensor([0.2,1])
    l_range = torch.tensor([0.2,1])

    list_of_cubes = []
    c = 0
    while len(list_of_cubes) < number_of_proposals:
        c += 1
        print(c)
        pred_xyz, pred_whl, pred_pose = make_random_box(x_range,y_range,depth_image,w_range,h_range,l_range,im_shape)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        pred_box = cube_to_box(pred_cube,K_scaled)
        if is_box_included_in_other_box(reference_box,pred_box):
            list_of_cubes.append(pred_cube)
    
    print('It took',c,'tries to find',number_of_proposals,'boxes.')
    return list_of_cubes




##### Scoring
def intersection_over_proposal_area(gt_box,proposal_box):
    '''
    gt_box: detectron Boxes
    proposal_box: Box
    '''
    proposal_corner = proposal_box.get_all_corners()
    proposal_lt = proposal_corner[0]
    proposal_rb = proposal_corner[2]
    gt_box = gt_box.tensor
    lt = torch.max(gt_box[:,:2],proposal_lt)
    rb = torch.min(gt_box[:,2:],proposal_rb)
    wh = (rb-lt).clamp(min=0)
    i = wh[:,0] * wh[:,1]
    a = proposal_box.width * proposal_box.height
    return i/a
