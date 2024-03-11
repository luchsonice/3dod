import torch
from spaces import Box, Bube, Cube
from conversions import bube_to_box, cube_to_bube, cube_to_box

# batch*n
def normalize_vector(v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag

    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3

    return matrix[1] # TODO at some point all above should be converted to only output one rotation matrix

def make_random_box(x_range, y_range, depth_image, w_range, h_range, l_range):
    '''
    need xyz, whl, and pose (R)
    TODO Maybe change such that input is in meters and then converts to vialble range. One problem right now is that z always will be depth of {(0,0),(0,1),(1,0),(1,1)}
    '''
    # xyz
    x = (x_range[0]-x_range[1]) * torch.rand(1) + x_range[1]
    y = (y_range[0]-y_range[1]) * torch.rand(1) + y_range[1]
    z = depth_image[int(x),int(y)]
    xyz = torch.tensor([x, y, z])

    # whl
    w = (w_range[0]-w_range[1]) * torch.rand(1) + w_range[1]
    h = (h_range[0]-h_range[1]) * torch.rand(1) + h_range[1]
    l = (l_range[0]-l_range[1]) * torch.rand(1) + l_range[1]
    whl = torch.tensor([w, h, l])

    # R
    rotation_matrix = compute_rotation_matrix_from_ortho6d(torch.vstack([torch.zeros(6),torch.rand(6)]))

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

def propose(reference_box, depth_image, K_scaled, number_of_proposals=1):
    x_range = torch.tensor([-0.8,0.8])
    y_range = torch.tensor([-0.8,0.8])
    w_range = torch.tensor([0.1,1.4])
    h_range = torch.tensor([0.1,1.4])
    l_range = torch.tensor([0.1,1.4])

    list_of_cubes = []
    while len(list_of_cubes) < number_of_proposals:
        pred_xyz, pred_whl, pred_pose = make_random_box(x_range,y_range,depth_image,w_range,h_range,l_range)
        pred_cube = Cube(torch.cat((pred_xyz, pred_whl), dim=0),pred_pose)
        pred_box = cube_to_box(pred_cube,K_scaled)
        if is_box_included_in_other_box(reference_box,pred_box):
            list_of_cubes.append(pred_cube)
    
    return list_of_cubes


