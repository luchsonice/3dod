from ProposalNetwork.proposals.proposals import setup_depth_model, depth_of_images

from spaces import Box, Bube, Cube
from conversions import bube_to_box, cube_to_bube, cube_to_box
from utils import compute_rotation_matrix_from_ortho6d, make_random_box, propose, intersection_over_proposal_area, custom_mapping

import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np

from cubercnn import util, vis
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer

#torch.manual_seed(1)

"""
R = torch.eye(3)
cube = Cube(torch.tensor([5,5,10,2,2,4]),R)
print('cube',cube.get_all_corners())

K = torch.eye(3)
bube = cube_to_bube(cube,K)
bube_corners = bube.get_all_corners()
print('bube',bube_corners)

box = bube_to_box(bube)
print('box',box.get_all_corners())

# Plot bube on 2D plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(torch.cat((bube_corners[:4,0],bube_corners[0,0].reshape(1))),torch.cat((bube_corners[:4,1],bube_corners[0,1].reshape(1))),color='r',linewidth=3)
ax.plot(torch.cat((bube_corners[4:,0],bube_corners[4,0].reshape(1))),torch.cat((bube_corners[4:,1],bube_corners[4,1].reshape(1))),color='r')
for i in range(4):
    ax.plot(torch.cat((bube_corners[i,0].reshape(1),bube_corners[4+i,0].reshape(1))),torch.cat((bube_corners[i,1].reshape(1),bube_corners[4+i,1].reshape(1))),color='r')
ax.scatter(0,0,color='b')
for i in range(8):
    ax.text(bube_corners[i,0], bube_corners[i,1], '(%d)' % i, ha='right')
ax.plot(torch.cat((box.get_all_corners()[:,0],box.get_all_corners()[0,0].reshape(1))),torch.cat((box.get_all_corners()[:,1],box.get_all_corners()[0,1].reshape(1))),color='b')
plt.savefig(os.path.join('/work3/s194369/3dod/ProposalNetwork/output/trash', 'test.png'),dpi=300, bbox_inches='tight')
"""




# Get image and scale intrinsics
with open('ProposalNetwork/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)


prop_box = Box(gt_instances[0].gt_boxes[0].tensor.squeeze()*1.1,format='x1, y1, x2, y2')
IoA = intersection_over_proposal_area(gt_instances[0].gt_boxes[0], prop_box)
print(IoA)
IoA = custom_mapping(IoA)
print(IoA)
exit()


input_format = 'BGR'
img = batched_inputs[0]['image']
img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
input = batched_inputs[0]

K = torch.tensor(input['K'])
scale = input['height']/img.shape[0]
K_scaled = torch.tensor(
    [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
    dtype=torch.float32) @ K

# Get 2 proposal boxes
box_size = min(len(proposals[0].proposal_boxes), 1)
v_pred = Visualizer(img, None)
v_pred = v_pred.overlay_instances(
    boxes=proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
)
box = torch.tensor(proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()[0])
box_width = box[2]-box[0]
box_height = box[3]-box[1]
box_center_x = box[0]+box_width/2
box_center_y = box[1]+box_height/2
reference_box = Box(torch.tensor([box_center_x,box_center_y, box_width,box_height]))

# Get depth info
depth_model = 'zoedepth'
pretrained_resource = 'local::depth/checkpoints/depth_anything_metric_depth_indoor.pt'
model = setup_depth_model(depth_model, pretrained_resource)
depth_image = depth_of_images(img, model)
#depth_image = torch.ones((img.shape[0],img.shape[1]))*3 # faster for checking

# Get Proposals
number_of_proposals = 1
pred_cubes = propose(reference_box, depth_image, K_scaled, img.shape[:2],number_of_proposals=number_of_proposals)
pred_meshes = []
for i in range(number_of_proposals):
    cube = pred_cubes[i].get_cube()
    pred_meshes.append(cube.__getitem__(0).detach())

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
prop_img = v_pred.get_image()
img_3DPR = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
vis_img_3d = img_3DPR.astype(np.uint8)
ax.imshow(vis_img_3d); ax.axis('off')
#ax.plot(torch.cat((pred_box.get_all_corners()[:,0],pred_box.get_all_corners()[0,0].reshape(1))),torch.cat((pred_box.get_all_corners()[:,1],pred_box.get_all_corners()[0,1].reshape(1))),color='b')
ax.plot(torch.cat((reference_box.get_all_corners()[:,0],reference_box.get_all_corners()[0,0].reshape(1))),torch.cat((reference_box.get_all_corners()[:,1],reference_box.get_all_corners()[0,1].reshape(1))),color='purple')
plt.savefig(os.path.join('ProposalNetwork/output/trash', 'test_real.png'),dpi=300, bbox_inches='tight')
