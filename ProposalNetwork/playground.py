from ProposalNetwork.proposals.proposals import propose_random

from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space
from ProposalNetwork.utils.utils import compute_rotation_matrix_from_ortho6d, make_cube, iou_2d, iou_3d, custom_mapping


import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np

from cubercnn import util, vis
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer

#torch.manual_seed(1)
'''
R = compute_rotation_matrix_from_ortho6d(torch.rand(6)*np.pi)
cube = Cube(torch.tensor([-0.4,-0.5,1,1,1.5,0.5]),R)
print('cube',cube.get_all_corners())

box = cube_to_box(cube,torch.eye(3))
print('box',box.get_all_corners())

bube_corners = cube.get_bube_corners(torch.tensor([[570.3422,   0.0000, 310.0000],
                                                   [  0.0000, 570.3422, 225.0000],
                                                   [  0.0000,   0.0000,   1.0000]]))

# Plot bube on 2D plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(torch.cat((bube_corners[:4,0],bube_corners[0,0].reshape(1))),torch.cat((bube_corners[:4,1],bube_corners[0,1].reshape(1))),color='r',linewidth=3)
ax.plot(torch.cat((bube_corners[4:,0],bube_corners[4,0].reshape(1))),torch.cat((bube_corners[4:,1],bube_corners[4,1].reshape(1))),color='r')
for i in range(4):
    ax.plot(torch.cat((bube_corners[i,0].reshape(1),bube_corners[4+i,0].reshape(1))),torch.cat((bube_corners[i,1].reshape(1),bube_corners[4+i,1].reshape(1))),color='r')
ax.scatter(-1,-1,color='b')
ax.scatter(-1,1,color='b')
ax.scatter(1,-1,color='b')
ax.scatter(1,1,color='b')
ax.scatter(-0.4,-0.5,color='b')
for i in range(8):
    ax.text(bube_corners[i,0], bube_corners[i,1], '(%d)' % i, ha='right')
ax.plot(torch.cat((box.get_all_corners()[:,0],box.get_all_corners()[0,0].reshape(1))),torch.cat((box.get_all_corners()[:,1],box.get_all_corners()[0,1].reshape(1))),color='b')
plt.savefig(os.path.join('/work3/s194369/3dod/ProposalNetwork/output/trash', 'test.png'),dpi=300, bbox_inches='tight')
#exit()
'''


# Get image and scale intrinsics
with open('ProposalNetwork/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)

# Necessary Ground Truths
# 2D
gt_box = Box(gt_instances[0].gt_boxes[0].tensor.squeeze())
# 3D
gt____whlxyz = gt_instances[0].gt_boxes3D[0]
gt_R = gt_instances[0].gt_poses[0]
gt_cube_ = Cube(torch.cat([gt____whlxyz[6:],gt____whlxyz[3:6]]),gt_R)
gt_cube = gt_cube_.get_cube()

# image
input_format = 'BGR'
img = batched_inputs[0]['image']
img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
input = batched_inputs[0]

K = torch.tensor(input['K'])
scale = input['height']/img.shape[0]
K_scaled = torch.tensor(
    [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
    dtype=torch.float32) @ K

reference_box = Box(proposals[0].proposal_boxes[0].tensor[0])

# Get depth info
depth_image = np.load(f"datasets/depth_maps/{batched_inputs[0]['image_id']}.npz")['depth']
from skimage.transform import resize
depth_image = resize(depth_image,(img.shape[0],img.shape[1]))

# Get Proposals
x_points = [1, 10, 100, 1000]#, 10000, 100000]
number_of_proposals = x_points[-1]
pred_cubes = propose_random(reference_box, depth_image, K_scaled, img.shape[:2],number_of_proposals=number_of_proposals)
proposed_box = [cube_to_box(pred_cubes[i],K_scaled) for i in range(number_of_proposals)]

# OB IoU2D
IoU2D = iou_2d(gt_box, proposed_box)
max_values = [np.max(IoU2D[:n]) for n in x_points]
idx_scores = [np.argmax(IoU2D[:n]) for n in x_points]
max_scores = [custom_mapping([IoU2D[i]])[0] for i in idx_scores]
idx_highest_iou = idx_scores[-1]

# OB IoU3D
IoU3D = iou_3d(gt_cube_,pred_cubes)
max_values3D = [np.max(IoU3D[:n]) for n in x_points]
idx_scores3D = [np.argmax(IoU3D[:n]) for n in x_points]
max_scores3D = [custom_mapping([IoU3D[i]])[0] for i in idx_scores3D]
idx_highest_iou3D = idx_scores3D[-1]

# Plotting
plt.figure()
plt.plot(x_points, max_values, marker='o', linestyle='-', label='2D') 
plt.plot(x_points, max_values3D, marker='o', linestyle='-', label='3D') 
plt.grid(True)
plt.scatter(x_points,max_scores,c='b')
plt.scatter(x_points,max_scores3D,c='orange')
plt.xscale('log')
plt.xlabel('Number of Proposals')
plt.ylabel('Maximum IoU')
plt.title('Maximum IoU vs Number of Proposals')
plt.legend()
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'OB.png'),dpi=300, bbox_inches='tight')

# Plot
# Get 2 proposal boxes
box_size = min(len(proposals[0].proposal_boxes), 1)
v_pred = Visualizer(img, None)
v_pred = v_pred.overlay_instances(
    boxes=proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
)

pred_meshes = []
for i in idx_scores[1:]:
    cube = pred_cubes[i].get_cube()
    pred_meshes.append(cube.__getitem__(0).detach())
# Take box with highest iou
pred_meshes = [pred_cubes[idx_highest_iou].get_cube().__getitem__(0).detach()]

# Add 3D GT
meshes_text = ['' for _ in range(len(pred_meshes))]
meshes_text.append('gt cube')
pred_meshes.append(gt_cube.__getitem__(0).detach())

fig = plt.figure()
ax = fig.add_subplot(111)
prop_img = v_pred.get_image()
img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes,text=meshes_text, blend_weight=0.5, blend_weight_overlay=0.85,scale = img.shape[0])
im_concat = np.concatenate((img_3DPR, img_novel), axis=1)
vis_img_3d = img_3DPR.astype(np.uint8)
ax.imshow(vis_img_3d)
#ax.plot(torch.cat((pred_box.get_all_corners()[:,0],pred_box.get_all_corners()[0,0].reshape(1))),torch.cat((pred_box.get_all_corners()[:,1],pred_box.get_all_corners()[0,1].reshape(1))),color='b')
ax.plot(torch.cat((gt_box.get_all_corners()[:,0],gt_box.get_all_corners()[0,0].reshape(1))),torch.cat((gt_box.get_all_corners()[:,1],gt_box.get_all_corners()[0,1].reshape(1))),color='purple')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'box_with_highest_iou.png'),dpi=300, bbox_inches='tight')
util.imwrite(im_concat, os.path.join('ProposalNetwork/output/AMOB', 'vis_result.jpg'))