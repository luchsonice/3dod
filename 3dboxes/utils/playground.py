from spaces import Box, Bube, Cube
from conversions import bube_to_box, cube_to_bube

import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np

from cubercnn import util, vis
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer



R = torch.eye(3)
cube = Cube(torch.tensor([5,5,10,2,2,4]),R)
print('cube',cube.get_all_corners())

K = torch.eye(3)
bube = cube_to_bube(cube,K)
bube_corners = bube.get_all_corners()
print('bube',bube_corners)

box = bube_to_box(bube)
print('box',box.get_all_corners())

with open('3dboxes/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)


def make_random_boxes(n_boxes=10):
    # rotation_matrix = torch.rand(3,3)*2*torch.pi

    rotation_matrix = torch.eye(3) # no rotation
    
    # need xyz, whl, and pose (R)
    # whl = torch.rand(3)*0.5
    whl = torch.tensor([0.3, 0.3, 0.3])
    xyz = torch.tensor([-0.1, 0, 1.7])
    # xyz = torch.rand(3)*1
    return xyz, whl, rotation_matrix

#####################
n_boxes = 1
pred_xyz, pred_whl, pred_pose = make_random_boxes(n_boxes=n_boxes)
pred_xyzwhl = torch.cat((pred_xyz, pred_whl), dim=0)

cube = Cube(torch.tensor([5,5,10,2,2,4]),pred_pose)

pred_colors = torch.tensor([util.get_color(i) for i in range(n_boxes)])/255.0

pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

input_format = 'BGR'
img = batched_inputs[0]['image']
img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
input = batched_inputs[0]
K = torch.tensor(input['K'])
scale = input['height']/img.shape[0]

K_scaled = torch.tensor(
    [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
    dtype=torch.float32) @ K
# convert to lists
pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]

# horizontal stack 3D GT and pred left/right


# 2 box
box_size = min(len(proposals[0].proposal_boxes), 2)
v_pred = Visualizer(img, None)
v_pred = v_pred.overlay_instances(
    boxes=proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
)
prop_img = v_pred.get_image()
img_3DPR = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes, text=['3d box'], mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
vis_img_3d = img_3DPR.astype(np.uint8)







# Plot bube on 2D plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(vis_img_3d); ax.axis('off')
ax.plot(torch.cat((bube_corners[:4,0],bube_corners[0,0].reshape(1))),torch.cat((bube_corners[:4,1],bube_corners[0,1].reshape(1))),color='r',linewidth=3)
ax.plot(torch.cat((bube_corners[4:,0],bube_corners[4,0].reshape(1))),torch.cat((bube_corners[4:,1],bube_corners[4,1].reshape(1))),color='r')
for i in range(4):
    ax.plot(torch.cat((bube_corners[i,0].reshape(1),bube_corners[4+i,0].reshape(1))),torch.cat((bube_corners[i,1].reshape(1),bube_corners[4+i,1].reshape(1))),color='r')
ax.scatter(0,0,color='b')
for i in range(8):
    ax.text(bube_corners[i,0], bube_corners[i,1], '(%d)' % i, ha='right')
ax.plot(torch.cat((box.get_all_corners()[:,0],box.get_all_corners()[0,0].reshape(1))),torch.cat((box.get_all_corners()[:,1],box.get_all_corners()[0,1].reshape(1))),color='b')
plt.savefig(os.path.join('/work3/s194369/3dod/3dboxes/output/trash', 'test_real.png'),dpi=300, bbox_inches='tight')
