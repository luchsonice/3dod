from ProposalNetwork.proposals.proposals import propose_random, propose

from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space, normalised_space_to_pixel
from ProposalNetwork.utils.utils import compute_rotation_matrix_from_ortho6d, make_cube, iou_2d, iou_3d, normalize_vector

from ProposalNetwork.scoring.scorefunction import score_segmentation, score_dimensions, score_iou, score_angles

from ProposalNetwork.utils.utils import show_mask

import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np

from cubercnn import util, vis
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer

from segment_anything import sam_model_registry, SamPredictor

from math import atan2, cos, sin, sqrt, pi
from skimage.transform import resize
import cv2
from sklearn.decomposition import PCA

def init_segmentation():
    # 1) first cd into the segment_anything and pip install -e .
    # to get the model stary in the root foler folder and run the download_model.sh 
    # 2) chmod +x download_model.sh && ./download_model.sh
    # the largest model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # this is the smallest model
    sam_checkpoint = "segment-anything/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]

#torch.manual_seed(1)

# Get image and scale intrinsics
with open('ProposalNetwork/proposals/network_out2.pkl', 'rb') as f:
        batched_inputs, images, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)

image = 1
gt_obj = 1

# Necessary Ground Truths
# 2D
gt_box = Box(gt_instances[image].gt_boxes[gt_obj].tensor.squeeze())
# 3D
gt____whlxyz = gt_instances[image].gt_boxes3D[gt_obj]
gt_R = gt_instances[image].gt_poses[gt_obj]
gt_cube_ = Cube(torch.cat([gt____whlxyz[6:],gt____whlxyz[3:6]]),gt_R)
gt_cube = gt_cube_.get_cube()
gt_z = gt_cube_.center[2]
#print('GT',gt____whlxyz,util.mat2euler(gt_R))
#print(gt_R - util.euler2mat(util.mat2euler(gt_R)))

# image
input_format = 'BGR'
img = batched_inputs[image]['image']

img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
input = batched_inputs[image]

K = torch.tensor(input['K'])
scale = input['height']/img.shape[0]
K_scaled = torch.tensor(
    [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
    dtype=torch.float32) @ K
reference_box = Box(proposals[image].proposal_boxes[0].tensor[0])

# Get depth info
depth_image = np.load(f"datasets/depth_maps/{batched_inputs[image]['image_id']}.npz")['depth']
depth_image = resize(depth_image,(img.shape[0],img.shape[1]))
depth_patch = depth_image[int(reference_box.x1):int(reference_box.x2),int(reference_box.y1):int(reference_box.y2)]

####################################################################################################################################################################################################################################################################################

# Get Proposals
x_points = [1]#, 10, 100]#, 1000, 10000]#, 100000]
number_of_proposals = x_points[-1]

with open('filetransfer/priors.pkl', 'rb') as f:
        priors, Metadatacatalog = pickle.load(f)
category = gt_instances[image].gt_classes[gt_obj]
priors_propose = priors['priors_dims_per_cat'][category]
pred_cubes = propose(reference_box, depth_patch, priors_propose, img.shape[:2][::-1], number_of_proposals=number_of_proposals, gt_cube=gt_cube_)
proposed_box = [cube_to_box(pred_cubes[i],K_scaled) for i in range(number_of_proposals)]

# OB IoU3D
IoU3D = np.array(iou_3d(gt_cube_,pred_cubes))
print('Percentage of cubes with no intersection:',int(np.count_nonzero(IoU3D == 0.0)/IoU3D.size*100))
idx_scores_iou3d = np.argsort(IoU3D)[::-1]
sorted_iou3d_IoU = [IoU3D[i] for i in idx_scores_iou3d]
print('Highest possible IoU3D score',sorted_iou3d_IoU[0])

# OB IoU2D
IoU2D = score_iou(gt_box, proposed_box)
idx_scores_iou2d = np.argsort(IoU2D)[::-1]
sorted_iou2d_IoU = [IoU3D[i] for i in idx_scores_iou2d]
iou2d_ious = [np.max(sorted_iou2d_IoU[:n]) for n in x_points]
print('IoU3D of best IoU2D score',sorted_iou2d_IoU[0])


# Segment Score
if os.path.exists('ProposalNetwork/mask'+str(image)+'.pkl'):
      # load
     with open('ProposalNetwork/mask'+str(image)+'.pkl', 'rb') as f:
        masks = pickle.load(f)
else:
    predictor = init_segmentation()
    predictor.set_image(img)
    input_box = np.array([reference_box.x1,reference_box.y1,reference_box.x2,reference_box.y2])

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    # dump
    with open('ProposalNetwork/mask'+str(image)+'.pkl', 'wb') as f:
        pickle.dump(masks, f)

seg_mask = masks[0]
bube_corners = [pred_cubes[i].get_bube_corners(K_scaled) for i in range(number_of_proposals)]
segment_scores = score_segmentation(seg_mask, bube_corners)
idx_scores_segment = np.argsort(segment_scores)[::-1]
sorted_segment_IoU = [IoU3D[i] for i in idx_scores_segment]
segment_ious = [np.max(sorted_segment_IoU[:n]) for n in x_points]
print('IoU3D of best segment score',sorted_segment_IoU[0])

# OB Dimensions
dimensions = [np.array(pred_cubes[i].dimensions) for i in range(len(pred_cubes))]
dim_scores = score_dimensions(priors_propose, dimensions)
idx_scores_dim = np.argsort(dim_scores)[::-1]
sorted_dim_IoU = [IoU3D[i] for i in idx_scores_dim]
dim_ious = [np.max(sorted_dim_IoU[:n]) for n in x_points]
print('IoU3D of best dim score',sorted_dim_IoU[0])

# Angles
angles = [np.array(util.mat2euler(pred_cubes[i].rotation)) for i in range(len(pred_cubes))]
angle_scores = score_angles(util.mat2euler(gt_R),angles)
idx_scores_angles = np.argsort(angle_scores)[::-1]
sorted_angles_IoU = [IoU3D[i] for i in idx_scores_angles]
angle_ious = [np.max(sorted_angles_IoU[:n]) for n in x_points]
print('IoU3D of best angle score',sorted_angles_IoU[0])

# 2D Contour
seg_mask_uint8 = np.array(seg_mask).astype(np.uint8) * 255
ret, thresh = cv2.threshold(seg_mask_uint8, 0.5, 1, 0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_x = []
contour_y = []
for i in range(len(contours)):
     for j in range(len(contours[i])):
          contour_x.append(contours[i][j][0][0])
          contour_y.append(contours[i][j][0][1])

# 3rd dimension
contour_z = np.zeros(len(contour_x))
for i in range(len(contour_x)):
     contour_z[i] = depth_image[contour_x[i],contour_y[i]]

min_val = np.min(contour_x)
max_val = np.max(contour_x)
scaled_contour_x = (contour_x - min_val) / (max_val - min_val)

min_val = np.min(contour_y)
max_val = np.max(contour_y)
scaled_contour_y = (contour_y - min_val) / (max_val - min_val)

min_val = np.min(contour_z)
max_val = np.max(contour_z)
scaled_contour_z = (contour_z - min_val) / (max_val - min_val)

contours3D = np.array([scaled_contour_x, scaled_contour_y, scaled_contour_z]).T

# PCA
pca = PCA(n_components=3)
pca.fit(contours3D)
orientations = pca.components_

def gram_schmidt(n):
    # Choose an arbitrary vector
    v1 = np.array([1.0, 0.0, 0.0])  # Choose a simple starting vector
    
    # Normalize the first vector
    v1 /= np.linalg.norm(v1)
    
    # Calculate the second vector using Gram-Schmidt process
    v2 = n - np.dot(n, v1) * v1
    v2 /= np.linalg.norm(v2)
    
    # Calculate the third vector as the cross product of v1 and v2
    v3 = np.cross(v1, v2)
    
    return v1, v2, v3

basis = gram_schmidt(orientations)
euler_angles = np.arctan2(basis[2, 1], basis[2, 2]), np.arcsin(-basis[2, 0]), np.arctan2(basis[1, 0], basis[0, 0])
print(basis.T)
print('found angles',np.array(euler_angles) % (pi / 2))
print('gt angles',util.mat2euler(gt_R) % (pi / 2))

def vectors_from_rotation_matrix(rotation_matrix):
    # Extract vectors from rotation matrix
    v1 = rotation_matrix[:, 0]
    v2 = rotation_matrix[:, 1]
    v3 = rotation_matrix[:, 2]

    return np.array([v1, v2, v3])

#orientations = vectors_from_rotation_matrix(np.array(gt_R)) #gt rotation














points_2d_homogeneous = np.dot(K_scaled, orientations.T).T

# Convert homogeneous coordinates to Cartesian coordinates
points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]


# Plotting
plt.figure()
plt.plot(x_points, dim_ious, marker='o', linestyle='-',c='green',label='dim') 
plt.plot(x_points, segment_ious, marker='o', linestyle='-',c='purple',label='segment')
plt.plot(x_points, iou2d_ious, marker='o', linestyle='-',c='orange',label='2d IoU') 
plt.plot(x_points, angle_ious, marker='o', linestyle='-',c='darkslategrey',label='angles') 
plt.grid(True)
plt.xscale('log')
plt.xlabel('Number of Proposals')
plt.ylabel('3D IoU')
plt.title('IoU vs Number of Proposals')
plt.legend()
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'BO.png'),dpi=300, bbox_inches='tight')

combined_score = np.array(segment_scores)*np.array(IoU2D)*np.array(dim_scores)*np.array(angle_scores)
plt.figure()
plt.hexbin(combined_score, IoU3D, gridsize=10)
plt.axis([combined_score.min(), combined_score.max(), IoU3D.min(), IoU3D.max()])
plt.xlabel('score')
plt.ylabel('3DIoU')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'combined_scores.png'),dpi=300, bbox_inches='tight')

""" Makes only sense when better results
fig, ax = plt.subplots()
ax.scatter(combined_score,IoU3D, alpha=0.3)
heatmap, xedges, yedges = np.histogram2d(combined_score,IoU3D, bins=10)
extent = [xedges[0], xedges[-1]+0.05, yedges[0], yedges[-1]+0.05]
cax = ax.imshow(heatmap.T, extent=extent, origin='lower')
cbar = fig.colorbar(cax)
fig.savefig(os.path.join('ProposalNetwork/output/AMOB', 'combined_scores.png'),dpi=300, bbox_inches='tight')
"""
####################################################################################################################################################################################################################################################################################


# Plot
# Get 2 proposal boxes
box_size = min(len(proposals[image].proposal_boxes), 1)
v_pred = Visualizer(img, None)
v_pred = v_pred.overlay_instances(
    boxes=proposals[image].proposal_boxes[0:box_size].tensor.cpu().numpy()
)

# Take box with highest iou
pred_meshes = [pred_cubes[idx_scores_iou3d[0]].get_cube().__getitem__(0).detach()]
#print(pred_cubes[idx_scores_iou3d[0]].__repr__)
# Add 3D GT
meshes_text = ['proposal cube' for _ in range(len(pred_meshes))]
meshes_text.append('gt cube')
pred_meshes.append(gt_cube.__getitem__(0).detach())

fig = plt.figure()
ax = fig.add_subplot(111)
prop_img = v_pred.get_image()
img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes,text=meshes_text, blend_weight=0.5, blend_weight_overlay=0.85,scale = img.shape[0])
im_concat = np.concatenate((img_3DPR, img_novel), axis=1)
vis_img_3d = img_3DPR.astype(np.uint8)
ax.imshow(vis_img_3d)
ax.plot(torch.cat((gt_box.get_all_corners()[:,0],gt_box.get_all_corners()[0,0].reshape(1))),torch.cat((gt_box.get_all_corners()[:,1],gt_box.get_all_corners()[0,1].reshape(1))),color='purple')
ax.scatter(gt____whlxyz[0],gt____whlxyz[1],color='r')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'box_with_highest_iou.png'),dpi=300, bbox_inches='tight')

distances = np.linalg.norm(points_2d, axis=1)

# Normalize points by dividing each coordinate by its distance from the origin
points_2d = points_2d / np.max(distances)
#points_2d = points_2d / distances[:, np.newaxis]

# Contour Plot
cntr = np.array(gt____whlxyz[:2])
p1 = (cntr[0] + points_2d[0][0], cntr[1] + points_2d[0][1])
p2 = (cntr[0] + points_2d[1][0], cntr[1] + points_2d[1][1])
p3 = (cntr[0] + points_2d[2][0], cntr[1] + points_2d[2][1])

fig = plt.figure()
ax = fig.add_subplot(111)
drawAxis(prop_img, cntr, p1, (255, 255, 0), 150)
drawAxis(prop_img, cntr, p2, (0, 0, 255), 150)
drawAxis(prop_img, cntr, p3, (0, 255, 255), 150)
ax.imshow(prop_img)
show_mask(seg_mask,ax)
#ax.scatter(contour_x, contour_y, c='r', s=1)
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'contour.png'),dpi=300, bbox_inches='tight')
####################################################################################################################################################################################################################################################################################


# convert from BGR to RGB
im_concat = im_concat[..., ::-1]
util.imwrite(im_concat, os.path.join('ProposalNetwork/output/AMOB', 'vis_result.jpg'))


# Take box with highest segment
pred_meshes = [pred_cubes[idx_scores_segment[0]].get_cube().__getitem__(0).detach()]

# Add 3D GT
meshes_text = ['highest segment']
meshes_text.append('gt cube')
pred_meshes.append(gt_cube.__getitem__(0).detach())

img_3DPR, _, _ = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes,text=meshes_text, blend_weight=0.5, blend_weight_overlay=0.85,scale = img.shape[0])
vis_img_3d = img_3DPR.astype(np.uint8)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(vis_img_3d)
ax.plot(torch.cat((gt_box.get_all_corners()[:,0],gt_box.get_all_corners()[0,0].reshape(1))),torch.cat((gt_box.get_all_corners()[:,1],gt_box.get_all_corners()[0,1].reshape(1))),color='purple')
show_mask(masks,ax)
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'box_with_highest_segment.png'),dpi=300, bbox_inches='tight')



# tmp
for i in range(len(IoU3D)):
     if IoU3D[i] == 0.0:
          idx = i
          break
     else:
          idx = -1

pred_meshes = [pred_cubes[idx].get_cube().__getitem__(0).detach()]
meshes_text = ['box with 0 3diou']
meshes_text.append('gt cube')
pred_meshes.append(gt_cube.__getitem__(0).detach())

fig = plt.figure()
ax = fig.add_subplot(111)
prop_img = v_pred.get_image()
img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes,text=meshes_text, blend_weight=0.5, blend_weight_overlay=0.85,scale = img.shape[0])
im_concat = np.concatenate((img_3DPR, img_novel), axis=1)
im_concat = im_concat[..., ::-1]
util.imwrite(im_concat, os.path.join('ProposalNetwork/output/AMOB', 'tmp.jpg'))

center = normalised_space_to_pixel(np.array(pred_cubes[idx].center)[:2],img.shape[:2][::-1])
fig = plt.figure()
ax = fig.add_subplot(111)
vis_img_3d = img_3DPR.astype(np.uint8)
ax.imshow(vis_img_3d)
ax.scatter([135.45,135.45,259.76,259.76],[121.6,236.29,121.6,236.29],color='b')
ax.scatter(center[0],center[1],color='r')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'tmp2.png'),dpi=300, bbox_inches='tight')