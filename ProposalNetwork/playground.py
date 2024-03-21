from ProposalNetwork.proposals.proposals import propose_random, propose

from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.conversions import cube_to_box, pixel_to_normalised_space
from ProposalNetwork.utils.utils import compute_rotation_matrix_from_ortho6d, make_cube, iou_2d, iou_3d, custom_mapping

from ProposalNetwork.scoring.scorefunction import score_segmentation, score_dimensions, score_iou

from ProposalNetwork.segment import show_mask

import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np

from cubercnn import util, vis
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer

from segment_anything import sam_model_registry, SamPredictor

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

predictor = init_segmentation()

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
gt_z = gt_cube_.center[2]


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
pred_cubes = propose(reference_box, depth_image, K_scaled, img.shape[:2],number_of_proposals=number_of_proposals)
proposed_box = [cube_to_box(pred_cubes[i],K_scaled) for i in range(number_of_proposals)]

# OB IoU3D
IoU3D = iou_3d(gt_cube_,pred_cubes)
max_values3D = [np.max(IoU3D[:n]) for n in x_points]
idx_scores3D = [np.argmax(IoU3D[:n]) for n in x_points]
max_scores3D = [IoU3D[i] for i in idx_scores3D]
idx_highest_iou3D = idx_scores3D[-1]
print('highest possible IoU', np.max(IoU3D))


# OB IoU2D
IoU2D = score_iou(gt_box, proposed_box)
idx_scores_iou2d = np.argsort(IoU2D)
sorted_iou2d_IoU = [IoU3D[i] for i in idx_scores_iou2d]
iou2d_ious = [np.max(sorted_iou2d_IoU[:n]) for n in x_points]
print('IoU2D score of best 3dIoU',IoU2D[idx_scores_iou2d[0]])

# Plotting
plt.figure()
plt.plot(x_points, iou2d_ious, marker='o', linestyle='-',c='orange') 
plt.grid(True)
plt.xscale('log')
plt.xlabel('Number of Proposals')
plt.ylabel('3D IoU')
plt.title('IoU vs Number of Proposals')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'BO_iou2d.png'),dpi=300, bbox_inches='tight')



# Segment Score
if os.path.exists('/work3/s194369/3dod/ProposalNetwork/mask.pkl'):
      # load
     with open('/work3/s194369/3dod/ProposalNetwork/mask.pkl', 'rb') as f:
        masks = pickle.load(f)
else:
    predictor.set_image(img)
    input_box = np.array([reference_box.x1,reference_box.y1,reference_box.x2,reference_box.y2])#np.array(reference_box.get_all_corners()).flatten()

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    # dump
    with open('/work3/s194369/3dod/ProposalNetwork/mask.pkl', 'wb') as f:
        pickle.dump(masks, f)

seg_mask = masks[0]
segment_scores = [score_segmentation(pred_cubes[i].get_bube_corners(K_scaled),seg_mask) for i in range(number_of_proposals)]
idx_scores_segment = np.argsort(segment_scores)
sorted_segment_IoU = [IoU3D[i] for i in idx_scores_segment]
segment_ious = [np.max(sorted_segment_IoU[:n]) for n in x_points]
print('Segment score of best 3dIoU',segment_scores[idx_scores_segment[0]])

# Plotting
plt.figure()
plt.plot(x_points, segment_ious, marker='o', linestyle='-',c='purple') 
plt.grid(True)
plt.xscale('log')
plt.xlabel('Number of Proposals')
plt.ylabel('3D IoU')
plt.title('IoU vs Number of Proposals')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'BO_segment.png'),dpi=300, bbox_inches='tight')




# OB Dimensions
dimensions = [np.array(pred_cubes[i].dimensions) for i in range(len(pred_cubes))]
dim_scores = score_dimensions(gt_instances[0].gt_classes[0], dimensions)
idx_scores_iou2d = np.argsort(IoU2D)
sorted_iou2d_IoU = [IoU3D[i] for i in idx_scores_iou2d]
iou2d_ious = [np.max(sorted_iou2d_IoU[:n]) for n in x_points]
print('IoU2D score of best 3dIoU',IoU2D[idx_scores_iou2d[0]])

# Plotting
plt.figure()
plt.plot(x_points, iou2d_ious, marker='o', linestyle='-',c='green') 
plt.grid(True)
plt.xscale('log')
plt.xlabel('Number of Proposals')
plt.ylabel('3D IoU')
plt.title('IoU vs Number of Proposals')
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'BO_dim.png'),dpi=300, bbox_inches='tight')











# Plot
# Get 2 proposal boxes
box_size = min(len(proposals[0].proposal_boxes), 1)
v_pred = Visualizer(img, None)
v_pred = v_pred.overlay_instances(
    boxes=proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
)


#pred_meshes = []
#for i in idx_scores[1:]:
#    cube = pred_cubes[i].get_cube()
#    pred_meshes.append(cube.__getitem__(0).detach())
# Take box with highest iou
pred_meshes = [pred_cubes[idx_highest_iou3D].get_cube().__getitem__(0).detach()]

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
show_mask(masks,ax)
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'box_with_highest_iou.png'),dpi=300, bbox_inches='tight')

# convert from BGR to RGB
im_concat = im_concat[..., ::-1]
util.imwrite(im_concat, os.path.join('ProposalNetwork/output/AMOB', 'vis_result.jpg'))


# Take box with highest iou
pred_meshes = [pred_cubes[idx_highest_iou3D].get_cube().__getitem__(0).detach()]

# Add 3D GT
meshes_text = ['highest segment']
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
show_mask(masks,ax)
plt.savefig(os.path.join('ProposalNetwork/output/AMOB', 'box_with_highest_segment.png'),dpi=300, bbox_inches='tight')