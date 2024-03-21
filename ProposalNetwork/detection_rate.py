import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from rich.progress import track
from tqdm import tqdm

from detectron2.data.detection_utils import convert_image_to_rgb
from ProposalNetwork.proposals.proposals import propose, propose_random
from ProposalNetwork.utils.conversions import cube_to_box
from ProposalNetwork.utils.spaces import Box, Cube
from ProposalNetwork.utils.utils import iou_3d

IoU_threshold = 0.3
#%% load
# Get image and scale intrinsics
with open('ProposalNetwork/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)

num_x_points = 20
detection_rates = []
x_points = np.logspace(0,3, num=num_x_points, dtype=int) # 10^0-10^3, 1-1000 # [1, 10, 100, 1000]#, 10000, 100000]
number_of_proposals = x_points[-1]

# Necessary Ground Truths
for batched_input, gt_instance, proposal in track(zip(batched_inputs, gt_instances, proposals), total=len(batched_inputs), description=None):
    # image
    input_format = 'BGR'
    img = batched_input['image']
    img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
    input = batched_input

    K = torch.tensor(input['K'])
    scale = input['height']/img.shape[0]
    K_scaled = torch.tensor(
        [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
        dtype=torch.float32) @ K

    # Get depth info
    depth_image = np.load(f"datasets/depth_maps/{batched_input['image_id']}.npz")['depth']
    depth_image = resize(depth_image,(img.shape[0],img.shape[1]))

    for gt_box_2d, gt_box3D, gt_pose, proposal_box in zip(gt_instance.gt_boxes, gt_instance.gt_boxes3D, gt_instance.gt_poses, proposal.proposal_boxes):
        # 2D
        gt_box = Box(gt_box_2d)
        # 3D
        gt____whlxyz = gt_box3D
        gt_R = gt_pose
        gt_cube_ = Cube(torch.cat([gt____whlxyz[6:],gt____whlxyz[3:6]]),gt_R)
        gt_cube = gt_cube_.get_cube()
        gt_z = gt_cube_.center[2]

        reference_box = Box(proposal_box)

        #%% Get Proposals
        pred_cubes = propose(reference_box, depth_image, K_scaled, img.shape[:2],number_of_proposals=number_of_proposals)
        proposed_box = [cube_to_box(pred_cubes[i],K_scaled) for i in range(number_of_proposals)]

        # OB IoU3D
        IoU3D = iou_3d(gt_cube_, pred_cubes)
        # measure if there is an IoU > threshold at the current number of proposals
        detection_rate = [(np.sum(IoU3D[:num_prop] > IoU_threshold) > 0) * 1 for num_prop in x_points]
        detection_rates.append(detection_rate)

# collapse the list of detection rates
detection_rate = np.mean(detection_rates, axis=0)

#%% plot
fig, ax = plt.subplots(1, figsize=(9, 5))
ax.set_xlabel('Number of Proposals')
ax.set_ylabel('Detection rate')
ax.set_title(f'Detection rate, IoU {IoU_threshold}')
ax.plot(x_points, detection_rate, label='random proposals')
ax.legend()
plt.savefig('ProposalNetwork/output/detection_rate.png', dpi=300, bbox_inches='tight')