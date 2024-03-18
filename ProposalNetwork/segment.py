import pickle
from detectron2.data.detection_utils import convert_image_to_rgb
from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

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

# helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# load image

with open('ProposalNetwork/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)

pred_box = proposals[0].proposal_boxes[0].tensor[0].numpy()
# centre point of the box
input_point = np.array([[pred_box[0]+(pred_box[2]-pred_box[0])/2, pred_box[1]+(pred_box[3]-pred_box[1])/2]])

img = batched_inputs[0]['image']
image = convert_image_to_rgb(img.permute(1, 2, 0), 'BGR')    

plt.figure()
plt.imshow(image)
plt.axis('on')
plt.show()

predictor.set_image(image)

# set a point
# input_point = np.array([[500, 375]])
input_label = np.array([1])
plt.figure()
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# With multimask_output=True (the default setting), SAM outputs 3 masks, where scores gives the model's own estimation of the quality of these masks. This setting is intended for ambiguous input prompts, and helps the model disambiguate different objects consistent with the prompt. When False, it will return a single mask. For ambiguous prompts such as a single point, it is recommended to use multimask_output=True even if only a single mask is desired; the best single mask can be chosen by picking the one with the highest score returned in scores. This will often result in a better mask.

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure()
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

# Specifying a specific object with a box
# The model can also take a box as input, provided in xyxy format.

input_box = pred_box # np.array([425, 600, 700, 875])
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
plt.figure()
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()