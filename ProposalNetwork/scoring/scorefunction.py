import torch
import numpy as np
import cv2

from ProposalNetwork.utils.utils import iou_2d, iou_3d, custom_mapping, mask_iou

def score_iou(gt_box, proposal_box):
    IoU = iou_2d(gt_box,proposal_box)
    #IoU = custom_mapping(IoU)
    return IoU

def score_segmentation(bube_corners, segmentation_mask):
    '''
    IoA between segmentation and bube.
    '''
    bube_mask = np.zeros(segmentation_mask.shape, dtype='uint8')

    # Remove "inner" points (2) and put others in correct order 
    # Calculate the convex hull of the points which also orders points correctly
    polygon_points = cv2.convexHull(np.array(bube_corners))
    polygon_points = np.array([polygon_points],dtype=np.int32)
    cv2.fillPoly(bube_mask, polygon_points, 1)

    return mask_iou(segmentation_mask, bube_mask)

def score_function(weights, gt_box, proposal_box):
    score = 1.0
    score *= score_iou(gt_box, proposal_box)

    return score