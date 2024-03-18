import torch
from ProposalNetwork.utils.utils import iou_2d, iou_3d, custom_mapping

def score_iou(gt_box, proposal_box):
    IoU = iou_2d(gt_box,proposal_box)
    #IoU = custom_mapping(IoU)
    return IoU

def score_segmentation_ioa(bube, segmentation):
    '''
    IoA between segmentation and bube.
    '''
    
    return 1

def score_function(weights, gt_box, proposal_box):
    score = 1.0
    score *= score_iou(gt_box, proposal_box)

    return score