import torch
from ProposalNetwork.utils.utils import iou_2d, custom_mapping

def score_proposal_in_2dgt(gt_box, proposal_box):
    IoU = iou_2d(gt_box,proposal_box)
    #IoU = custom_mapping(IoU)
    return IoU

def score_function(weights, gt_box, proposal_box):
    score = 1.0
    score *= score_proposal_in_2dgt(gt_box, proposal_box)

    return score