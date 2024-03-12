import torch
from ProposalNetwork.utils.utils import intersection_over_proposal_area, custom_mapping

def score_proposal_in_2dgt(gt_box, proposal_box):
    IoA = intersection_over_proposal_area(gt_box,proposal_box)
    IoA = custom_mapping(IoA)
    return IoA

def score_function(weights, gt_box, proposal_box):
    score = 1.0
    score *= score_proposal_in_2dgt(gt_box, proposal_box)

    return score