import torch
import numpy as np
import cv2
import pickle
from scipy.stats import pearsonr 

from ProposalNetwork.utils.utils import iou_2d, mask_iou, euler_to_unit_vector

def score_iou(gt_box, proposal_box):
    IoU = iou_2d(gt_box,proposal_box)
    return IoU

def score_segmentation(segmentation_mask, bube_corners):
    '''
    segmentation_mask   : Mask
    bube_corners        : List of Lists
    '''

    scores = []
    for i in range(len(bube_corners)):
        bube_mask = np.zeros(segmentation_mask.shape, dtype='uint8')

        # Remove "inner" points (2) and put others in correct order 
        # Calculate the convex hull of the points which also orders points correctly
        polygon_points = cv2.convexHull(np.array(bube_corners[i]))
        polygon_points = np.array([polygon_points],dtype=np.int32)
        cv2.fillPoly(bube_mask, polygon_points, 1)
        scores.append(mask_iou(segmentation_mask, bube_mask))

    return scores

def score_dimensions(category, dimensions):
    '''
    category   : List
    dimensions : List of Lists
    P(dim|priors)
    '''
    with open('filetransfer/priors.pkl', 'rb') as f:
        priors, Metadatacatalog = pickle.load(f)

    score = []
    for i in range(len(dimensions)):
        #category_name = Metadatacatalog.thing_classes[category] # for printing and checking that correct
        [prior_mean, prior_std] = priors['priors_dims_per_cat'][category]
        dimension = dimensions[i]
        dimensions_scores = np.exp(-1/2 * ((dimension - prior_mean)/prior_std)**2)
        score.append(np.mean(dimensions_scores))

    return score

def score_angles(gt_angles, pred_angles):
    gt_nv = euler_to_unit_vector(gt_angles)
    correlation = []
    for i in range(len(pred_angles)):
        pred_nv = euler_to_unit_vector(pred_angles[i])
        correlation.append(abs(pearsonr(gt_nv,pred_nv)[0]))

    return correlation

def score_function(gt_box, proposal_box, bube_corners, segmentation_mask, category, dimensions):
    score = 1.0
    score *= score_iou(gt_box, proposal_box)
    score *= score_segmentation(bube_corners, segmentation_mask)
    score *= score_dimensions(category, dimensions)

    return score