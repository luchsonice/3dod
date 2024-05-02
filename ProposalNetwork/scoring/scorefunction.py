import torch
import numpy as np
import cv2
import pickle
from scipy.stats import pearsonr 
import ProposalNetwork.utils.spaces as spaces

from ProposalNetwork.utils.utils import iou_2d, mask_iou, euler_to_unit_vector

def score_point_cloud(point_cloud:torch.Tensor, cubes:list[spaces.Cubes], K:torch.Tensor=None, segmentation_mask:torch.Tensor=None):
    '''
    score the cube according to the density (number of points) of the point cloud in the cube
    '''
    # must normalise the point cloud to have the same density for the entire depth
    verts = cubes.get_all_corners().squeeze(0)
    min_x, _, = verts[:,0].min(1); max_x, _ = verts[:,0].max(1)
    min_y, _, = verts[:,1].min(1); max_y, _ = verts[:,1].max(1)
    min_z, _, = verts[:,2].min(1); max_z, _ = verts[:,2].max(1)
    point_cloud_dens = ((point_cloud[:,0].view(-1,1) > min_x) & 
                        (point_cloud[:,0].view(-1,1) < max_x) & 
                        (point_cloud[:,1].view(-1,1) > min_y) & 
                        (point_cloud[:,1].view(-1,1) < max_y) & 
                        (point_cloud[:,2].view(-1,1) > min_z) & 
                        (point_cloud[:,2].view(-1,1) < max_z))
    score = point_cloud_dens.sum(0)

        # method 1
        # just in case this is needed in the future, the function can be found at commit ID: 4a06501c46beda804fd3b8ddfcbb27211f89ef66
        # area = cube.get_projected_2d_area(K).item()
        # if area != 0:
        #     score /= area
        
        # method 2
        # corners = cube.get_bube_corners(K)
        # bube_mask = np.zeros(segmentation_mask.shape, dtype=np.uint8)
        # polygon_points = cv2.convexHull(corners.numpy())
        # polygon_points = np.array([polygon_points],dtype=np.int32)
        # cv2.fillPoly(bube_mask, polygon_points, 1)

        # normalisation = (bube_mask).sum()
        # if normalisation != 0:
        #     score = score/normalisation

    return score



def score_iou(gt_box, proposal_box):
    IoU = iou_2d(gt_box,proposal_box)
    return IoU

def score_segmentation(segmentation_mask, bube_corners):
    '''
    segmentation_mask   : Mask
    bube_corners        : List of Lists
    '''

    scores = []
    bube_corners = bube_corners.squeeze(0) # remove instance dim
    for i in range(len(bube_corners)):
        bube_mask = np.zeros(segmentation_mask.shape, dtype='uint8')

        # Remove "inner" points (2) and put others in correct order 
        # Calculate the convex hull of the points which also orders points correctly
        polygon_points = cv2.convexHull(np.array(bube_corners[i]))
        polygon_points = np.array([polygon_points],dtype=np.int32)
        cv2.fillPoly(bube_mask, polygon_points, 1)
        scores.append(mask_iou(segmentation_mask, bube_mask)) # TODO I think we should try diving by gt as its unfair in combined

    return scores

def score_dimensions(category, dimensions):
    '''
    category   : List
    dimensions : List of Lists
    P(dim|priors)
    '''
    # category_name = Metadatacatalog.thing_classes[category] # for printing and checking that correct
    [prior_mean, prior_std] = category
    dimensions_scores = torch.exp(-1/2 * ((dimensions - prior_mean)/prior_std)**2)
    scores = dimensions_scores.mean(1)
    return scores

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


if __name__ == '__main__':
    # testing
    s = score_point_cloud(torch.tensor([[0.1,0.1,0.1],[0.2,0.2,0.2],[-3,0,0]]), [spaces.Cube(torch.tensor([0.5,0.5,0.5,1,1,1]), torch.eye(3))])
    print(s)
    assert s == 2