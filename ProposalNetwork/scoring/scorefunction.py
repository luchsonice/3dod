import torch
import numpy as np
import cv2
import pickle
from scipy.stats import pearsonr 
import ProposalNetwork.utils.spaces as spaces

from ProposalNetwork.utils.utils import iou_2d, mask_iou, euler_to_unit_vector

def score_point_cloud(point_cloud:torch.Tensor, cubes:list[spaces.Cube], K:torch.Tensor, segmentation_mask:torch.Tensor):
    '''
    score the cube according to the density (number of points) of the point cloud in the cube
    '''
    # must normalise the point cloud to have the same density for the entire depth

    scores = []
    for cube in cubes:
        verts = cube.get_all_corners()
        min_x, max_x = verts[:,0].min(), verts[:,0].max()
        min_y, max_y = verts[:,1].min(), verts[:,1].max()
        min_z, max_z = verts[:,2].min(), verts[:,2].max()
        point_cloud_dens = ((point_cloud[:,0] > min_x) & 
                            (point_cloud[:,0] < max_x) & 
                            (point_cloud[:,1] > min_y) & 
                            (point_cloud[:,1] < max_y) & 
                            (point_cloud[:,2] > min_z) & 
                            (point_cloud[:,2] < max_z))
        score = point_cloud_dens.sum().item()

        # method 1
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

        scores.append(score)

    return scores



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
        scores.append(mask_iou(segmentation_mask, bube_mask)) # TODO I think we should try diving by gt as its unfair in combined

    return scores

def score_dimensions(category, dimensions):
    '''
    category   : List
    dimensions : List of Lists
    P(dim|priors)
    '''
    score = []
    [prior_mean, prior_std] = category
    for i in range(len(dimensions)):
        #category_name = Metadatacatalog.thing_classes[category] # for printing and checking that correct
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


if __name__ == '__main__':
    # testing
    s = score_point_cloud(torch.tensor([[0.1,0.1,0.1],[0.2,0.2,0.2],[-3,0,0]]), [spaces.Cube(torch.tensor([0.5,0.5,0.5,1,1,1]), torch.eye(3))])
    print(s)
    assert s == 2