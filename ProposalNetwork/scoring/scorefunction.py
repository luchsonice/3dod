import torch
import numpy as np
import cv2
from ProposalNetwork.scoring.convex_outline import tracing_outline_robust
import ProposalNetwork.utils.spaces as spaces
from scipy.spatial import cKDTree
from ProposalNetwork.utils.utils import iou_2d, mask_iou

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

def modified_chamfer_distance(set1, set2):
    tree2 = cKDTree(set2)
    # For each point in set1 (seg point), find the distance to the nearest point in set2 (bube corner)
    distances2, _ = tree2.query(set1)
    
    return np.mean(distances2)

def score_corners(segmentation_mask, bube_corners):
    mask_np = segmentation_mask.cpu().numpy().astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the minimum area rectangle around the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)

    bube_corners = bube_corners.squeeze(0) # remove instance dim
    scores = torch.zeros(len(bube_corners), device=segmentation_mask.device)
    for i in range(len(bube_corners)):
        # Chamfer distance bube corners and box
        scores[i] = modified_chamfer_distance(box, bube_corners[i].cpu().numpy())
    
    max_score = torch.max(scores)
    
    return 1 - scores / max_score


def score_segmentation(segmentation_mask, bube_corners):
    '''
    segmentation_mask   : Mask
    bube_corners        : List of Lists
    '''
    bube_corners = bube_corners.to(device=segmentation_mask.device)
    bube_corners = bube_corners.squeeze(0) # remove instance dim
    scores = torch.zeros(len(bube_corners), device=segmentation_mask.device)
    for i in range(len(bube_corners)):
        bube_mask = np.zeros(segmentation_mask.shape, dtype='uint8')

        # Remove "inner" points (2) and put others in correct order 
        # Calculate the convex hull of the points which also orders points correctly
        polygon_points = cv2.convexHull(np.array(bube_corners[i]))
        polygon_points = np.array([polygon_points],dtype=np.int32)
        cv2.fillPoly(bube_mask, polygon_points, 1)
        scores[i] = mask_iou(segmentation_mask[::4,::4], bube_mask[::4,::4]) # TODO I think we should try diving by gt as its unfair in combined

    return scores

def score_segmentation_v2(segmentation_mask, pred_cubes, K):

    scores = []
    for i in range(len(pred_cubes.tensor.squeeze())):
        v_2d = pred_cubes[:, i].get_bube_corners(K).squeeze()
        _, f = pred_cubes[:, i].get_cuboids_verts_faces()
        f = f.squeeze()
        points, ids = tracing_outline_robust(v_2d.numpy(), f.numpy()) # not doing any projection,just simply take the verts's x and y .

        bube_mask = np.zeros(segmentation_mask.shape, dtype='uint8')
        # append first point to close the loop
        # points = np.append(points, [points[0]], axis=0)
        cv2.fillPoly(bube_mask, np.expand_dims(points,0).astype(int), 1)
        scores.append(mask_iou(segmentation_mask, bube_mask))
    return scores

def score_dimensions(category, dimensions, gt_boxes, pred_boxes):
    '''
    category   : List
    dimensions : List of Lists
    P(dim|priors)
    '''
    # category_name = Metadatacatalog.thing_classes[category] # for printing and checking that correct
    [prior_mean, prior_std] = category
    dimensions_scores = torch.exp(-1/2 * ((dimensions - prior_mean)/prior_std)**2)
    scores = dimensions_scores.mean(1)

    gt_ratio = (gt_boxes.tensor[0,2]-gt_boxes.tensor[0,0])/(gt_boxes.tensor[0,3]-gt_boxes.tensor[0,1])
    pred_ratios = (pred_boxes.tensor[:,2]-pred_boxes.tensor[:,0])/(pred_boxes.tensor[:,3]-pred_boxes.tensor[:,1])
    differences = torch.abs(gt_ratio-pred_ratios)
    max_difference = torch.max(differences)
    
    return (1 - differences / max_difference) * scores



def score_ratios(gt_box,pred_boxes):
    gt_points = gt_box.tensor[0]
    differences = torch.abs(pred_boxes.tensor - gt_points).sum(axis=1)
    max_difference = torch.max(differences)
    
    return 1 - differences / max_difference

    # 3D Dim Ratio
    gt_ratio = gt_dim[0]/gt_dim[1]
    pred_ratios = pred_dims[:,0]/pred_dims[:,1]
    differences = torch.abs(pred_ratios-gt_ratio)
    max_difference = torch.max(differences)
    
    return 1 - differences / max_difference

    # 2D Dim Ratio
    gt_ratio = (gt_dim.tensor[0,2]-gt_dim.tensor[0,0])/(gt_dim.tensor[0,3]-gt_dim.tensor[0,1])
    pred_ratios = (pred_dims.tensor[:,2]-pred_dims.tensor[:,0])/(pred_dims.tensor[:,3]-pred_dims.tensor[:,1])

    differences = torch.abs(pred_ratios-gt_ratio)
    max_difference = torch.max(differences)
    
    return 1 - differences / max_difference

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