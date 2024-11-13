import torch
from detectron2.data.catalog import MetadataCatalog
from cubercnn import data
from cubercnn.data import calculate_alpha
from detectron2.structures import Boxes, BoxMode
from cubercnn.util.math_util import estimate_truncation, mat2euler, R_to_allocentric
import os
import numpy as np
from tqdm import tqdm
from cubercnn import util

base_path = 'output/kitti_val_ours_K/KITTI_formatted_predictions'
for path in tqdm(os.listdir(base_path)):
    full_path = base_path + '/' + path
    with open(f'datasets/label_2_omni/{path}', 'r') as file:
        gt = file.readlines()
    with open(full_path, 'r') as file:
        dt = file.readlines()

    dt = [d.split() for d in dt]
    gt = [g.split() for g in gt]
    if len(dt) == 0:
        continue
    # overwrite the 13th number, corresponding to z
    dt_boxes = []
    gt_boxes = []
    for d in dt:
        x1, y1, x2, y2 = map(float, d[4:8])
        dt_boxes.append([x1, y1, x2, y2])
    for g in gt:
        if g[0] == 'DontCare' or g[0] == 'Misc' or g[0] == 'Tram' or g[0] == 'Person_sitting' or float(g[1]) == 1.00:
            continue
        gtx1, gty1, gtx2, gty2 = map(float, g[4:8])
        gt_boxes.append([gtx1, gty1, gtx2, gty2])
    dt_array = np.array(dt_boxes)
    gt_array = np.array(gt_boxes)
       
    # match dt's with gt's based on iou
    # iou matrix [[dt1 dt1, dt1 gt2, dt1 gt3],
    #             [dt2 gt1, dt2 gt2, dt2 gt3],
    #             [...]]
    # ie. each column is a dt, each row is a gt
    quality_matrix = util.iou(dt_array, gt_array)
    nearest_gt = quality_matrix.argmax(axis=1)[0]
    nearest_gt_iou = quality_matrix.max(axis=1)[0]
    # match detections based on their 2d iou 
    valid_match = quality_matrix >= 0.7

    for i, dt_ in enumerate(valid_match):
        for j in range(valid_match.shape[1]):
            if dt_[j]:
                dt[i][13] = gt[j][13]

    dt = [' '.join(d) + '\n' for d in dt]

    with open(full_path, 'w') as file:
        file.writelines(dt)