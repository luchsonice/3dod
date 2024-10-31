import json
from detectron2.data.catalog import MetadataCatalog
from cubercnn import data
from cubercnn.util.math_util import estimate_truncation
import os
import numpy as np

name = 'KITTI'
split = 'test_mini'
dataset_paths_to_json = [f'datasets/Omni3D/{name}_{split}.json',]
os.makedirs('output/KITTI_formatted_predictions', exist_ok=True)

# Example 1. load all images
dataset = data.Omni3D(dataset_paths_to_json)
imgIds = dataset.getImgIds()
imgs = dataset.loadImgs(imgIds)

# Example 2. load annotations for image index 0
annIds = dataset.getAnnIds(imgIds=imgs[0]['id'])
anns = dataset.loadAnns(annIds)

data.register_and_store_model_metadata(dataset, 'output')

thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id



with open('output/weak-cube-w_low/inference/iter_final/SUNRGBD_test_mini2/omni_instances_results.json', 'r') as f:
    data_json = json.load(f)

a = 2

types = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
# omni3d only has these ['pedestrian', 'car', 'cyclist', 'van', 'truck']

# reference
# https://github.com/ZrrSkywalker/MonoDETR/blob/c724572bddbc067832a0e0d860a411003f36c2fa/lib/helpers/tester_helper.py#L114
for annotation in data_json:
    for keys in annotation:
        for key in keys: #image_id, category_id, bbox, score, depth, bbox3D, center_cam, center_2D, dimensions, pose
            category = dataset_id_to_contiguous_id[annotation[key]]
            truncation = estimate_truncation(K, bbox3D, pose, width, height)
            occluded = 3
            alpha = 0
            bbox = bbox
            dimensions = dimensions
            location = center_cam
            rotation_y = np.atan2(pose[0, 0], pose[2, 0])
            score = score

# 7518 test images
for img_id in range(7518):

    img_id_str = str(img_id).zfill(6)
    with open(f'output/KITTI_formatted_predictions/{img_id_str}.txt', 'w') as f:
        f.write()

# write to file 
# #Values    Name      Description
# ----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.

# output to files 000000.txt 000001.txt ... 

# example file
# Car 0.00 0 -1.56 564.62 174.59 616.43 224.74 1.61 1.66 3.20 -0.69 1.69 25.01 -1.59
# Car 0.00 0 1.71 481.59 180.09 512.55 202.42 1.40 1.51 3.70 -7.43 1.88 47.55 1.55
# Car 0.00 0 1.64 542.05 175.55 565.27 193.79 1.46 1.66 4.05 -4.71 1.71 60.52 1.56
# Cyclist 0.00 0 1.89 330.60 176.09 355.61 213.60 1.72 0.50 1.95 -12.63 1.88 34.09 1.54
# DontCare -1 -1 -10 753.33 164.32 798.00 186.74 -1 -1 -1 -1000 -1000 -1000 -10
# DontCare -1 -1 -10 738.50 171.32 753.27 184.42 -1 -1 -1 -1000 -1000 -1000 -10