import torch
from detectron2.data.catalog import MetadataCatalog
from cubercnn import data
from detectron2.structures import Boxes, BoxMode
from cubercnn.util.math_util import estimate_truncation, mat2euler, R_to_allocentric
import os
import numpy as np
from tqdm import tqdm

def perp_vector(a, b):
    return np.array([b, -a])  

def rotate_vector(x, y, theta):
    # Calculate the rotated coordinates
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
    
    return np.array([x_rotated, y_rotated])

def calculate_alpha(location, ry):
    '''
    location: x, y, z coordinates
    ry: rotation around y-axis, negative counter-clockwise,
    
    positive x-axis is to the right
    calculate the angle from a line perpendicular to the camera to the center of the bounding box'''

    # get vector from camera to object
    ry = -ry
    x, y, z = location
    # vector from [0,0,0] to the center of the bounding box
    # we can do the whole thing in 2D, top down view
    # vector perpendicular to center
    perpendicular = perp_vector(x,z)
    # vector corresponding to ry
    ry_vector = np.array([np.cos(ry), np.sin(ry)])
    # angle between perpendicular and ry_vector
    dot = perpendicular[0]*ry_vector[0] + perpendicular[1]*ry_vector[1]      # Dot product between [x1, y1] and [x2, y2]
    det = perpendicular[0]*ry_vector[1] - perpendicular[1]*ry_vector[0]      # Determinant
    alpha = -np.arctan2(det, dot)

    # wrap to -pi to pi
    if alpha > np.pi:
        alpha -= 2*np.pi
    if alpha < -np.pi:
        alpha += 2*np.pi
    return alpha

def test_calculate_alpha():
    location = [-3.67, 1.67, 6.05]
    ry = -1.24
    expected = -0.72
    result1 = calculate_alpha(location, ry)

    location = [-9.48, 2.08, 26.41]
    ry = 1.77
    expected = 2.11
    result2 = calculate_alpha(location, ry)

    location = [4.19, 1.46, 44.41]
    ry = -1.35
    expected = -1.45
    result3 = calculate_alpha(location, ry)

    location = [-6.41, 2.04, 46.74]
    ry = 1.68
    expected = 1.82
    result4 = calculate_alpha(location, ry)

    location = [0.28, 2.08, 17.74]
    ry = -1.58
    expected = -1.59
    result5 = calculate_alpha(location, ry)

    location = [-3.21, 1.97, 11.22]
    ry = -0.13
    expected = 0.15
    result6 = calculate_alpha(location, ry)

    # assert np.isclose(result, expected, atol=0.01)
    return result1


def main():
    alpha = test_calculate_alpha()


    name = 'KITTI'
    split = 'test'
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
    cats = {'pedestrian', 'car', 'cyclist', 'van', 'truck'}

    input_folder = 'kitti_omni_eq'

    out_path = 'output/'+input_folder+'/KITTI_formatted_predictions/'
    in_path = 'output/'+input_folder+'/KITTI_pred/instances_predictions.pth'
    print('saving to', out_path)
    data_json = torch.load(in_path)
    # 
    # reference
    # https://github.com/ZrrSkywalker/MonoDETR/blob/c724572bddbc067832a0e0d860a411003f36c2fa/lib/helpers/tester_helper.py#L114
    files = {}
    for image in tqdm(data_json):
        K = image['K']
        K_inv = np.linalg.inv(K)
        width, height = image['width'], image['height']
        image_id = image['image_id']
        l = []
        for pred in image['instances']:

            category = thing_classes[pred['category_id']]
            if category not in cats:
                continue
            occluded = 0
            # truncation = estimate_truncation(K, torch.tensor([x3d, y3d, z3d, w3d, h3d, l3d]), pred['pose'], width, height)
            truncation = 0.0 # it does not matter
            rotation_y = mat2euler(np.array(pred['pose']))[1]
            bbox = BoxMode.convert(pred['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) # x1, y1, x2, y2 -> convert to left, top, right, bottom
            h3d, w3d, l3d = pred['dimensions']
            # unproject, this should yield the same 
            # cen_2d = np.array(pred['center_2D'] + [1])
            # z3d = pred['center_cam'][2]
            # x3d, y3d, z3d = (K_inv @ (z3d*cen_2d))

            x3d, y3d, z3d = pred['center_cam']

            location = pred['center_cam']
            score = pred['score']
            alpha = calculate_alpha(location, rotation_y)

            # convert to KITTI format
            li = [category, truncation, occluded, alpha, bbox[0], bbox[1], bbox[2], bbox[3], h3d, w3d, l3d, x3d, y3d, z3d, rotation_y, score]
            l.append(li)
        # sort l by z3d
        l = sorted(l, key=lambda x: x[13])
        files[image_id] = l

    # 7518 test images
    os.makedirs(out_path, exist_ok=True)
    for img_id, content in files.items():

        img_id_str = str(img_id).zfill(6)
        with open(out_path+f'{img_id_str}.txt', 'w') as f:
            str_i = ''
            for i in content:
                # t = f'{category} {truncation:.2f} {occluded} {alpha:.2f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} {w3d:.2f} {h3d:.2f} {l3d:.2f} {x3d:.2f} {y3d:.2f} {z3d:.2f} {rotation_y:.2f} {score:.2f}\n'
                t = f'{i[0][0].upper() + i[0][1:]} {i[1]:.2f} {i[2]} {i[3]:.2f} {i[4]:.2f} {i[5]:.2f} {i[6]:.2f} {i[7]:.2f} {i[8]:.2f} {i[9]:.2f} {i[10]:.2f} {i[11]:.2f} {i[12]:.2f} {i[13]:.2f} {i[14]:.2f} {i[15]:.2f}\n'
                str_i += t
            f.write(str_i)

if __name__ == '__main__':
    main()

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
#                      detection, needuhued for p/r curves, higher is better.

# output to files 000000.txt 000001.txt ... 

# example file
# Car 0.00 0 -1.56 564.62 174.59 616.43 224.74 1.61 1.66 3.20 -0.69 1.69 25.01 -1.59
# Car 0.00 0 1.71 481.59 180.09 512.55 202.42 1.40 1.51 3.70 -7.43 1.88 47.55 1.55
# Car 0.00 0 1.64 542.05 175.55 565.27 193.79 1.46 1.66 4.05 -4.71 1.71 60.52 1.56
# Cyclist 0.00 0 1.89 330.60 176.09 355.61 213.60 1.72 0.50 1.95 -12.63 1.88 34.09 1.54
# DontCare -1 -1 -10 753.33 164.32 798.00 186.74 -1 -1 -1 -1000 -1000 -1000 -10
# DontCare -1 -1 -10 738.50 171.32 753.27 184.42 -1 -1 -1 -1000 -1000 -1000 -10