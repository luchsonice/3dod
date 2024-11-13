import torch
from detectron2.data.catalog import MetadataCatalog
from cubercnn import data
from cubercnn.data import calculate_alpha
from detectron2.structures import Boxes, BoxMode
from cubercnn.util.math_util import estimate_truncation, mat2euler, R_to_allocentric
import os
import numpy as np
from tqdm import tqdm

name = 'KITTI'
split = 'test'
dataset_paths_to_json = [f'datasets/Omni3D/{name}_{split}.json',]

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


# convert 
for img in tqdm(imgs):
    annIds = dataset.getAnnIds(imgIds=img['id'])
    anns = dataset.loadAnns(annIds)
    ys = []
    xs = []
    zs = []
    ws = []
    hs = []
    ls = []
    rys = []
    alphas = []
    for ann in anns:
        x, y, z = ann['center_cam']
        ys.append(y); xs.append(x); zs.append(z)
        h, w, l = ann['dimensions']
        ws.append(w); hs.append(h); ls.append(l)
        ry = mat2euler(np.array(ann['R_cam']))[1]
        alpha = calculate_alpha(ann['center_cam'], ry)
        rys.append(ry); alphas.append(alpha)

    path = img['file_path'].split('/')[-1][:6] + '.txt'
    with open(f'datasets/label_2_omni/{path}', 'r') as file:
    # read a list of lines into data
        d = file.readlines()

    # overwrite the 4th and 3rd last number, corresponding to x and y
    for i, line in enumerate(d):
        line = line.split()
        if line[0] == 'DontCare' or line[0] == 'Misc' or line[0] == 'Tram' or line[0] == 'Person_sitting' or float(line[1]) == 1.00:
            continue
        line[3] = str(round(alphas.pop(0),2))
        line[8] = str(round(hs.pop(0),2))
        line[9] = str(round(ws.pop(0),2))
        line[10] = str(round(ls.pop(0),2))
        line[11] = str(round(xs.pop(0),2))
        line[12] = str(round(ys.pop(0),2))
        line[13] = str(round(zs.pop(0),2))
        line[14] = str(round(rys.pop(0),2))

        line = ' '.join(line) + '\n'
        d[i] = line

    with open(f'datasets/label_2_omni/{path}', 'w') as f:
        for line in d:
            f.write(line)