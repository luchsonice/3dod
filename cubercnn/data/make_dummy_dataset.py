import json
from PIL import Image
import os
from tqdm import tqdm

with open('datasets/Omni3D/KITTI_test.json', 'r') as f:
    a = json.load(f)

json_dict = {}

json_dict['info'] = a['info']
json_dict['info']['name'] = 'KITTI pred'
json_dict['info']['split'] = 'pred'
json_dict['categories'] = a['categories']
json_dict['id'] = 30
images = []

base_path = 'datasets/KITTI_object/testing/image_2/'
d_path = 'KITTI_object/testing/image_2/'
for image in tqdm(os.listdir(base_path)):
    img = Image.open(base_path + image)
    img_id = image.split('.')[0]
    w, h = img.size
    focal_length_ndc = 4.0
    focal_length = focal_length_ndc * h / 2
    px, py = w/2, h/2
    K = [
        [focal_length, 0.0, px], 
        [0.0, focal_length, py], 
        [0.0, 0.0, 1.0]
        ]
    images.append(
        {'width': w, 'height': h, 'file_path': d_path + image, 'K': K, 'src_90_rotate': 0,
        'src_flagged': False,
        'incomplete': False,
        'id': img_id,
        'dataset_id': 30}
    )

json_dict['images'] = images

with open('datasets/Omni3D/KITTI_pred.json', 'w') as f:
    json.dump(json_dict, f)