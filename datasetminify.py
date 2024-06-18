import json
import random
random.seed(0)

def minify_dataset(path, num_images=10):
    with open(path, 'r') as f:
        data = json.load(f)
    
    new_file = {}
    new_file['info'] = data['info']
    idx = random.sample(range(len(data['images'])), num_images)
    new_file['images'] = [data['images'][i] for i in idx]
    new_file['categories'] = data['categories']
    # grab only annotation for the image ids
    new_file['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in new_file['images']]]
    
    with open(path.replace('.json', '_mini.json'), 'w') as f:
        json.dump(new_file, f)

cats = set({'bicycle', 'books', 'bottle', 'chair', 'cup', 'laptop', 'shoes', 'towel', 'blinds', 'window', 'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat', 'curtain', 'clothes', 'stationery', 'refrigerator', 'bin', 'stove', 'oven', 'machine'})
n_images = 103
minify_dataset('datasets/Omni3D/SUNRGBD_test.json', n_images*2)
minify_dataset('datasets/Omni3D/SUNRGBD_train.json', n_images)
minify_dataset('datasets/Omni3D/SUNRGBD_val.json', n_images)

def minify_dataset_cats(path, cats):
    '''make a mini dataset which has all the specified categories'''
    with open(path, 'r') as f:
        data = json.load(f)
    
    new_file = {}
    new_file['info'] = data['info']
    i = 0
    while len(cats) > 0:
        idx = random.sample(range(len(data['images'])), 1)
        new_file['images'] = [data['images'][i] for i in idx]
        # grab only annotation for the image ids
        new_file['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in new_file['images']]]
        # check if all categories are present
        cat_in_img = set([i['category_name'] for i in new_file['annotations']])
        cats = cats - cat_in_img
        i += 1
    print('num_ ', i)
    with open(path.replace('.json', '_mini.json'), 'w') as f:
        json.dump(new_file, f)


# minify_dataset_cats('datasets/Omni3D/SUNRGBD_test.json', cats)
# minify_dataset_cats('datasets/Omni3D/SUNRGBD_train.json', cats)
# minify_dataset_cats('datasets/Omni3D/SUNRGBD_val.json', cats)

# def minify_dataset_idx(path, idx):
#     with open(path, 'r') as f:
#         data = json.load(f)
    
#     new_file = {}
#     new_file['info'] = data['info']
#     # find only image with idx
#     new_file['images'] = [i for i in data['images'] if i['id'] == idx]
#     new_file['categories'] = data['categories']
#     # grab only annotation for the image ids
#     new_file['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in new_file['images']]]
    
#     with open(path.replace('_train.json', '_test_mini.json'), 'w') as f:
#         json.dump(new_file, f)

# minify_dataset_idx('datasets/Omni3D/SUNRGBD_train.json', 167896)