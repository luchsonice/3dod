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

# minify_dataset('datasets/Omni3D/SUNRGBD_test.json', 15)
# minify_dataset('datasets/Omni3D/SUNRGBD_train.json', 15)
# minify_dataset('datasets/Omni3D/SUNRGBD_val.json', 15)

def minify_dataset_idx(path, idx):
    with open(path, 'r') as f:
        data = json.load(f)
    
    new_file = {}
    new_file['info'] = data['info']
    # find only image with idx
    new_file['images'] = [i for i in data['images'] if i['id'] == idx]
    new_file['categories'] = data['categories']
    # grab only annotation for the image ids
    new_file['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in new_file['images']]]
    
    with open(path.replace('_train.json', '_test_mini.json'), 'w') as f:
        json.dump(new_file, f)

minify_dataset_idx('datasets/Omni3D/SUNRGBD_train.json', 167896)