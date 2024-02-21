import json

def minify_dataset(path, num_images=10):
    with open(path, 'r') as f:
        data = json.load(f)
    
    new_file = {}
    new_file['info'] = data['info']
    new_file['images'] = data['images'][:num_images]
    new_file['categories'] = data['categories']
    # grab only annotation for the image ids
    new_file['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in [img['id'] for img in new_file['images']]]
    
    with open(path.replace('.json', '_mini.json'), 'w') as f:
        json.dump(new_file, f)

minify_dataset('datasets/Omni3D/SUNRGBD_test.json', 10)
minify_dataset('datasets/Omni3D/SUNRGBD_train.json', 10)
minify_dataset('datasets/Omni3D/SUNRGBD_val.json', 10)