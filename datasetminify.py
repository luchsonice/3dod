import json

def minify_dataset(path, num_images=10):
    with open(path, 'r') as f:
        data = json.load(f)
    
    new_file = {}
    new_file['info'] = data['info']
    new_file['images'] = data['images'][:num_images]
    new_file['categories'] = data['categories'][:num_images]
    new_file['annotations'] = data['annotations'][:num_images]
    
    with open(path.replace('.json', '_mini.json'), 'w') as f:
        json.dump(new_file, f)

minify_dataset('datasets/Omni3D/SUNRGBD_test.json', 10)
minify_dataset('datasets/Omni3D/SUNRGBD_train.json', 10)
minify_dataset('datasets/Omni3D/SUNRGBD_val.json', 10)