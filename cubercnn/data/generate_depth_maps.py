import torch
import cv2
# might need to export PYTHONPATH=/work3/$username/3dod/
from depth.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
def depth_of_images(encoder='vitl', dataset='hypersim', max_depth=20, device='cpu'):
    """
    This function takes in a list of images and returns the depth of the images
    
    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device, weights_only=False))
    model.eval()
    model.to(device)
    return model

def init_dataset():
    ''' dataloader stuff.
     I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
    cfg, filter_settings = get_config_and_filter_settings()

    dataset_names = ['SUNRGBD_train','SUNRGBD_val','SUNRGBD_test', 'KITTI_train', 'KITTI_val', 'KITTI_test',]
    dataset_paths_to_json = ['datasets/Omni3D/'+dataset_name+'.json' for dataset_name in dataset_names]
    # for dataset_name in dataset_names:
    #     simple_register(dataset_name, filter_settings, filter_empty=True)

    # Get Image and annotations
    datasets = data.Omni3D(dataset_paths_to_json, filter_settings=filter_settings)
    data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

    infos = datasets.dataset['info']

    dataset_id_to_unknown_cats = {}
    possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
    
    dataset_id_to_src = {}

    for info in infos:
        dataset_id = info['id']
        known_category_training_ids = set()
        
        if not dataset_id in dataset_id_to_src:
            dataset_id_to_src[dataset_id] = info['source']

        for id in info['known_category_ids']:
            if id in dataset_id_to_contiguous_id:
                known_category_training_ids.add(dataset_id_to_contiguous_id[id])
        
        # determine and store the unknown categories.
        unknown_categories = possible_categories - known_category_training_ids
        dataset_id_to_unknown_cats[dataset_id] = unknown_categories

    return datasets

if __name__ == '__main__':
    import os
    from detectron2.data.catalog import MetadataCatalog
    import numpy as np

    from cubercnn import data
    from priors import get_config_and_filter_settings

    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = init_dataset()

    os.makedirs('datasets/depth_maps', exist_ok=True)

    model = depth_of_images(device=device)

    for img_id, img_info in tqdm(datasets.imgs.items()):
        file_path = img_info['file_path']
        img = cv2.imread('datasets/'+file_path)
        depth = model.infer_image(img) # HxW depth map in meters in numpy
        np.savez_compressed(f'datasets/depth_maps/{img_id}.npz', depth=depth)