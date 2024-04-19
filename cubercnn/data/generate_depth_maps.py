import os
from detectron2.data.catalog import MetadataCatalog
import numpy as np
from ProposalNetwork.proposals.playground import setup_depth_model, depth_of_images
from PIL import Image

from cubercnn import data
from cubercnn.data.build import build_detection_test_loader, build_detection_train_loader
from cubercnn.data.dataset_mapper import DatasetMapper3D
from cubercnn.data.datasets import simple_register
from priors import get_config_and_filter_settings

from rich.progress import track

def init_dataset():
    ''' dataloader stuff.
    currently not used anywhere, because I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
    cfg, filter_settings = get_config_and_filter_settings()

    dataset_names = ['SUNRGBD_train','SUNRGBD_val','SUNRGBD_test']
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


datasets = init_dataset()

os.makedirs('datasets/depth_maps', exist_ok=True)

depth_model = 'zoedepth'
pretrained_resource = 'local::depth/checkpoints/depth_anything_metric_depth_indoor.pt'
model = setup_depth_model(depth_model, pretrained_resource)
for img_id, img_info in track(datasets.imgs.items()):
    file_path = img_info['file_path']
    width = img_info['width']
    height = img_info['height']
    img = np.array(Image.open('datasets/'+file_path))
    depth = depth_of_images(img, model)
    np.savez_compressed(f'datasets/depth_maps/{img_id}.npz', depth=depth)
