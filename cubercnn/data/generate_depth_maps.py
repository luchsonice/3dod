import os
from detectron2.data.catalog import MetadataCatalog
import numpy as np
from PIL import Image

from cubercnn import data
from priors import get_config_and_filter_settings
import torch
import torchvision.transforms as transforms
from depth.metric_depth.zoedepth.models.builder import build_model
from depth.metric_depth.zoedepth.utils.config import get_config

from rich.progress import track

def depth_of_images(image, model):
    """
    This function takes in a list of images and returns the depth of the images"""
    # Born out of Issue 36. 
    # Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
    # Make sure you have the necessary libraries
    # Code by @1ssb

    # Global settings
    DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

    color_image = Image.fromarray(image).convert('RGB')
    original_width, original_height = color_image.size
    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    pred_o = model(image_tensor, dataset=DATASET)
    if isinstance(pred_o, dict):
        pred = pred_o.get('metric_depth', pred_o.get('out'))
        features = pred_o.get('depth_features', None)
    elif isinstance(pred_o, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)

    # resized_pred is the image shaped to the original image size, depth is in meters
    return np.array(resized_pred)

def setup_depth_model(model_name, pretrained_resource):
    DATASET = 'nyu' # Lets not pick a fight with the model's dataloader
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

def init_dataset():
    ''' dataloader stuff.
     I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
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