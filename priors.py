import os
import logging

from detectron2.config.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.logger import setup_logger
import pandas as pd
from cubercnn import data, util, vis
from cubercnn.config.config import get_cfg_defaults
from cubercnn.data.datasets import simple_register
        
logger = logging.getLogger("cubercnn")


def get_config_and_filter_settings(config_file='configs/Base_Omni3D.yaml'):
    # we must load the config file to get the filter settings
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(config_file)
    # must setup logger to get info about filtered out annotations
    setup_logger(output=cfg.OUTPUT_DIR, name="cubercnn")
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    return cfg, filter_settings

def priors_of_objects(dataset):
        
    cfg, filter_settings = get_config_and_filter_settings()

    dataset_names = ['SUNRGBD_train','SUNRGBD_val', 'SUNRGBD_test']
    for dataset_name in dataset_names:
        simple_register(dataset_name, filter_settings, filter_empty=True)

    dataset_paths = ['datasets/Omni3D/'+dataset_name+'.json' for dataset_name in dataset_names]

    datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)

    # determine the meta data given the datasets used. 
    data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
    
    '''
    It may be useful to keep track of which categories are annotated/known
    for each dataset in use, in case a method wants to use this information.
    '''

    infos = datasets.dataset['info']

    if type(infos) == dict:
        infos = [datasets.dataset['info']]

    dataset_id_to_unknown_cats = {}
    possible_categories = set(i for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1))
    
    dataset_id_to_src = {}

    dset_classes = []

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

        # log the per-dataset categories
        avail_cats = [thing_classes[i] for i in (possible_categories & known_category_training_ids)]
        logger.info('Available categories for {}'.format(info['name']))
        logger.info(avail_cats)

        dset_classes.append(avail_cats)

    # set difference between the available categories for each dataset.
    dset1 = set(dset_classes[0])
    dset2 = set(dset_classes[1])
    logger.info(f'Categories in {dataset_names[0]} missing from {dataset_names[1]}:')
    logger.info(dset1 - dset2)
    
    # compute priors given the training data.
    # interested in priors['priors_dims_per_cat'], some of them have [1,1,1], as they are invalid categories, e.g. car and bus which are not present SUNRGBD 
    # priors are w / h / d, std for (w / h / d) 
    priors = util.compute_priors(cfg, datasets)
    priors_bins = util.compute_priors(cfg, datasets,n_bins=5)
    # TODO: can we emulate the behaviour of this function
    # without ever having access to 3D annotations?
    priors2 = pd.read_csv('datasets/typical sizes of 3d items.csv')
    print(priors)
    pass


if __name__ == "__main__":
    priors_of_objects('SUNRGBD')
    