# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import ExitStack
import itertools
import logging
import os
from detectron2.evaluation.evaluator import inference_context
import numpy as np
import torch
import datetime
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
)
from detectron2.utils.logger import setup_logger
import wandb
import torch.nn as nn

from cubercnn.evaluation.omni3d_evaluation import instances_to_coco_json

logger = logging.getLogger("cubercnn")

from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    build_detection_test_loader,
    simple_register
)
from cubercnn.evaluation import (
    Omni3DEvaluationHelper,
    inference_on_dataset
)
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis, data
# even though this import is unused, it initializes the backbone registry
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone


MAX_TRAINING_ATTEMPTS = 10



def inference_on_dataset_custom(model, data_loader):
    """
    Run model on the data_loader. 
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    inference_json = []

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            for input, output in zip(inputs, outputs):

                prediction = {
                    "image_id": input["image_id"],
                    "K": input["K"],
                    "width": input["width"],
                    "height": input["height"],
                }

                # convert to json format
                instances = output["instances"].to('cpu')
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

                # store in overall predictions
                inference_json.append(prediction)

    return inference_json



def do_test(cfg, model, iteration='final', storage=None):
        
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'iter_{}'.format(iteration))

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test, 
        filter_settings, 
        output_folder, 
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        """
        Cycle through each dataset and test them individually.
        This loop keeps track of each per-image evaluation result, 
        so that it doesn't need to be re-computed for the collective.
        """

        '''
        Distributed Cube R-CNN inference
        '''
        data_loader = build_detection_test_loader(cfg, dataset_name, num_workers=2)
        results_json = inference_on_dataset_custom(model, data_loader)

        '''
        Individual dataset evaluation
        '''
        eval_helper.add_predictions(dataset_name, results_json)
        eval_helper.save_predictions(dataset_name)
        eval_helper.evaluate(dataset_name)

        '''
        Optionally, visualize some instances
        '''
        instances = torch.load(os.path.join(output_folder, dataset_name, 'instances_predictions.pth'))
        log_str = vis.visualize_from_instances(
            instances, data_loader.dataset, dataset_name, 
            cfg.INPUT.MIN_SIZE_TEST, os.path.join(output_folder, dataset_name), 
            MetadataCatalog.get('omni3d_model').thing_classes, iteration
        )
        logger.info(log_str)

        
    '''
    Summarize each Omni3D Evaluation metric
    '''  
    eval_helper.summarize_all()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file
    
    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    cfg.SEED = 12 
    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)
    
    dataset_names_test = cfg.DATASETS.TEST

    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            simple_register(dataset_name, filter_settings, filter_empty=False)
    
    return cfg


def main(args):
    
    cfg = setup(args)
    
    name = f'cube {datetime.datetime.now().isoformat()}'
    # wandb.init(project="cube", sync_tensorboard=True, name=name, config=cfg)

    logger.info('Preprocessing Training Datasets')

    priors = None

    category_path = os.path.join(util.file_parts(args.opts[1])[0], 'category_meta.json')
    
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)

    # register the categories
    thing_classes = metadata['thing_classes']
    id_map = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
    MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id  = id_map


    # build the  model.
    model = build_model(cfg, priors=priors)

    # skip straight to eval mode
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    main(args)