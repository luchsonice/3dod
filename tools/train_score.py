import warnings
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry")

# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import ExitStack
import logging
import os
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import torch
import datetime

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
)
from detectron2.utils.logger import setup_logger
import torch.nn as nn
from rich.progress import track
import pickle

from cubercnn.data.dataset_mapper import DatasetMapper3D

logger = logging.getLogger("scoring")

from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    build_detection_test_loader,
    simple_register
)
from cubercnn.evaluation import (
    Omni3DEvaluationHelper,
)
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis, data
# even though this import is unused, it initializes the backbone registry
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone

# Below imports followed with do_train
import torch.distributed as dist
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
    default_writers, 
    launch
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
import wandb
from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.data import (
    load_omni3d_json,
    DatasetMapper3D,
    build_detection_train_loader,
    build_detection_test_loader,
    get_omni3d_categories,
    simple_register
)
from cubercnn.evaluation import (
    Omni3DEvaluator, Omni3Deval,
    Omni3DEvaluationHelper,
    inference_on_dataset
)

def init_segmentation(device='cpu') -> Sam:
    # 1) first cd into the segment_anything and pip install -e .
    # to get the model stary in the root foler folder and run the download_model.sh 
    # 2) chmod +x download_model.sh && ./download_model.sh
    # the largest model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # this is the smallest model
    if os.path.exists('sam-hq/sam_hq_vit_b.pth'):
        sam_checkpoint = "sam-hq/sam_hq_vit_b.pth"
        model_type = "vit_b"
    else:
        sam_checkpoint = "sam-hq/sam_hq_vit_tiny.pth"
        model_type = "vit_tiny"
    logger.info(f'SAM device: {device}, model_type: {model_type}')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def do_test(cfg, model, iteration='final', storage=None):
        
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'iter_{}'.format(iteration))

    segmentor = init_segmentation(device=cfg.MODEL.DEVICE)

    for dataset_name in dataset_names_test:
        """
        Cycle through each dataset and test them individually.
        This loop keeps track of each per-image evaluation result, 
        so that it doesn't need to be re-computed for the collective.
        """

        '''
        Distributed Cube R-CNN inference
        '''
        dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TEST]
        datasets = data.Omni3D(dataset_paths, filter_settings=filter_settings)

        # determine the meta data given the datasets used. 
        data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        
        infos = datasets.dataset['info']

        if type(infos) == dict:
            infos = [datasets.dataset['info']]

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


        # we need the dataset mapper to get 
        data_mapper = DatasetMapper3D(cfg, is_train=False, mode='eval_with_gt')
        data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=1)

        experiment_type = {}

        if cfg.PLOT.EVAL == 'MABO': experiment_type['output_recall_scores'] = True
        else: experiment_type['output_recall_scores'] = False
        # either use pred_boxes or GT boxes
        if cfg.PLOT.MODE2D == 'PRED': experiment_type['use_pred_boxes'] = True
        else: experiment_type['use_pred_boxes'] = False
        if experiment_type['output_recall_scores']:
            _ = mean_average_best_overlap(model, data_loader, segmentor, experiment_type)
        
        else:
            results_json = inference_on_dataset(model, data_loader, segmentor, experiment_type)

            eval_helper = Omni3DEvaluationHelper(
                dataset_names_test, 
                filter_settings, 
                output_folder, 
                iter_label=iteration,
                only_2d=only_2d,
            )
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
                MetadataCatalog.get('omni3d_model').thing_classes, iteration, visualize_every=1
            )
            logger.info(log_str)

        
    if cfg.PLOT.EVAL != 'MABO':
        '''
        Summarize each Omni3D Evaluation metric
        '''  
        eval_helper.summarize_all()





def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False):

    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # bookkeeping
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)    
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    
    # create the dataloader
    data_mapper = DatasetMapper3D(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src, num_workers=4)

    # give the mapper access to dataset_ids
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

    if cfg.MODEL.WEIGHTS_PRETRAIN != '':
        
        # load ONLY the model, no checkpointables.
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # determine the starting iteration, if resuming
    start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))

    if not cfg.MODEL.USE_BN:
        freeze_bn(model)

    data_iter = iter(data_loader)

    with EventStorage(start_iter) as storage:
        
        while True:

            data = next(data_iter)
            storage.iter = iteration

            # forward
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            
            # send loss scalars to tensorboard.
            storage.put_scalars(total_loss=losses, **loss_dict)
        
            # backward and step
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # Evaluate only when the loss is not diverging.
            if (do_eval and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0 and iteration != (max_iter - 1)):
                logger.info('Starting test for iteration {}'.format(iteration+1))
                do_test(cfg, model, iteration=iteration+1, storage=storage)
                
            if not cfg.MODEL.USE_BN: 
                freeze_bn(model)

            periodic_checkpointer.step(iteration)

            iteration += 1
            if iteration >= max_iter:
                break
    
    # success
    return True

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
    cfg.SEED = 13
    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, name="scoring")
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)
    
    dataset_names_test = cfg.DATASETS.TEST

    # filter_ = True if cfg.PLOT.EVAL == 'MABO' else False
    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            # TODO: empties should not be filtering in test normally, or maybe they should??
            simple_register(dataset_name, filter_settings, filter_empty=True)
    
    return cfg


def main(args):
    
    cfg = setup(args)
    
    name = f'cube {datetime.datetime.now().isoformat()}'
    wandb.init(project="cube", sync_tensorboard=True, name=name, config=cfg)

    priors = None
    with open('filetransfer/priors.pkl', 'rb') as f:
        priors, _ = pickle.load(f)

    category_path = 'output/Baseline_sgd/category_meta.json'
    
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
    # load the saved model if using eval boxes
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return do_train(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.opts.append('PLOT.EVAL')
    # args.opts.append('MABO') or 'AP'
    print("Command Line Args:", args)

    main(args)