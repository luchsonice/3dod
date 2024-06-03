# Copyright (c) Meta Platforms, Inc. and affiliates
import sys
import warnings

warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry")
from cubercnn.data.generate_ground_segmentations import init_segmentation

import logging
import os
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

from cubercnn.data.dataset_mapper import DatasetMapper3D

logger = logging.getLogger("scoring")

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model, build_model_scorenet
from cubercnn import util, vis, data
# even though this import is unused, it initializes the backbone registry
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone

# Below imports followed with do_train
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

from tqdm import tqdm


def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False):
    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    modelbase = model[0]
    modelbase.eval()
    model = model[1]
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # bookkeeping
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    # create the dataloader
    
    data_mapper = DatasetMapper3D(cfg, is_train=False, mode='load_proposals')
    dataset_name = cfg.DATASETS.TRAIN[0]
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
        freeze_bn(modelbase)

    data_iter = iter(data_loader)
    pbar = tqdm(range(start_iter, max_iter), initial=start_iter, total=max_iter, desc="Training", smoothing=0.05)

    segmentor = init_segmentation(device=cfg.MODEL.DEVICE)

    with EventStorage(start_iter) as storage:

        while True:
            data = next(data_iter)
            storage.iter = iteration
            # forward
            combined_features = modelbase(data)
            loss_1, loss_2, instances = model(data, combined_features)
            # scale the dimension L1-loss by a factor of 1000 to have both the scoring and regression losses in a similar range
            loss_1 /= 2
            loss_1 /= len(data)
            #loss_2 = 0.5 * loss_2/1_000
            total_loss = loss_1 + loss_2
            # send loss scalars to tensorboard.
            storage.put_scalars(total_loss=total_loss, IoU_loss=loss_1, second_loss=loss_2)

            # backward and step
            total_loss.backward()
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(name, param.grad)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            periodic_checkpointer.step(iteration)

            # logging stuff 
            pbar.update(1)
            pbar.set_postfix({"tot.loss": total_loss.item(), "S.loss": loss_1.item(), "R.loss": loss_2.item()})
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers[1:]: # 3 writers; 1: prints, 2: json logs, 3: tensorboard
                    writer.write()
            
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
    
    name = f'learned proposal {datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'
    
    if sys.platform == 'linux':
        # only log to wandb on hpc/linux
        #wandb.init(project="cube", sync_tensorboard=True, name=name, config=cfg, mode='online')
        True

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
    modelbase = build_model_scorenet(cfg, 'ScoreNetBase')
    model = build_model_scorenet(cfg, 'ScoreNet')

    filter_settings = data.get_filter_settings_from_cfg(cfg)

    # setup and join the data.
    dataset_paths = [os.path.join('datasets', 'Omni3D', name + '.json') for name in cfg.DATASETS.TRAIN]
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
        # logger.info('Available categories for {}'.format(info['name']))
        # logger.info([thing_classes[i] for i in (possible_categories & known_category_training_ids)])
    
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    return do_train(cfg, (modelbase, model), dataset_id_to_unknown_cats, dataset_id_to_src, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    main(args)