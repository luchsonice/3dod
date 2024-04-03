# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import ExitStack
import itertools
import logging
import os
from detectron2.evaluation.evaluator import inference_context
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
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
from rich.progress import track

from cubercnn.data.dataset_mapper import DatasetMapper3D
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

def init_segmentation(device='cpu'):
    # 1) first cd into the segment_anything and pip install -e .
    # to get the model stary in the root foler folder and run the download_model.sh 
    # 2) chmod +x download_model.sh && ./download_model.sh
    # the largest model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # this is the smallest model
    sam_checkpoint = "segment-anything/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    print('SAM device:', device)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def inference_on_dataset_custom(model, data_loader, segmentor, output_recall_scores:bool=False):
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

        for idx, inputs in track(enumerate(data_loader), description="Inference", total=total):
            # model should be modified in cubercnn.modeling.roi_heads.cube_head.py class CubeHead_Vanilla forward function"

            outputs = model(inputs, segmentor, output_recall_scores=False)
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

def mean_average_best_overlap(model, data_loader, segmentor, output_recall_scores:bool):
        
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        outputs = []

        for idx, inputs in track(enumerate(data_loader), description="Making Mean average best overlap plots", total=total):
            if idx >2: break
            output = model(inputs, segmentor, output_recall_scores)
            outputs.append(output)

        # mean over all the outputs
        Iou3D = np.array([x[1] for x in outputs])
        Iou2D = np.array([x[2] for x in outputs])
        Iou3D = Iou3D.mean(axis=0)
        Iou2D = Iou2D.mean(axis=0)
        
        plt.figure(figsize=(8,5))
        plt.plot(Iou3D, marker='o', linestyle='-',c='purple') 
        plt.grid(True)
        plt.xscale('log')
        plt.xlabel('Number of Proposals')
        plt.ylabel('3D IoU')
        plt.title('Mean Average Best Overlap vs Number of Proposals')
        f_name = os.path.join('ProposalNetwork/output/MABO', 'MABO_segment.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        print('saved to ', f_name)


        # ## for debugging
        p_info = outputs[0][0]
        pred_box_classes_names = [util.MetadataCatalog.get('omni3d_model').thing_classes[i] for i in p_info.box_classes[:len(p_info.gt_boxes3D)]]
        fig, (ax, ax1) = plt.subplots(2,1, figsize=(14, 10))
        input = next(iter(data_loader))[0]
        images_raw = input['image']
        K = input['K']
        prop_img = images_raw.permute(1,2,0).cpu().numpy().copy()
        img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, K[0].cpu().numpy(), p_info.pred_cube_meshes,text=pred_box_classes_names, blend_weight=0.5, blend_weight_overlay=0.85,scale = prop_img.shape[0])
        vis_img_3d = img_3DPR.astype(np.uint8)
        ax.set_title('Predicted')
        ax.imshow(np.concatenate((vis_img_3d, img_novel), axis=1))
        box_size = p_info.gt_boxes3D.shape[0]
        v_pred = Visualizer(prop_img, None)
        v_pred = v_pred.overlay_instances(
            boxes=p_info.gt_boxes[0:box_size].tensor.cpu().numpy()
        )
        # prop_img = v_pred.get_image()
        gt_box_classes_names = [util.MetadataCatalog.get('omni3d_model').thing_classes[i] for i in p_info.gt_box_classes]
        img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, K[0].cpu().numpy(), p_info.gt_cube_meshes,text=gt_box_classes_names, blend_weight=0.5, blend_weight_overlay=0.85,scale = prop_img.shape[0])
        vis_img_3d = img_3DPR.astype(np.uint8)
        im_concat = np.concatenate((vis_img_3d, img_novel), axis=1)
        # for mask in mask_per_image:
        #     show_mask(mask[0].cpu().numpy(), ax1, random_color=True)
        ax1.set_title('GT')
        ax1.imshow(im_concat)
        plt.show()
        ##### end debugging




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

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper, num_workers=1)
        if cfg.PLOT.RECALL_SCORES: output_recall_scores = True
        else: output_recall_scores = False
        mean_average_best_overlap_scores = mean_average_best_overlap(model, data_loader, segmentor, output_recall_scores)
        results_json = inference_on_dataset_custom(model, data_loader, segmentor, output_recall_scores=False)

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
    cfg.SEED = 13
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
    args.opts.append('PLOT.RECALL_SCORES')
    args.opts.append(True)
    print("Command Line Args:", args)

    main(args)