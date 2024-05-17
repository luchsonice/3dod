import warnings

from cubercnn.data.build import build_detection_train_loader
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry")

# Copyright (c) Meta Platforms, Inc. and affiliates
from contextlib import ExitStack
import logging
import os
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.evaluation.evaluator import inference_context
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
import numpy as np
from cubercnn.data.generate_ground_segmentations import init_segmentation
import torch
import datetime
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
)
from detectron2.utils.logger import setup_logger
import torch.nn as nn
from tqdm import tqdm
import pickle

from ProposalNetwork.utils.utils import show_mask2
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
)
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis, data
# even though this import is unused, it initializes the backbone registry
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone


def inference_on_dataset(model, data_loader, segmentor, experiment_type, proposal_function):
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

        for idx, inputs in tqdm(enumerate(data_loader), desc="Average Precision", total=total):
            outputs = model(inputs, segmentor, experiment_type, proposal_function)
            for input, output in zip(inputs, outputs):

                prediction = {
                    "image_id": input["image_id"],
                    "K": input["K"],
                    "width": input["width"],
                    "height": input["height"],
                }

                # convert to json format
                instances = output.to('cpu')
                # instances = output["instances"].to('cpu')
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

                # store in overall predictions
                inference_json.append(prediction)

    return inference_json

def percent_of_boxes(model, data_loader, segmentor, experiment_type, proposal_functions):
    '''make the detection that have a certain 3D IoU score plots
    if you give the proposal function as input to argparser as:
    
    `PLOT.PROPOSAL_FUNC, ['random', 'z', 'xy', 'dim', 'rotation', 'aspect' ,'full']`
    
    it will work
    '''
    total = len(data_loader)  # inference data loader must have a fixed length

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        torch.set_float32_matmul_precision('high')
        outputs = []
        for i, inputs in tqdm(enumerate(data_loader), desc=f"IoU3D plots, proposal method: {proposal_functions}", total=total):
            output = model(inputs, segmentor, experiment_type, proposal_functions)
            outputs.append(output.numpy())
        np.savez_compressed('ProposalNetwork/output/outputs.npz', outputs=outputs)

        xlim = [0.2,1]
        IoUat = [0.15, 0.25, 0.4]
        
        fig, axes = plt.subplots(1, figsize=(7.5,5))
        fig2, axes2 = plt.subplots(1, len(IoUat), figsize=(20,5))
        axes.set_ylabel('Detection rate')
        axes.set_ylim([0,1])
        axes.grid(True)
        axes.set_xlabel('3D Intersection over Union')
        axes.set_xlim(xlim)
        axes.set_title('Varying proposal method, 1000 proposals')
        for Iou, ax in zip(IoUat, axes2):
            ax.set_ylabel('Detection rate')
            ax.set_ylim([0,1])
            ax.grid(True)
            ax.set_title(f'Variants, IoU3D = {Iou}')
            ax.set_xlim([1,1000])
            ax.set_xlabel('Number of Proposals')

        for k, proposal_function in enumerate(proposal_functions):
            IoU3Ds = np.concatenate([x[:,k,:] for x in outputs])
            maxIOU_per_instance = np.max(IoU3Ds,axis=1)
            sorted_IoU3D = np.sort(IoU3Ds,axis=1)
            # detection rate vs. IoU3D
            thresholds = np.arange(xlim[0],xlim[1],0.025)
            detection_rate = np.zeros(len(thresholds))
            for i in range(len(thresholds)):
                detection_rate[i] = np.mean(maxIOU_per_instance > thresholds[i],axis=0)

            # detection rate vs. no. of proposals
            detection_rates = np.zeros((len(IoUat), IoU3Ds.shape[1]))
            for i, IoU in enumerate(IoUat):
                detection_rates[i] = np.mean(sorted_IoU3D > IoU,axis=0)

            axes.plot(thresholds, detection_rate, label=f'{proposal_function}')
            for j, ax in enumerate(axes2):
                ax.plot(list(range(1, 1001)), detection_rates[j], label=f'{proposal_function}')
    for ax in axes2:
        ax.legend()
    axes.legend()
    fig.savefig('ProposalNetwork/output/detection_rate.png', dpi=300, bbox_inches='tight')
    fig2.savefig('ProposalNetwork/output/IoU_varying.png', dpi=300, bbox_inches='tight')


    return
            

def mean_average_best_overlap(model, data_loader, segmentor, experiment_type, proposal_function):
        
    total = len(data_loader)  # inference data loader must have a fixed length

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        outputs = []
        for i, inputs in tqdm(enumerate(data_loader), desc="Mean average best overlap plots", total=total):
            output = model(inputs, segmentor, experiment_type, proposal_function)
            # p_info, IoU3D, score_IoU2D, score_seg, score_dim, score_combined, score_random, score_point_cloud, stat_empty_boxes, stats_im, stats_off, stats_off_impro
            if output is not None:
                outputs.append(output)
            """      
            Iou2D = np.concatenate([np.array(sublist) for sublist in (x[1] for x in outputs)])
            Iou2D = Iou2D.mean(axis=0)
            score_random = np.concatenate([np.array(sublist) for sublist in (x[5] for x in outputs)])
            score_random = score_random.mean(axis=0)
            total_num_instances = np.sum([x[0].gt_boxes3D.shape[0] for x in outputs])
            plt.figure(figsize=(8,5))
            plt.plot(Iou2D, linestyle='-',c='orange',label='2d IoU') 
            plt.plot(score_random, linestyle='-',c='grey',label='random') 
            plt.grid(True)
            plt.xscale('log')
            plt.xlim(left=1)
            plt.xlabel('Number of Proposals')
            plt.ylabel('3D IoU')
            plt.legend()
            plt.title('Mean Average Best Overlap vs Number of Proposals ({} images, {} instances)'.format(1+i,total_num_instances))
            f_name = os.path.join('ProposalNetwork/output/MABO', 'MABO.png')
            plt.savefig(f_name, dpi=300, bbox_inches='tight')
            plt.close()
            """
            

        # mean over all the outputs
        Iou2D             = np.concatenate([np.array(sublist) for sublist in (x[1] for x in outputs)])
        score_seg         = np.concatenate([np.array(sublist) for sublist in (x[2] for x in outputs)])
        score_dim         = np.concatenate([np.array(sublist) for sublist in (x[3] for x in outputs)])
        score_combined    = np.concatenate([np.array(sublist) for sublist in (x[4] for x in outputs)])
        score_random      = np.concatenate([np.array(sublist) for sublist in (x[5] for x in outputs)])
        score_point_cloud = np.concatenate([np.array(sublist) for sublist in (x[6] for x in outputs)])
        stat_empty_boxes  = np.array([x[7] for x in outputs])
        #logger.info('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))
        print('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))

        Iou2D = Iou2D.mean(axis=0)
        score_seg = score_seg.mean(axis=0)
        score_dim = score_dim.mean(axis=0)
        score_combined = score_combined.mean(axis=0)
        score_random = score_random.mean(axis=0)
        score_point_cloud = score_point_cloud.mean(axis=0)
        total_num_instances = np.sum([x[0].gt_boxes3D.shape[0] for x in outputs])
                
        plt.figure(figsize=(8,5))
        plt.plot(score_combined, linestyle='-',c='black', label='combined') 
        plt.plot(score_dim, linestyle='-',c='teal',label='dim') 
        plt.plot(score_seg, linestyle='-',c='purple',label='segment')
        plt.plot(Iou2D, linestyle='-',c='orange',label='2d IoU') 
        plt.plot(score_random, linestyle='-',c='grey',label='random') 
        plt.plot(score_point_cloud, linestyle='-',c='green',label='point cloud')
        plt.grid(True)
        plt.xscale('log')
        plt.xlim(left=1)
        plt.xlabel('Number of Proposals')
        plt.ylabel('3D IoU')
        plt.legend()
        plt.title('Mean Average Best Overlap vs Number of Proposals ({} images, {} instances)'.format(1+i,total_num_instances))
        f_name = os.path.join('ProposalNetwork/output/MABO', 'MABO.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name)
        print('saved to ', f_name)

        # Statistics
        stats = torch.cat([x[8] for x in outputs],dim=0)
        num_bins = 40
        titles = ['x','y','z','w','h','l','rx','ry','rz']
        plt.figure(figsize=(15, 15))
        plt.suptitle("Histogram about the Ground Truths in Normalised Perspective to Searched Range", fontsize=20)
        for i,title in enumerate(titles):
            plt.subplot(3, 3, 1+i)
            plt.hist(stats[:,i].numpy(), bins=num_bins, color='darkslategrey',density=True)
            plt.axvline(x=0, color='red')
            plt.axvline(x=1, color='red')
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO', 'stats.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name
        print('saved to ', f_name)

        stats_off = np.concatenate([np.array(sublist) for sublist in (x[9] for x in outputs)])
        plt.figure(figsize=(15, 15))
        for i,title in enumerate(titles):
            plt.subplot(3, 3, 1+i)
            plt.scatter(stats_off[:,1+i],stats_off[:,0])
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO', 'stats_off.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name)
        print('saved to ', f_name)

        plt.figure(figsize=(15, 15))
        for i,title in enumerate(titles):
            plt.subplot(3, 3, 1+i)
            plt.scatter(stats_off[:,1+i],stats_off[:,0])
            plt.title(title)
            plt.xlim([0,2])
            plt.ylim([0,1])
        f_name = os.path.join('ProposalNetwork/output/MABO', 'stats_off_zoom.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name)
        print('saved to ', f_name)
        
        # ## for vis
        d_iter = iter(data_loader)
        for i , _ in tqdm(enumerate(outputs), desc="Plotting every single image", total=len(outputs)):
            p_info = outputs[i][0]
            pred_box_classes_names = [util.MetadataCatalog.get('omni3d_model').thing_classes[label] for label in p_info.pred_cubes.labels.cpu().numpy()]
            box_size = p_info.pred_cubes.num_instances
            for x in range(box_size-len(pred_box_classes_names)):
                pred_box_classes_names.append(f'z={p_info.pred_cubes[x].dimensions[2]}, s={p_info.pred_cubes[x].scores}')
            colors = [np.concatenate([np.random.random(3), np.array([0.6])], axis=0) for _ in range(box_size)]
            fig, (ax, ax1) = plt.subplots(2,1, figsize=(14, 10))
            input = next(d_iter)[0]
            images_raw = input['image']
            
            prop_img = convert_image_to_rgb(images_raw.permute(1,2,0).cpu().numpy(), 'BGR').copy()
            v_pred = Visualizer(prop_img, None)
            v_pred = v_pred.overlay_instances(
                boxes=p_info.gt_boxes[0:box_size].tensor.cpu().numpy()
                , assigned_colors=colors
            )
            prop_img = v_pred.get_image()
            pred_cube_meshes = [p_info.pred_cubes[j].get_cubes().__getitem__(0).detach() for j in range(box_size)]
            img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, p_info.K, pred_cube_meshes, text=pred_box_classes_names, blend_weight=0.5, blend_weight_overlay=0.85,scale = prop_img.shape[0],colors=colors)
            vis_img_3d = img_3DPR.astype(np.uint8)
            vis_img_3d = show_mask2(p_info.mask_per_image.cpu().numpy(), vis_img_3d, random_color=colors) # NOTE Uncomment to add segmentation mask to pred image
            #vis_img_3d = np.concatenate((vis_img_3d, np.zeros((vis_img_3d.shape[0],vis_img_3d.shape[1],1))), axis=-1)
            ax.set_title('Predicted')
            # expand_img_novel to have alpha channel
            img_novel = np.concatenate((img_novel, np.ones_like(img_novel[:,:,0:1])*255), axis=-1)/255
            ax.imshow(np.concatenate((vis_img_3d, img_novel), axis=1))
            box_size = len(p_info.gt_cube_meshes)
            gt_box_classes_names = [util.MetadataCatalog.get('omni3d_model').thing_classes[i] for i in p_info.gt_box_classes]
            img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, p_info.K, p_info.gt_cube_meshes,text=gt_box_classes_names, blend_weight=0.5, blend_weight_overlay=0.85,scale = prop_img.shape[0],colors=colors)
            vis_img_3d = img_3DPR.astype(np.uint8)
            im_concat = np.concatenate((vis_img_3d, img_novel), axis=1)
            ax1.set_title('GT')
            ax1.imshow(im_concat)
            f_name = os.path.join('ProposalNetwork/output/MABO/vis/', f'vis_{i}.png')
            plt.savefig(f_name, dpi=300, bbox_inches='tight')
            plt.close()

            # with open(f'ProposalNetwork/output/MABO/vis/out_{i}.pkl', 'wb') as f:
            #     out = images_raw.permute(1,2,0).cpu().numpy(), K, p_info.mask_per_image.cpu().numpy(), p_info.gt_boxes3D, p_info.gt_boxes[0], pred_box_classes_names
            #     # im, K, mask, gt_boxes3D, gt_boxes, pred_box_classes_names
            #     pickle.dump(out, f)




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
        data_mapper = DatasetMapper3D(cfg, is_train=False, mode='get_depth_maps')
        data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=4)

        experiment_type = {}

        if cfg.PLOT.EVAL == 'MABO': experiment_type['output_recall_scores'] = True
        else: experiment_type['output_recall_scores'] = False
        # either use pred_boxes or GT boxes
        if cfg.PLOT.MODE2D == 'PRED': experiment_type['use_pred_boxes'] = True
        else: experiment_type['use_pred_boxes'] = False
        if cfg.PLOT.SCORING_FUNC == False:
            experiment_type['scoring_func'] = False
        # define proposal function to use
        if experiment_type['output_recall_scores']:
            _ = mean_average_best_overlap(model, data_loader, segmentor, experiment_type, cfg.PLOT.PROPOSAL_FUNC)
        elif not cfg.PLOT.SCORING_FUNC:
            _ = percent_of_boxes(model, data_loader, segmentor, experiment_type, cfg.PLOT.PROPOSAL_FUNC)
        else:
            results_json = inference_on_dataset(model, data_loader, segmentor, experiment_type, cfg.PLOT.PROPOSAL_FUNC)

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

        
    if cfg.PLOT.EVAL == 'AP':
        '''
        Summarize each Omni3D Evaluation metric
        '''  
        eval_helper.summarize_all()

def do_train(cfg, model):
    """
    Run model on the data_loader. 
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in train mode.

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
    segmentor = init_segmentation(device=cfg.MODEL.DEVICE)

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

    # we need the dataset mapper to get 
    dataset_names = cfg.DATASETS.TRAIN
    data_mapper = DatasetMapper3D(cfg, is_train=False, mode='get_depth_maps')
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats
    assert cfg.TRAIN.pseudo_gt in ['learn', 'pseudo'], "control what kind of proposal should be saved by setting TRAIN.pseudo_gt to either 'learn' or 'pseudo'"
    experiment_type = {}
    experiment_type['use_pred_boxes'] = cfg.PLOT.MODE2D if cfg.PLOT.MODE2D != '' else False
    experiment_type['pseudo_gt'] = cfg.TRAIN.pseudo_gt
    os.makedirs(f'datasets/proposals_{cfg.TRAIN.pseudo_gt}',exist_ok=True)
    # this controls the flow of the program in the model class
    model.train()
    for dataset_name in dataset_names:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper, num_workers=4)

        total = len(data_loader)  # inference data loader must have a fixed length

        for idx, inputs in tqdm(enumerate(data_loader), desc="Generating pseudo GT", total=total):
            cubes = model(inputs, segmentor, experiment_type)
            input_ = inputs[0]
            img_id = input_['image_id']
            torch.save(cubes.to('cpu'), f'datasets/proposals/proposals_{cfg.TRAIN.pseudo_gt}/{img_id}.pt')

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

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")
    
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, filter_empty=True)
    
    dataset_names_test = cfg.DATASETS.TEST

    # filter_ = True if cfg.PLOT.EVAL == 'MABO' else False
    for dataset_name in dataset_names_test:
        if not(dataset_name in cfg.DATASETS.TRAIN):
            # empties should be filtering in test normally
            simple_register(dataset_name, filter_settings, filter_empty=True)
    
    return cfg


def main(args):
    
    cfg = setup(args)

    if args.eval_only:
        assert cfg.PLOT.MODE2D in ['GT', 'PRED'], 'MODE2D must be either GT or PRED'
        assert cfg.PLOT.EVAL in ['AP', 'MABO', 'IoU3D'], 'EVAL must be either AP or MABO'
        if cfg.PLOT.EVAL == 'MABO':
            assert cfg.PLOT.MODE2D == 'GT', 'MABO only works with GT boxes'
    
    name = f'cube {datetime.datetime.now().isoformat()}'
    # wandb.init(project="cube", sync_tensorboard=True, name=name, config=cfg)

    priors = None
    with open('tools/priors.pkl', 'rb') as f:
        priors, _ = pickle.load(f)

    category_path = 'output/Baseline_sgd/category_meta.json'
    # category_path = os.path.join(util.file_parts(args.opts[1])[0], 'category_meta.json')
    
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


    if args.eval_only:
        # skip straight to eval mode
        # load the saved model if using eval boxes
        if cfg.PLOT.MODE2D == 'PRED':
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=False)
        return do_test(cfg, model)
    else:
        logger.info('Making pseudo GT')
        return do_train(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    main(args)