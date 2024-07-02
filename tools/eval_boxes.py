import json
import warnings

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

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

from matplotlib.patches import PathPatch
from matplotlib.path import Path

def create_striped_patch(ax, x_start, x_end, color, alpha=0.3):
    ylim = ax.get_ylim()
    stripe_height = (ylim[1] - ylim[0]) / 20  # Adjust stripe height as needed
    vertices = []
    codes = []
    for y in np.arange(ylim[0], ylim[1], stripe_height * 2):
        vertices.extend([(x_start, y), (x_end, y + stripe_height), (x_end, y + stripe_height * 2), (x_start, y + stripe_height)])
        codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha, hatch='/')
    ax.add_patch(patch)

color_palette = ['#008dff','#ff73bf','#c701ff','#4ecb8d','#ff9d3a','#f0c571','#384860','#d83034']

def inference_on_dataset(model, data_loader, experiment_type, proposal_function):
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
            outputs = model(inputs, experiment_type, proposal_function)
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

def percent_of_boxes(model, data_loader, experiment_type, proposal_functions):
    '''make the detection that have a certain 3D IoU score plots
    if you give the proposal function as input to argparser as:
    
    `PLOT.PROPOSAL_FUNC, ['random', 'z', 'xy', 'dim', 'rotation', 'aspect' ,'full']`
    
    it will work
    '''
    total = len(data_loader)  # inference data loader must have a fixed length
    torch.set_float32_matmul_precision('high')

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        # if not os.path.exists('ProposalNetwork/output/outputs.pkl'):
        if True:
            outputs = []
            for i, inputs in tqdm(enumerate(data_loader), desc=f"IoU3D plots, proposal method: {proposal_functions}", total=total):
                output = model(inputs, experiment_type, proposal_functions)
                outputs.append(output.cpu().numpy())
        #     with open('ProposalNetwork/output/outputs.pkl', 'wb') as f:
        #         pickle.dump(outputs, f)
        # else:
        #     with open('ProposalNetwork/output/outputs_10k.pkl', 'rb') as f:
        #         outputs = pickle.load(f)
        xlim = [0.2,1]
        IoUat = [0.25, 0.4, 0.6]
        n_proposals = outputs[0].shape[-1]
        
        fig, axes = plt.subplots(1, figsize=(7.5,5))
        fig2, axes2 = plt.subplots(1, len(IoUat), figsize=(20,5))
        axes.set_ylabel('Detection rate')
        axes.set_ylim([0,1])
        axes.grid(True)
        axes.set_xlabel('3D Intersection over Union')
        axes.set_xlim(xlim)
        axes.set_title(f'Varying proposal method, {n_proposals} proposals')
        for Iou, ax in zip(IoUat, axes2):
            ax.set_ylabel('Detection rate')
            ax.set_ylim([0,1])
            ax.grid(True)
            ax.set_title(f'Variants, IoU3D = {Iou}')
            ax.set_xlim([1,n_proposals])
            ax.set_xlabel('Number of Proposals')

        for k, proposal_function in enumerate(proposal_functions):
            IoU3Ds = np.concatenate([x[:,k,:] for x in outputs])
            maxIOU_per_instance = np.max(IoU3Ds, axis=1)
            np.random.shuffle(IoU3Ds.T) #transpose to shuffle along the proposal axis
            # detection rate vs. IoU3D
            thresholds = np.arange(xlim[0],xlim[1],0.025)
            detection_rate = np.zeros(len(thresholds))
            for i in range(len(thresholds)):
                detection_rate[i] = np.mean(maxIOU_per_instance > thresholds[i],axis=0)

            # detection rate vs. no. of proposals
            detection_rates = np.zeros((len(IoUat), IoU3Ds.shape[1]))
            for i, IoU in enumerate(IoUat):
                mask_positive_values = IoU3Ds >= IoU
                first_above_threshold = np.argmax(mask_positive_values, axis=1)
                any_above_threshold = mask_positive_values.any(axis=1)
                detection_rate_at_IoU = np.zeros_like(IoU3Ds)

                for j in range(IoU3Ds.shape[0]):
                    if any_above_threshold[j]:
                        detection_rate_at_IoU[j, :first_above_threshold[j]] = 0
                        detection_rate_at_IoU[j, first_above_threshold[j]:] = 1
                
                detection_rates[i] = np.mean(detection_rate_at_IoU, axis=0)

            axes.plot(thresholds, detection_rate, label=f'{proposal_function}', color=color_palette[k])
            for j, ax in enumerate(axes2):
                ax.semilogx(list(range(1, n_proposals+1)), detection_rates[j], label=f'{proposal_function}', color=color_palette[k])
    for ax in axes2:
        ax.legend()
    axes.legend()
    print("saved to 'ProposalNetwork/output/'")
    fig.savefig(f'ProposalNetwork/output/detection_rate_{n_proposals}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'ProposalNetwork/output/IoU_varying_{n_proposals}.png', dpi=300, bbox_inches='tight')


    return
            

def mean_average_best_overlap(model, data_loader, experiment_type, proposal_function):
        
    total = len(data_loader)  # inference data loader must have a fixed length
    os.makedirs('output/pkl_files', exist_ok=True)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        outputs = []
        for i, inputs in tqdm(enumerate(data_loader), desc="Mean average best overlap plots", total=total):
            logger.info('iteration %s',i)
            output = model(inputs, experiment_type, proposal_function)
            # p_info, IoU3D, score_IoU2D, score_seg, score_dim, score_combined, score_random, score_point_cloud, stat_empty_boxes, stats_im, stats_off
            if output is not None:
                outputs.append(output)
        with open('output/pkl_files/outputs_'+str(proposal_function)+'.pkl', 'wb') as f:
            pickle.dump(outputs, f)

        # Create output folder
        if not os.path.exists('ProposalNetwork/output/MABO_'+str(proposal_function)):
            os.makedirs('ProposalNetwork/output/MABO_'+str(proposal_function)) # This is maybe unnecessary
            os.makedirs('ProposalNetwork/output/MABO_'+str(proposal_function)+'/vis/')

        # mean over all the outputs
        Iou2D             = np.concatenate([np.array(sublist) for sublist in (x[1] for x in outputs)])
        score_seg         = np.concatenate([np.array(sublist) for sublist in (x[2] for x in outputs)])
        score_dim         = np.concatenate([np.array(sublist) for sublist in (x[3] for x in outputs)])
        score_combined    = np.concatenate([np.array(sublist) for sublist in (x[4] for x in outputs)])
        score_random      = np.concatenate([np.array(sublist) for sublist in (x[5] for x in outputs)])
        score_point_cloud = np.concatenate([np.array(sublist) for sublist in (x[6] for x in outputs)])
        score_seg_mod     = np.concatenate([np.array(sublist) for sublist in (x[10] for x in outputs)])
        score_corners     = np.concatenate([np.array(sublist) for sublist in (x[11] for x in outputs)])
        stat_empty_boxes  = np.array([x[7] for x in outputs])
        combinations      = np.mean(np.concatenate([np.array(sublist) for sublist in (x[12] for x in outputs)]),axis=0)
        #logger.info('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))
        print('Percentage of cubes with no intersection:',np.mean(stat_empty_boxes))
        logger.info('combination scores:%s',combinations)
        #print('combination scores:',combinations)
        print('best combination is C'+str(np.argmax(combinations)+1))

        

        Iou2D = Iou2D.mean(axis=0)
        score_seg = score_seg.mean(axis=0)
        score_dim = score_dim.mean(axis=0)
        score_combined = score_combined.mean(axis=0)
        score_random = score_random.mean(axis=0)
        score_point_cloud = score_point_cloud.mean(axis=0)
        score_seg_mod = score_seg_mod.mean(axis=0)
        score_corners = score_corners.mean(axis=0)
        total_num_instances = np.sum([x[0].gt_boxes3D.shape[0] for x in outputs])
        
        print('Avg IoU of chosen cube:', score_combined[0])
        print('Best possible IoU:', score_combined[-1])

        x_range = np.arange(1,1001)
        plt.figure(figsize=(8,5))
        plt.plot(x_range,score_combined, linestyle='-',c=color_palette[6], label='combined') 
        plt.plot(x_range,score_dim, linestyle='-',c=color_palette[5],label='dim') 
        plt.plot(x_range,score_seg, linestyle='-',c=color_palette[2],label='segment')
        plt.plot(x_range,Iou2D, linestyle='-',c=color_palette[4],label='2D IoU') 
        plt.plot(x_range,score_corners, linestyle='-',c=color_palette[7],label='corner dist')
        plt.plot(x_range,score_random, linestyle='-',c='grey',label='random') 
        plt.plot(x_range,score_point_cloud, linestyle='-',c=color_palette[3],label='point cloud')
        plt.plot(x_range,score_seg_mod, linestyle='-',c=color_palette[0],label='mod segment')
        plt.grid(True)
        plt.xscale('log')
        plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
        plt.xlim(left=1, right=len(Iou2D))
        plt.xlabel('Number of Proposals')
        plt.ylabel('3D IoU')
        plt.legend()
        plt.title('Average Best Overlap vs Number of Proposals ({} images, {} instances)'.format(1+i,total_num_instances))
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'MABO_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name)
        print('saved to ', f_name)
        
        # Statistics
        stats = torch.cat([x[8] for x in outputs],dim=0)
        num_bins = 40
        titles = ['x','y','z']
        plt.figure(figsize=(15, 5))
        plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
        for i,title in enumerate(titles):
            ax = plt.subplot(1, 3, 1+i)
            plt.hist(stats[:,i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
            plt.axvline(x=0, color='#97a6c4',zorder=2)
            plt.axvline(x=1, color='#97a6c4',zorder=2)
            create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_center_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        print('saved to ', f_name)
        num_bins = 120
        plt.figure(figsize=(15, 5))
        plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
        for i,title in enumerate(titles):
            ax = plt.subplot(1, 3, 1+i)
            plt.hist(stats[:,i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
            plt.axvline(x=0, color='#97a6c4', zorder=2)
            plt.axvline(x=1, color='#97a6c4', zorder=2)
            create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
            plt.xlim([-2,2])
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_center_zoom_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        print('saved to ', f_name)
        num_bins = 40
        titles = ['w','h','l']
        plt.figure(figsize=(15, 5))
        plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
        for i,title in enumerate(titles):
            ax = plt.subplot(1, 3, 1+i)
            plt.hist(stats[:,3+i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
            plt.axvline(x=0, color='#97a6c4', zorder=2)
            plt.axvline(x=1, color='#97a6c4', zorder=2)
            create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_dim_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        print('saved to ', f_name)
        titles = ['rx','ry','rz']
        plt.figure(figsize=(15, 5))
        plt.suptitle("Distribution of Ground Truths in Normalised Searched Range", fontsize=20)
        for i,title in enumerate(titles):
            ax = plt.subplot(1, 3, 1+i)
            plt.hist(stats[:,6+i].numpy(), bins=num_bins, color=color_palette[6],density=True, zorder=2)
            plt.axvline(x=0, color='#97a6c4', zorder=2)
            plt.axvline(x=1, color='#97a6c4', zorder=2)
            create_striped_patch(ax, 0, 1, '#97a6c4', alpha=0.8)
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_rot_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        print('saved to ', f_name)

        stats_off = np.concatenate([np.array(sublist) for sublist in (x[9] for x in outputs)])
        plt.figure(figsize=(15, 15))
        for i,title in enumerate(titles):
            plt.subplot(3, 3, 1+i)
            plt.scatter(stats_off[:,1+i],stats_off[:,0], color=color_palette[6])
            plt.title(title)
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_off_'+str(proposal_function)+'.png')
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.close()
        #logger.info('saved to ', f_name)
        print('saved to ', f_name)

        plt.figure(figsize=(15, 15))
        for i,title in enumerate(titles):
            plt.subplot(3, 3, 1+i)
            plt.scatter(stats_off[:,1+i],stats_off[:,0], color=color_palette[6])
            plt.title(title)
            plt.xlim([0,2])
            plt.ylim([0,1])
        f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function), 'stats_off_zoom_'+str(proposal_function)+'.png')
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
            
            org_img = convert_image_to_rgb(images_raw.permute(1,2,0).cpu().numpy(), 'BGR').copy()
            v_pred = Visualizer(org_img, None)
            v_pred = v_pred.overlay_instances(
                boxes=p_info.pred_boxes[0:box_size].tensor.cpu().numpy()
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
            v_pred = Visualizer(org_img, None)
            v_pred = v_pred.overlay_instances(boxes=p_info.gt_boxes[0:box_size].tensor.cpu().numpy(), assigned_colors=colors)
            prop_img = v_pred.get_image()
            gt_box_classes_names = [util.MetadataCatalog.get('omni3d_model').thing_classes[i] for i in p_info.gt_box_classes]
            img_3DPR, img_novel, _ = vis.draw_scene_view(prop_img, p_info.K, p_info.gt_cube_meshes,text=gt_box_classes_names, blend_weight=0.5, blend_weight_overlay=0.85,scale = prop_img.shape[0],colors=colors)
            vis_img_3d = img_3DPR.astype(np.uint8)
            im_concat = np.concatenate((vis_img_3d, img_novel), axis=1)
            ax1.set_title('GT')
            ax1.imshow(im_concat)
            f_name = os.path.join('ProposalNetwork/output/MABO_'+str(proposal_function)+'/vis/', f'vis_{i}.png')
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
            _ = mean_average_best_overlap(model, data_loader, experiment_type, cfg.PLOT.PROPOSAL_FUNC)
        elif not cfg.PLOT.SCORING_FUNC:
            _ = percent_of_boxes(model, data_loader, experiment_type, cfg.PLOT.PROPOSAL_FUNC)
        else:
            results_json = inference_on_dataset(model, data_loader, experiment_type, cfg.PLOT.PROPOSAL_FUNC)

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

    # lol I think we have to hardcode this part in
    
    dataset_json = {}
    dataset_json.update({"categories": [{"supercategory": "nan", "id": 18, "name": "chair"}, {"supercategory": "nan", "id": 31, "name": "door"}, {"supercategory": "nan", "id": 37, "name": "table"}, {"supercategory": "nan", "id": 26, "name": "shelves"}, {"supercategory": "nan", "id": 51, "name": "kitchen pan"}, {"supercategory": "nan", "id": 52, "name": "bin"}, {"supercategory": "nan", "id": 38, "name": "counter"}, {"supercategory": "nan", "id": 29, "name": "cabinet"}, {"supercategory": "nan", "id": 53, "name": "stove"}, {"supercategory": "nan", "id": 28, "name": "sink"}, {"supercategory": "nan", "id": 14, "name": "books"}, {"supercategory": "nan", "id": 49, "name": "refrigerator"}, {"supercategory": "nan", "id": 54, "name": "microwave"}, {"supercategory": "nan", "id": 15, "name": "bottle"}, {"supercategory": "nan", "id": 55, "name": "plates"}, {"supercategory": "nan", "id": 56, "name": "bowl"}, {"supercategory": "nan", "id": 57, "name": "oven"}, {"supercategory": "nan", "id": 58, "name": "vase"}, {"supercategory": "nan", "id": 59, "name": "faucet"}, {"supercategory": "nan", "id": 22, "name": "towel"}, {"supercategory": "nan", "id": 60, "name": "tissues"}, {"supercategory": "nan", "id": 61, "name": "machine"}, {"supercategory": "nan", "id": 62, "name": "printer"}, {"supercategory": "nan", "id": 33, "name": "desk"}, {"supercategory": "nan", "id": 63, "name": "monitor"}, {"supercategory": "nan", "id": 64, "name": "podium"}, {"supercategory": "nan", "id": 35, "name": "bookcase"}, {"supercategory": "nan", "id": 41, "name": "dresser"}, {"supercategory": "nan", "id": 65, "name": "cart"}, {"supercategory": "nan", "id": 66, "name": "projector"}, {"supercategory": "nan", "id": 67, "name": "electronics"}, {"supercategory": "nan", "id": 68, "name": "computer"}, {"supercategory": "nan", "id": 34, "name": "box"}, {"supercategory": "nan", "id": 36, "name": "picture"}, {"supercategory": "nan", "id": 20, "name": "laptop"}, {"supercategory": "nan", "id": 42, "name": "pillow"}, {"supercategory": "nan", "id": 39, "name": "bed"}, {"supercategory": "nan", "id": 69, "name": "air conditioner"}, {"supercategory": "nan", "id": 25, "name": "lamp"}, {"supercategory": "nan", "id": 40, "name": "night stand"}, {"supercategory": "nan", "id": 50, "name": "board"}, {"supercategory": "nan", "id": 43, "name": "sofa"}, {"supercategory": "nan", "id": 71, "name": "coffee maker"}, {"supercategory": "nan", "id": 72, "name": "toaster"}, {"supercategory": "nan", "id": 73, "name": "potted plant"}, {"supercategory": "nan", "id": 48, "name": "stationery"}, {"supercategory": "nan", "id": 74, "name": "painting"}, {"supercategory": "nan", "id": 75, "name": "bag"}, {"supercategory": "nan", "id": 76, "name": "tray"}, {"supercategory": "nan", "id": 19, "name": "cup"}, {"supercategory": "nan", "id": 70, "name": "drawers"}, {"supercategory": "nan", "id": 77, "name": "keyboard"}, {"supercategory": "nan", "id": 21, "name": "shoes"}, {"supercategory": "vehicle & road", "id": 11, "name": "bicycle"}, {"supercategory": "nan", "id": 78, "name": "blanket"}, {"supercategory": "nan", "id": 44, "name": "television"}, {"supercategory": "nan", "id": 79, "name": "rack"}, {"supercategory": "nan", "id": 27, "name": "mirror"}, {"supercategory": "nan", "id": 47, "name": "clothes"}, {"supercategory": "nan", "id": 80, "name": "phone"}, {"supercategory": "nan", "id": 81, "name": "mouse"}, {"supercategory": "person", "id": 7, "name": "person"}, {"supercategory": "nan", "id": 82, "name": "fire extinguisher"}, {"supercategory": "nan", "id": 83, "name": "toys"}, {"supercategory": "nan", "id": 84, "name": "ladder"}, {"supercategory": "nan", "id": 85, "name": "fan"}, {"supercategory": "nan", "id": 32, "name": "toilet"}, {"supercategory": "nan", "id": 30, "name": "bathtub"}, {"supercategory": "nan", "id": 86, "name": "glass"}, {"supercategory": "nan", "id": 87, "name": "clock"}, {"supercategory": "nan", "id": 88, "name": "toilet paper"}, {"supercategory": "nan", "id": 89, "name": "closet"}, {"supercategory": "nan", "id": 46, "name": "curtain"}, {"supercategory": "nan", "id": 24, "name": "window"}, {"supercategory": "nan", "id": 90, "name": "fume hood"}, {"supercategory": "nan", "id": 91, "name": "utensils"}, {"supercategory": "nan", "id": 45, "name": "floor mat"}, {"supercategory": "nan", "id": 92, "name": "soundsystem"}, {"supercategory": "nan", "id": 93, "name": "fire place"}, {"supercategory": "nan", "id": 94, "name": "shower curtain"}, {"supercategory": "nan", "id": 23, "name": "blinds"}, {"supercategory": "nan", "id": 95, "name": "remote"}, {"supercategory": "nan", "id": 96, "name": "pen"}]})

    d_id_to_contiguous = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
    contiguous_to_id = {v:k for k,v in d_id_to_contiguous.items()}

    global_id = 1
    # this controls the flow of the program in the model class
    model.train()
    for dataset_name in dataset_names:
        idd = 12
        if 'val' in dataset_name:
            idd = 13
        dataset_json.update({"info": {"id": idd, "source": "SUNRGBD", "name": "SUNRGBD Train", "split": "Train", "version": "0.1", "url": "https://rgbd.cs.princeton.edu/"}})
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper, num_workers=4)

        total = len(data_loader)  # inference data loader must have a fixed length

        annotations = []
        images = []
        for idx, inputs in tqdm(enumerate(data_loader), desc="Generating pseudo GT", total=total):
            cubes = model(inputs, experiment_type)
            instances = cubes[0]['instances']
            input_ = inputs[0]
            img_id = input_['image_id']
            input_['instances'].proposal_boxes = input_['instances'].gt_boxes
            bboxes = GeneralizedRCNN._postprocess([input_['instances']], [input_], [input_['instances']._image_size])
            bboxes = bboxes[0]['instances'].proposal_boxes
            # build json for each image
            img = {'width':input_['width'], 'height':input_['height'], 'file_path':input_['file_name'][9:], 'K':input_['K'], 'src_90_rotate':False, 'src_flagged':False, 'incomplete':False, 'id':img_id, 'dataset_id':12}
            for bbox, gt_class, center, dimensions, bbox3D, rotation in zip(bboxes, input_['instances'].gt_classes,
                                                                            instances.pred_center_cam.tolist(), instances.pred_dimensions.tolist(), 
                                                                            instances.pred_bbox3D.tolist(), instances.pred_pose.tolist()):
                c_id = util.MetadataCatalog.get('omni3d_model').thing_classes[gt_class]
                ann = {'behind_camera':False, 'truncation': 0, 'bbox2D_proj':bbox.tolist(), 'bbox2D_tight':-1, 'visibility':1.0, 'segmentation_pts':-1, 'lidar_pts':-1,\
                       'valid3D':True, 'category_id':contiguous_to_id[gt_class.tolist()], 'category_name':c_id, \
                       'id':global_id, 'image_id':img_id, 'dataset_id':idd, 'depth_error':-1, 'center_cam':center,\
                       'dimensions':dimensions, 'bbox3D_cam':bbox3D, 'R_cam':rotation}
                annotations.append(ann)
                global_id += 1

            images.append(img)

        dataset_json.update({'images':images, 'annotations':annotations})

        with open(f'datasets/Omni3D/SUNRGBD_pseudo_gt_{dataset_name}.json', 'w') as f:
            json.dump(dataset_json, f)

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
        assert cfg.PLOT.EVAL in ['AP', 'MABO', 'IoU3D'], 'EVAL must be either AP, MABO or IoU3D'
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