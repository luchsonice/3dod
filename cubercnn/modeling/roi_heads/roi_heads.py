from detectron2.layers.nms import batched_nms
import pyransac3d as pyrsc
#import open3d as o3d
import copy
from segment_anything.utils.transforms import ResizeLongestSide


from dataclasses import dataclass
import logging

import numpy as np

from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.transforms.so3 import (
    so3_relative_angle
)
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes, pairwise_iou, pairwise_ioa
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import (
    StandardROIHeads, ROI_HEADS_REGISTRY, select_foreground_proposals,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ROIAlign
import ProposalNetwork.proposals.proposals as proposals
from ProposalNetwork.scoring.scorefunction import score_dimensions, score_iou, score_point_cloud, score_segmentation
from ProposalNetwork.utils.conversions import cube_to_box, cubes_to_box
from ProposalNetwork.utils.spaces import Cubes
from ProposalNetwork.utils.utils import iou_3d
from cubercnn.modeling.roi_heads.cube_head import build_cube_head
from cubercnn.modeling.proposal_generator.rpn import subsample_labels
from cubercnn.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from cubercnn import util

logger = logging.getLogger(__name__)

E_CONSTANT = 2.71828183
SQRT_2_CONSTANT = 1.41421356

@dataclass
class Plotinfo:
    '''simple dataclass to store plot information access as Plotinfo.x
    fields: pred_cubes, gt_cube_meshes, gt_boxes3D, gt_boxes, gt_box_classes, mask_per_image, K'''
    pred_cubes: List[Cubes]
    gt_cube_meshes: List
    gt_boxes3D: List
    gt_boxes: List
    gt_box_classes: List
    mask_per_image: List
    K: np.array

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def build_roi_heads(cfg, input_shape, priors=None):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape, priors=priors)

def build_roi_heads_score(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

@ROI_HEADS_REGISTRY.register()
class ROIHeads_Boxer(StandardROIHeads):
    '''The 3D box prediction head.'''

    @configurable
    def __init__(self, *, 
                 dims_priors_enabled = None, priors=None, 
                 **kwargs, ):
        super().__init__(**kwargs)

        # misc
        self.dims_priors_enabled = dims_priors_enabled
        # the dimensions could rely on pre-computed priors
        if self.dims_priors_enabled and priors is not None:
            self.priors_dims_per_cat = nn.Parameter(torch.FloatTensor(priors['priors_dims_per_cat']).unsqueeze(0))
        else:
            self.priors_dims_per_cat = nn.Parameter(torch.ones(1, self.num_classes, 2, 3))

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], priors=None):
        
        ret = super().from_config(cfg, input_shape)
        
        # pass along priors
        ret["box_predictor"] = FastRCNNOutputs(cfg, ret['box_head'].output_shape)
        ret.update(cls._init_cube_head(cfg, input_shape))
        ret["priors"] = priors

        return ret
    
    @classmethod
    def _init_cube_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
    
        return {'dims_priors_enabled': cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED,}

    def forward(self, images, images_raw, depth_maps, ground_maps, features, proposals, Ks, im_scales_ratio, segmentor, experiment_type, proposal_function, targets=None):
        # proposals are GT here
        im_dims = [image.shape[1:] for image in images]

        if self.training:
            masks = self.object_masks(images_raw.tensor, proposals, segmentor, {'use_pred_boxes': False})
            experiment_type['use_pred_boxes'] = False
            results = self._forward_cube(images, images_raw, masks, depth_maps, ground_maps, features, proposals, Ks, im_dims, im_scales_ratio, experiment_type, proposal_function)
            return results
        
        else:

            # when oracle is available, by pass the box forward.
            # simulate the predicted instances by creating a new 
            # instance for each passed in image.
            if isinstance(proposals, list) and ~np.any([isinstance(p, Instances) for p in proposals]):
                pred_instances = []
                for proposal, im_dim in zip(proposals, im_dims):
                    
                    pred_instances_i = Instances(im_dim)
                    pred_instances_i.pred_boxes = Boxes(proposal['gt_bbox2D'])
                    pred_instances_i.pred_classes =  proposal['gt_classes']
                    pred_instances_i.scores = torch.ones_like(proposal['gt_classes']).float()
                    pred_instances.append(pred_instances_i)
            
            if experiment_type['use_pred_boxes']:
                pred_instances = self._forward_box(features, proposals)
                # Do we only want proposals with a logit > 0, this corresponds to points with a score > 0.5 ???
                # as a logit of 0 indicates that the odds of the event occurring are equal to the odds of the event not occurring
                # https://deepai.org/machine-learning-glossary-and-terms/logit
                # we can utilise the fact that the objectness_logits are sorted
                def filter_proposals(pred_instances, score_threshold=0.2):
                    for instance in pred_instances:
                        for i, score in enumerate(instance.scores):
                            if score < score_threshold: 
                                pred_boxes = instance.pred_boxes[:i]
                                scores = instance.scores[:i]
                                scores_full = instance.scores_full[:i]
                                pred_classes = instance.pred_classes[:i]

                                instance.remove('pred_boxes'); instance.remove('scores'); instance.remove('scores_full'); instance.remove('pred_classes')
                                instance.pred_boxes = pred_boxes; instance.scores = scores; instance.scores_full = scores_full; instance.pred_classes = pred_classes
                                break
                    return pred_instances

                #pred_instances = filter_proposals(pred_instances)

                ## NMS
                filtered_pred_instances = []
                for instances_i in pred_instances:
                    max_vis_prop = min(len(instances_i), 20)

                    # perform a simple NMS, which is not cls dependent. 
                    keep = batched_nms(
                        instances_i.pred_boxes.tensor, 
                        instances_i.scores, 
                        torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device), 
                        0.9)
                    
                    keep = keep[:max_vis_prop]
                    new_instances = Instances(instances_i.image_size)
                    new_instances.pred_boxes = instances_i.pred_boxes[keep]
                    new_instances.scores = instances_i.scores[keep]
                    new_instances.scores_full = instances_i.scores_full[keep]
                    new_instances.pred_classes = instances_i.pred_classes[keep]

                    filtered_pred_instances.append(new_instances)

            # mask for each proposal
            # NOTE: at the the moment the this assumes a batch size of 1, since the test loader has it hardcoded
            target_instances = filtered_pred_instances if experiment_type['use_pred_boxes'] else proposals
            if experiment_type['use_pred_boxes']:
                if len(target_instances[0].pred_boxes) == 0:
                    return target_instances
            masks = self.object_masks(images_raw.tensor, target_instances, segmentor, experiment_type) # over all images in batch
            pred_instances = self._forward_cube(images, images_raw, masks, depth_maps, ground_maps, features, target_instances, Ks, im_dims, im_scales_ratio, experiment_type, proposal_function)
            return pred_instances
        
    def object_masks(self, images, instances, segmentor, ex):
        '''list of masks for each object in the image.
        Returns
        ------
        mask_per_image: List of torch.Tensor of shape (N_instance, 1, H, W)
        '''
        org_shape = images.shape[-2:]
        resize_transform = ResizeLongestSide(segmentor.image_encoder.img_size)
        batched_input = []
        images = resize_transform.apply_image_torch(images*1.0)# .permute(2, 0, 1).contiguous()
        for image, instance in zip(images, instances):
            if ex['use_pred_boxes']:
                boxes = instance.pred_boxes.tensor
            else:
                boxes = instance.gt_boxes.tensor
            transformed_boxes = resize_transform.apply_boxes_torch(boxes, org_shape) # Bx4
            batched_input.append({'image': image, 'boxes': transformed_boxes, 'original_size':org_shape})

        seg_out = segmentor(batched_input, multimask_output=False)

        mask_per_image = [i['masks'] for i in seg_out]
        return mask_per_image

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals, 
            )
            pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                predictions, proposals
            )
            for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                proposals_per_image.pred_boxes = Boxes(pred_boxes_per_image)

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, )
            return pred_instances
        
    def accumulate_scores(self, scores, IoU3D):
        idx = np.argsort(scores)[::-1]
        scores = np.array([IoU3D[i] for i in idx])
        scores = np.maximum.accumulate(scores)
        return scores

    def predict_cubes(self, gt_boxes, priors, depth_maps, im_shape, K, number_of_proposals, proposal_function, normal_vec, gt_3d=None):
        '''wrap propose'''
        reference_box = gt_boxes
        if proposal_function == 'random':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_random(reference_box, None, None, None, None, number_of_proposals=number_of_proposals)
        elif proposal_function == 'z':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_z(reference_box, depth_maps.tensor.squeeze(), None, None, None, number_of_proposals=number_of_proposals)
        elif proposal_function == 'xy':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_xy_patch(reference_box, depth_maps.tensor.squeeze(), None, im_shape, None, number_of_proposals=number_of_proposals)
        elif proposal_function == 'dim':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_random_dim(reference_box, depth_maps.tensor.squeeze(), None, None, K, number_of_proposals=number_of_proposals)
        elif proposal_function == 'rotation':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_random_rotation(reference_box, depth_maps.tensor.squeeze(), priors, None, K, number_of_proposals=number_of_proposals)
        elif proposal_function == 'aspect':
            pred_cubes, stats_instance, stats_ranges = proposals.propose_aspect_ratio(reference_box, depth_maps.tensor.squeeze(), priors, None, K, number_of_proposals=number_of_proposals)
        else:
            pred_cubes, stats_instance, stats_ranges = proposals.propose(reference_box, depth_maps.tensor.squeeze(), priors, im_shape, K, number_of_proposals=number_of_proposals, gt_cube=gt_3d, ground_normal=normal_vec)
        
        pred_boxes = cubes_to_box(pred_cubes, K)
        return pred_cubes, pred_boxes, stats_instance, stats_ranges
    
    def _forward_cube(self, images, images_raw, mask_per_image, depth_maps, ground_maps, features, instances, Ks, im_current_dims, im_scales_ratio, experiment_type, proposal_function):

        # send all objects to cpu
        # images_raw = images_raw.to('cpu')
        # mask_per_image = [mask.to('cpu') for mask in mask_per_image]
        # depth_maps = depth_maps.to('cpu')
        # if ground_maps is not None:
        #     ground_maps = ground_maps.to('cpu')
        # Ks = [K.to('cpu') for K in Ks]
        # instances = [instance.to('cpu') for instance in instances]


        if experiment_type['use_pred_boxes']:
            gt_box_classes = (torch.cat([p.pred_classes for p in instances], dim=0) if len(instances) else torch.empty(0))
            gt_boxes = torch.cat([p.pred_boxes for p in instances], dim=0,) if len(instances) > 1 else instances[0].pred_boxes
        else:
            gt_box_classes = (torch.cat([p.gt_classes for p in instances], dim=0) if len(instances) else torch.empty(0))
            gt_boxes3D = torch.cat([p.gt_boxes3D for p in instances], dim=0,) if len(instances) else torch.empty(0)
            gt_boxes = torch.cat([p.gt_boxes for p in instances], dim=0,) if len(instances) > 1 else instances[0].gt_boxes
            gt_poses = torch.cat([p.gt_poses for p in instances], dim=0,)

        n_gt = len(gt_boxes)
        
        # nothing to do..
        if n_gt == 0:
            return instances if not self.training else (instances, {})
        
        Ks_scaled_per_box = (Ks[0]/im_scales_ratio[0]).to(images.device)
        Ks_scaled_per_box[-1, -1] = 1

        if self.dims_priors_enabled:
            # gather prior dimensions
            prior_dims = self.priors_dims_per_cat.detach()#.to('cpu')
            prior_dims = prior_dims[:, gt_box_classes, :, :].squeeze(0)
            prior_dims_mean = prior_dims[:, 0, :]
            prior_dims_std = prior_dims[:, 1, :]
        
        # ### point cloud
        use_nth = 5
        dp_map = depth_maps.tensor.cpu().squeeze()[::use_nth,::use_nth]
        focal_length_x, focal_length_y = dp_map.shape[1] / 2, dp_map.shape[0] / 2
        FINAL_WIDTH, FINAL_HEIGHT = dp_map.shape[1], dp_map.shape[0]
        x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        x = (x - FINAL_WIDTH / 2) / focal_length_x
        y = (y - FINAL_HEIGHT / 2) / focal_length_y
        z = np.array(dp_map)

        if ground_maps is not None:
        # select only the points in x,y,z that are part of the ground map
            ground = ground_maps.tensor.squeeze().cpu()[::use_nth,::use_nth]
            zg = z[ground > 0]
            xg = x[ground > 0]
            yg = y[ground > 0]
            # im = images_raw.tensor[0].permute(1,2,0)[::use_nth,::use_nth].cpu().numpy()[ground > 0]
            z_no_g = z[ground == 0]
            x_no_g = x[ground == 0]
            y_no_g = y[ground == 0]
        else:
            zg = z; xg = x; yg = y

        # normalise the points
        points = np.stack((np.multiply(xg, zg/2), np.multiply(yg, zg/2), zg), axis=-1).reshape(-1, 3)
        # colors = im.reshape(-1, 3) / 255.0
        #colors = np.array(images_raw.tensor[0].permute(1,2,0)[::use_nth,::use_nth]).reshape(-1, 3) / 255.0
        plane = pyrsc.Plane()
        # best_eq is the ground plane as a,b,c,d in the equation ax + by + cz + d = 0
        best_eq, best_inliers = plane.fit(points, thresh=0.05, maxIteration=1000)
        normal_vec = np.array(best_eq[:-1])

        # remove ground plane from the points that are fed to the scoring function
        points_all = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        if ground_maps is not None:
            points_no_ground = np.stack((np.multiply(x_no_g, z_no_g), np.multiply(y_no_g, z_no_g), z_no_g), axis=-1).reshape(-1, 3)
        else:
            points_no_ground = points_all

        #normal_vec = np.array([normal_vec[1], normal_vec[0], normal_vec[2]])
        x_up = np.array([1,0,0])
        y_up = np.array([0,1,0])
        z_up = np.array([0,0,1])
        # make sure normal vector is consistent with y-up
        if abs(normal_vec @ z_up) > abs(normal_vec @ y_up):
            # this means the plane has been found as the back wall
            # to rectify this we can turn the vector 90 degrees around the local x-axis
            # note that this assumes that the walls are perpendicular to the floor
            normal_vec = np.array([normal_vec[0], normal_vec[2], -normal_vec[1]])
        if abs(normal_vec @ x_up) > abs(normal_vec @ y_up):
            # this means the plane has been found as the side wall
            # to rectify this we can turn the vector 90 degrees around the local y-axis
            # note that this assumes that the walls are perpendicular to the floor
            normal_vec = np.array([-normal_vec[2], normal_vec[0], normal_vec[1]])
        if normal_vec @ y_up < 0:
            normal_vec *= -1

        normal_vec = torch.from_numpy(normal_vec).to(images_raw.device)

        number_of_proposals = 1000
        mask_per_image = mask_per_image[0] # this should be looped over
        gt_cube_meshes = []
        im_shape = images_raw.tensor.shape[2:][::-1] # im shape should be (x,y)
        sum_percentage_empty_boxes = 0
        score_IoU2D    = np.zeros((n_gt, number_of_proposals))
        score_seg      = np.zeros((n_gt, number_of_proposals))
        score_dim      = np.zeros((n_gt, number_of_proposals))
        score_combined = np.zeros((n_gt, number_of_proposals))
        score_random   = np.zeros((n_gt, number_of_proposals))
        score_point_c  = np.zeros((n_gt, number_of_proposals))
        stats_image    = torch.zeros(n_gt, 9)
        stats_off      = np.zeros((n_gt, 10))
                
        pred_cubes_out = Cubes(torch.zeros(len(gt_boxes), 1, 15, device=images_raw.device), scores=torch.zeros(len(gt_boxes), 1, device=images_raw.device),labels=gt_box_classes)
        
        if self.training: # generate and save all proposals
            assert not experiment_type['use_pred_boxes'], 'must use GT boxes for training'

            pred_cubes, many_pred_boxes, _, _ = self.predict_cubes(gt_boxes, (prior_dims_mean, prior_dims_std), depth_maps, im_shape, Ks_scaled_per_box, number_of_proposals, proposal_function, normal_vec)
            pred_cubes.scores = torch.zeros(pred_cubes.tensor.shape[:-1], device=pred_cubes.tensor.device)
            for i, (gt_box, pred_boxes) in enumerate(zip(gt_boxes, many_pred_boxes)):
                IoU2D_scores = score_iou(Boxes(gt_box.unsqueeze(0)), pred_boxes)
                pred_cubes.scores[i] = IoU2D_scores

            if experiment_type['pseudo_gt'] == 'learn':
                return pred_cubes

            elif experiment_type['pseudo_gt'] == 'gt':
                for i, (gt_box) in enumerate(gt_boxes):
                    segment_scores = score_segmentation(mask_per_image[i][0], pred_cubes[i].get_bube_corners(Ks_scaled_per_box))
                    segment_scores = torch.stack(segment_scores)
                    dim_scores = score_dimensions((prior_dims_mean[i], prior_dims_std[i]), pred_cubes[i].dimensions[0])
                    combined_score = IoU2D_scores*dim_scores*segment_scores
                    
                    score_to_use = combined_score
                    highest_score = torch.argmax(score_to_use)
                    pred_cube = pred_cubes[i,highest_score]
                    pred_cubes_out.scores[i] = torch.as_tensor(score_to_use[highest_score])
                    pred_cubes_out.tensor[i] = pred_cube.tensor[0]
                return pred_cubes_out



        if experiment_type['use_pred_boxes']:
            pred_cubes, pred_boxes, _, _ = self.predict_cubes(gt_boxes, (prior_dims_mean, prior_dims_std), depth_maps, im_shape, Ks_scaled_per_box, number_of_proposals, proposal_function, normal_vec)
            for i, (gt_box) in enumerate(gt_boxes):
                IoU2D_scores = score_iou(Boxes(gt_box.unsqueeze(0)), pred_boxes[i])
                segment_scores = score_segmentation(mask_per_image[i][0], pred_cubes[i].get_bube_corners(Ks_scaled_per_box))
                segment_scores = torch.stack(segment_scores)
                dim_scores = score_dimensions((prior_dims_mean[i], prior_dims_std[i]), pred_cubes[i].dimensions[0])
                #point_cloud_scores = score_point_cloud(torch.from_numpy(points_no_ground), pred_cubes[i])
                combined_score = IoU2D_scores*dim_scores*segment_scores
                
                score_to_use = combined_score
                highest_score = torch.argmax(score_to_use)
                pred_cube = pred_cubes[i,highest_score]
                pred_cubes_out.scores[i] = torch.as_tensor(score_to_use[highest_score])
                pred_cubes_out.tensor[i] = pred_cube.tensor[0]
        else:
            assert len(gt_boxes3D) == len(gt_boxes), f"gt_boxes3D and gt_boxes should have the same length. but was {len(gt_boxes3D)} and {len(gt_boxes)} respectively."
            gt_cubes = Cubes(torch.cat((gt_boxes3D[:,6:].unsqueeze(0),gt_boxes3D[:,3:6].unsqueeze(0), gt_poses.view(n_gt,9).unsqueeze(0)),dim=2).permute(1,0,2))
            pred_cubes, pred_boxes, stats_instance, stats_ranges = self.predict_cubes(gt_boxes, (prior_dims_mean, prior_dims_std), depth_maps, im_shape, Ks_scaled_per_box, number_of_proposals, proposal_function, normal_vec, gt_cubes)
            
            for i in range(n_gt):
                # iou
                IoU3D = iou_3d(gt_cubes[i], pred_cubes[i]).cpu().numpy()

                # scoring
                IoU2D_scores = score_iou(cubes_to_box(gt_cubes[i], Ks_scaled_per_box)[0], pred_boxes[i])
                point_cloud_scores = score_point_cloud(torch.from_numpy(points_no_ground).to(pred_cubes.device), pred_cubes[i])
                segment_scores = score_segmentation(mask_per_image[i][0], pred_cubes[i].get_bube_corners(Ks_scaled_per_box))
                dim_scores = score_dimensions((prior_dims_mean[i], prior_dims_std[i]), pred_cubes[i].dimensions[0])
                combined_score = np.array(IoU2D_scores)*np.array(dim_scores)*np.array(segment_scores)
                random_score = np.random.rand(number_of_proposals)
                
                score_IoU2D[i,:] = self.accumulate_scores(IoU2D_scores.cpu().numpy(), IoU3D)
                score_point_c[i,:] = self.accumulate_scores(point_cloud_scores.cpu().numpy(), IoU3D)
                score_seg[i,:] = self.accumulate_scores(segment_scores, IoU3D)
                score_dim[i,:] = self.accumulate_scores(dim_scores.cpu().numpy(), IoU3D)
                score_combined[i,:] = self.accumulate_scores(combined_score, IoU3D)
                score_random[i,:] = self.accumulate_scores(random_score, IoU3D)
                
                score_to_use = combined_score
                highest_score = np.argmax(score_to_use)
                pred_cube = pred_cubes[i,highest_score]
                gt_cube_meshes.append(gt_cubes[i].get_cubes().__getitem__(0).detach())
                pred_cubes_out.scores[i] = torch.as_tensor(score_to_use[highest_score])
                pred_cubes_out.tensor[i] = pred_cube.tensor[0]

                # stats
                sum_percentage_empty_boxes += int(np.count_nonzero(IoU3D == 0.0)/IoU3D.size*100)
                stats_image[i] = stats_instance
                nested_list = [[IoU3D.max()],abs(gt_cubes[i].centers.numpy()-pred_cube.centers.numpy())[0][0]/stats_ranges[:3],abs(gt_cubes[i].dimensions.numpy()-pred_cube.dimensions.numpy())[0][0]/stats_ranges[3:6],abs(util.mat2euler(gt_cubes[i].rotations[0][0])-util.mat2euler(pred_cube.rotations[0][0]))/stats_ranges[6:]]
                stats_off[i] = [item for sublist in nested_list for item in sublist]
            stat_empty_boxes = sum_percentage_empty_boxes/n_gt
            p_info = Plotinfo(pred_cubes_out, gt_cube_meshes, gt_boxes3D, gt_boxes, gt_box_classes, mask_per_image, Ks_scaled_per_box.cpu().numpy())

            
        if experiment_type['output_recall_scores']: # MABO
            return p_info, score_IoU2D, score_seg, score_dim, score_combined, score_random, score_point_c, stat_empty_boxes, stats_image, stats_off
        
        elif not experiment_type['output_recall_scores']: # AP
            # list of Instances with the fields: pred_boxes, scores, pred_classes, pred_bbox3D, pred_center_cam, pred_center_2D, pred_dimensions, pred_pose
            # it is possible to assign multiple element to each Instances object at once.
            # such that the loop can be over the images.
            pred_instances = [Instances(size) for size in images_raw.image_sizes] # each instance object contains all boxes in one image, the list is for each image
            for instances_i in pred_instances:
                instances_i.pred_boxes = Boxes.cat(cubes_to_box(pred_cubes_out, Ks_scaled_per_box.to('cpu')))
                instances_i.scores = pred_cubes_out.scores
                instances_i.pred_classes = pred_cubes_out.labels
                instances_i.pred_bbox3D = pred_cubes_out.get_all_corners().squeeze(1)
                instances_i.pred_center_cam = pred_cubes_out.centers.squeeze(1)
                instances_i.pred_dimensions = pred_cubes_out.dimensions.squeeze(1)
                instances_i.pred_pose = pred_cubes_out.rotations.squeeze(1)
                instances_i.pred_center_2D = instances_i.pred_boxes.get_centers()  

            return pred_instances

@ROI_HEADS_REGISTRY.register()
class ROIHeads_Score(StandardROIHeads):
    '''The score prediction head.'''

    @configurable
    def __init__(self, *, 
                 cube_pooler:nn.Module, mlp:nn.Sequential, 
                 **kwargs, ):
        super().__init__(**kwargs)

        self.cube_pooler = cube_pooler
        self.mlp = mlp

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        
        ret = super().from_config(cfg, input_shape)
        
        # pass along priors
        ret["box_predictor"] = FastRCNNOutputs(cfg, ret['box_head'].output_shape)
        ret.update(cls._init_cube_head(cfg, input_shape))

        return ret
    
    @classmethod
    def _init_cube_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        pooler_scales = (1/32,)#(1.0,)
        pooler_resolution = cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION 
        pooler_sampling_ratio = cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE

        cube_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=pooler_sampling_ratio,
            pooler_type=pooler_type,
        )

        # TODO: make the sizes configurable
        res = pooler_resolution*pooler_resolution*512
        mlp = nn.Sequential(
            nn.Linear(res, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
        return {'cube_pooler': cube_pooler,
                'mlp': mlp}

    def forward(self, pred_cubes, instances, Ks, im_scales_ratio, combined_features):
        if self.training:
            instances_3d, losses = self._forward_cube(pred_cubes, instances, Ks, im_scales_ratio, combined_features)
            return instances_3d, losses
        else:
            pred_instances = self._forward_cube(pred_cubes, instances, Ks, im_scales_ratio, combined_features)
            return pred_instances
        
    def _forward_cube(self, pred_cubes, instances, Ks, im_scales_ratio, combined_features):
        Ks_scaled = (Ks[0]/im_scales_ratio[0]).to(combined_features.device)
        Ks_scaled[-1, -1] = 1
        pred_cubes = pred_cubes[0] # NOTE Batchsize 1?? Or is this part of the code ment to be as if it's 1? next(data_iter) gives image in one by one
        
        if self.training:
            total_num_of_boxes_per_image = 64
            balance = 4
            total_num_of_positive_boxes_per_image = int(total_num_of_boxes_per_image / balance)
            total_num_of_negative_boxes_per_image = int(((balance-1) * total_num_of_boxes_per_image)/balance)
            
            # Choose boxes
            all_scores = pred_cubes.scores

            threshold = 0.5
            positive_mask = all_scores > threshold
            negative_mask = all_scores < threshold

            # Get indices of positive and negative cubes
            positive_indices = torch.nonzero(positive_mask, as_tuple=False)
            negative_indices = torch.nonzero(negative_mask, as_tuple=False)

            # Randomly shuffle the indices
            positive_indices_shuffled = positive_indices[torch.randperm(positive_indices.size(0))]
            negative_indices_shuffled = negative_indices[torch.randperm(negative_indices.size(0))] # TODO check that these are valid permutations (because of 2d)

            # Repeat positive indices if needed to meet the required number
            num_positive_samples = min(total_num_of_positive_boxes_per_image, positive_indices_shuffled.size(0))
            num_repeats = total_num_of_positive_boxes_per_image // num_positive_samples
            remaining_repeats = total_num_of_positive_boxes_per_image % num_positive_samples

            positive_cubes_indices = positive_indices_shuffled[:num_positive_samples].repeat(num_repeats,1)
            if remaining_repeats > 0:
                positive_cubes_indices = torch.cat([positive_cubes_indices, positive_indices_shuffled[:remaining_repeats]], dim=0)

            # Sample negative indices
            negative_cubes_indices = negative_indices_shuffled[:total_num_of_negative_boxes_per_image]

            positive_cubes = pred_cubes.tensor[positive_cubes_indices[:, 0], positive_cubes_indices[:, 1]]
            negative_cubes = pred_cubes.tensor[negative_cubes_indices[:, 0], negative_cubes_indices[:, 1]]

            chosen_cubes = torch.cat([positive_cubes, negative_cubes], dim=0).unsqueeze(1)

            positive_scores = all_scores[positive_cubes_indices[:, 0], positive_cubes_indices[:, 1]]
            negative_scores = all_scores[negative_cubes_indices[:, 0], negative_cubes_indices[:, 1]]
            chosen_cubes_scores = torch.cat([positive_scores, negative_scores], dim=0)
            binary_chosen_cubes_scores = (chosen_cubes_scores > 0.5).float()

            chosen_boxes = Boxes.cat(cubes_to_box(Cubes(chosen_cubes, chosen_cubes_scores.unsqueeze(1)), Ks_scaled))

            cube_features = self.cube_pooler([combined_features], [chosen_boxes]).flatten(1)
            print(cube_features.shape)

            pred_iou2d_scores = self.mlp(cube_features).flatten()

            criterion = nn.BCELoss()
            loss = criterion(pred_iou2d_scores, binary_chosen_cubes_scores)
            #loss = F.mse_loss(pred_iou2d_scores, chosen_cubes_scores, reduction='mean')

            return None, loss
    

@ROI_HEADS_REGISTRY.register()
class ROIHeads3D(StandardROIHeads):

    @configurable
    def __init__(
        self,
        *,
        ignore_thresh: float,
        cube_head: nn.Module,
        cube_pooler: nn.Module,
        loss_w_3d: float,
        loss_w_xy: float,
        loss_w_z: float,
        loss_w_dims: float,
        loss_w_pose: float,
        loss_w_joint: float,
        use_confidence: float,
        inverse_z_weight: bool,
        z_type: str,
        pose_type: str,
        cluster_bins: int,
        priors = None,
        dims_priors_enabled = None,
        dims_priors_func = None,
        disentangled_loss=None,
        virtual_depth=None,
        virtual_focal=None,
        test_scale=None,
        allocentric_pose=None,
        chamfer_pose=None,
        scale_roi_boxes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scale_roi_boxes = scale_roi_boxes

        # rotation settings
        self.allocentric_pose = allocentric_pose
        self.chamfer_pose = chamfer_pose

        # virtual settings
        self.virtual_depth = virtual_depth
        self.virtual_focal = virtual_focal

        # loss weights, <=0 is off
        self.loss_w_3d = loss_w_3d
        self.loss_w_xy = loss_w_xy
        self.loss_w_z = loss_w_z
        self.loss_w_dims = loss_w_dims
        self.loss_w_pose = loss_w_pose
        self.loss_w_joint = loss_w_joint

        # loss modes
        self.disentangled_loss = disentangled_loss
        self.inverse_z_weight = inverse_z_weight

        # misc
        self.test_scale = test_scale
        self.ignore_thresh = ignore_thresh
        
        # related to network outputs
        self.z_type = z_type
        self.pose_type = pose_type
        self.use_confidence = use_confidence

        # related to priors
        self.cluster_bins = cluster_bins
        self.dims_priors_enabled = dims_priors_enabled
        self.dims_priors_func = dims_priors_func

        # if there is no 3D loss, then we don't need any heads. 
        if loss_w_3d > 0:
            
            self.cube_head = cube_head
            self.cube_pooler = cube_pooler
            
            # the dimensions could rely on pre-computed priors
            if self.dims_priors_enabled and priors is not None:
                self.priors_dims_per_cat = nn.Parameter(torch.FloatTensor(priors['priors_dims_per_cat']).unsqueeze(0))
            else:
                self.priors_dims_per_cat = nn.Parameter(torch.ones(1, self.num_classes, 2, 3))

            # Optionally, refactor priors and store them in the network params
            if self.cluster_bins > 1 and priors is not None:

                # the depth could have been clustered based on 2D scales                
                priors_z_scales = torch.stack([torch.FloatTensor(prior[1]) for prior in priors['priors_bins']])
                self.priors_z_scales = nn.Parameter(priors_z_scales)

            else:
                self.priors_z_scales = nn.Parameter(torch.ones(self.num_classes, self.cluster_bins))

            # the depth can be based on priors
            if self.z_type == 'clusters':
                
                assert self.cluster_bins > 1, 'To use z_type of priors, there must be more than 1 cluster bin'
                
                if priors is None:
                    self.priors_z_stats = nn.Parameter(torch.ones(self.num_classes, self.cluster_bins, 2).float())
                else:

                    # stats
                    priors_z_stats = torch.cat([torch.FloatTensor(prior[2]).unsqueeze(0) for prior in priors['priors_bins']])
                    self.priors_z_stats = nn.Parameter(priors_z_stats)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], priors=None):
        
        ret = super().from_config(cfg, input_shape)
        
        # pass along priors
        ret["box_predictor"] = FastRCNNOutputs(cfg, ret['box_head'].output_shape)
        ret.update(cls._init_cube_head(cfg, input_shape))
        ret["priors"] = priors

        return ret

    @classmethod
    def _init_cube_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        pooler_resolution = cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION 
        pooler_sampling_ratio = cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE

        cube_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=pooler_sampling_ratio,
            pooler_type=pooler_type,
        )

        in_channels = [input_shape[f].channels for f in in_features][0]
        shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        cube_head = build_cube_head(cfg, shape)

        return {
            'cube_head': cube_head,
            'cube_pooler': cube_pooler,
            'use_confidence': cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE,
            'inverse_z_weight': cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT,
            'loss_w_3d': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D,
            'loss_w_xy': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY,
            'loss_w_z': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z,
            'loss_w_dims': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS,
            'loss_w_pose': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE,
            'loss_w_joint': cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT,
            'z_type': cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE,
            'pose_type': cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE,
            'dims_priors_enabled': cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED,
            'dims_priors_func': cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC,
            'disentangled_loss': cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS,
            'virtual_depth': cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH,
            'virtual_focal': cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL,
            'test_scale': cfg.INPUT.MIN_SIZE_TEST,
            'chamfer_pose': cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE,
            'allocentric_pose': cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE,
            'cluster_bins': cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS,
            'ignore_thresh': cfg.MODEL.RPN.IGNORE_THRESHOLD,
            'scale_roi_boxes': cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES,
        }


    def forward(self, images, features, proposals, Ks, im_scales_ratio, targets=None):

        im_dims = [image.shape[1:] for image in images]

        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        
        del targets

        if self.training:

            losses = self._forward_box(features, proposals)
            if self.loss_w_3d > 0:
                instances_3d, losses_cube = self._forward_cube(features, proposals, Ks, im_dims, im_scales_ratio)
                losses.update(losses_cube)

            return instances_3d, losses
        
        else:

            # when oracle is available, by pass the box forward.
            # simulate the predicted instances by creating a new 
            # instance for each passed in image.
            if isinstance(proposals, list) and ~np.any([isinstance(p, Instances) for p in proposals]):
                pred_instances = []
                for proposal, im_dim in zip(proposals, im_dims):
                    
                    pred_instances_i = Instances(im_dim)
                    pred_instances_i.pred_boxes = Boxes(proposal['gt_bbox2D'])
                    pred_instances_i.pred_classes =  proposal['gt_classes']
                    pred_instances_i.scores = torch.ones_like(proposal['gt_classes']).float()
                    pred_instances.append(pred_instances_i)
            else:
                pred_instances = self._forward_box(features, proposals)
            
            if self.loss_w_3d > 0:
                pred_instances = self._forward_cube(features, pred_instances, Ks, im_dims, im_scales_ratio)
            return pred_instances, {}
    

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(
                predictions, proposals, 
            )
            pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                predictions, proposals
            )
            for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                proposals_per_image.pred_boxes = Boxes(pred_boxes_per_image)

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, )
            return pred_instances

    def l1_loss(self, vals, target):
        return F.smooth_l1_loss(vals, target, reduction='none', beta=0.0)

    def chamfer_loss(self, vals, target):
        B = vals.shape[0]
        xx = vals.view(B, 8, 1, 3)
        yy = target.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        l1 = (l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1))
        return l1

    # optionally, scale proposals to zoom RoI in (<1.0) our out (>1.0)
    def scale_proposals(self, proposal_boxes):
        if self.scale_roi_boxes > 0:

            proposal_boxes_scaled = []
            for boxes in proposal_boxes:
                centers = boxes.get_centers()
                widths = boxes.tensor[:, 2] - boxes.tensor[:, 0]
                heights = boxes.tensor[:, 2] - boxes.tensor[:, 0]
                x1 = centers[:, 0] - 0.5*widths*self.scale_roi_boxes
                x2 = centers[:, 0] + 0.5*widths*self.scale_roi_boxes
                y1 = centers[:, 1] - 0.5*heights*self.scale_roi_boxes
                y2 = centers[:, 1] + 0.5*heights*self.scale_roi_boxes
                boxes_scaled = Boxes(torch.stack([x1, y1, x2, y2], dim=1))
                proposal_boxes_scaled.append(boxes_scaled)
        else:
            proposal_boxes_scaled = proposal_boxes

        return proposal_boxes_scaled
    
    def _forward_cube(self, features, instances, Ks, im_current_dims, im_scales_ratio):
        
        features = [features[f] for f in self.in_features]

        # training on foreground
        if self.training:

            losses = {}

            # add up the amount we should normalize the losses by. 
            # this follows the same logic as the BoxHead, where each FG proposal 
            # is able to contribute the same amount of supervision. Technically, 
            # this value doesn't change during training unless the batch size is dynamic.
            self.normalize_factor = max(sum([i.gt_classes.numel() for i in instances]), 1.0)

            # The loss is only defined on positive proposals
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            pred_boxes = [x.pred_boxes for x in proposals]

            box_classes = (torch.cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
            gt_boxes3D = torch.cat([p.gt_boxes3D for p in proposals], dim=0,)
            gt_poses = torch.cat([p.gt_poses for p in proposals], dim=0,)

            assert len(gt_poses) == len(gt_boxes3D) == len(box_classes)
        
        # eval on all instances
        else:
            proposals = instances
            pred_boxes = [x.pred_boxes for x in instances]
            proposal_boxes = pred_boxes
            box_classes = torch.cat([x.pred_classes for x in instances])

        proposal_boxes_scaled = self.scale_proposals(proposal_boxes)

        # forward features
        cube_features = self.cube_pooler(features, proposal_boxes_scaled).flatten(1)

        n = cube_features.shape[0]
        
        # nothing to do..
        if n == 0:
            return instances if not self.training else (instances, {})

        num_boxes_per_image = [len(i) for i in proposals]

        # scale the intrinsics according to the ratio the image has been scaled. 
        # this means the projections at the current scale are in sync.
        Ks_scaled_per_box = torch.cat([
            (Ks[i]/im_scales_ratio[i]).unsqueeze(0).repeat([num, 1, 1]) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)
        Ks_scaled_per_box[:, -1, -1] = 1

        focal_lengths_per_box = torch.cat([
            (Ks[i][1, 1]).unsqueeze(0).repeat([num]) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        im_ratios_per_box = torch.cat([
            torch.FloatTensor([im_scales_ratio[i]]).repeat(num) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        # scaling factor for Network resolution -> Original
        im_scales_per_box = torch.cat([
            torch.FloatTensor([im_current_dims[i][0]]).repeat(num) 
            for (i, num) in enumerate(num_boxes_per_image)
        ]).to(cube_features.device)

        im_scales_original_per_box = im_scales_per_box * im_ratios_per_box

        if self.virtual_depth:
                
            virtual_to_real = util.compute_virtual_scale_from_focal_spaces(
                focal_lengths_per_box, im_scales_original_per_box, 
                self.virtual_focal, im_scales_per_box
            )
            real_to_virtual = 1 / virtual_to_real

        else:
            real_to_virtual = virtual_to_real = 1.0

        # 2D boxes are needed to apply deltas
        src_boxes = torch.cat([box_per_im.tensor for box_per_im in proposal_boxes], dim=0)
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_scales = (src_heights**2 + src_widths**2).sqrt()
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        # For some methods, we need the predicted 2D box,
        # e.g., the differentiable tensors from the 2D box head. 
        pred_src_boxes = torch.cat([box_per_im.tensor for box_per_im in pred_boxes], dim=0)
        pred_widths = pred_src_boxes[:, 2] - pred_src_boxes[:, 0]
        pred_heights = pred_src_boxes[:, 3] - pred_src_boxes[:, 1]
        pred_src_x = (pred_src_boxes[:, 2] + pred_src_boxes[:, 0]) * 0.5
        pred_src_y = (pred_src_boxes[:, 3] + pred_src_boxes[:, 1]) * 0.5
        
        # forward predictions
        cube_2d_deltas, cube_z, cube_dims, cube_pose, cube_uncert = self.cube_head(cube_features)
        
        # simple indexing re-used commonly for selection purposes
        fg_inds = torch.arange(n)

        # Z when clusters are used
        if cube_z is not None and self.cluster_bins > 1:
        
            # compute closest bin assignments per batch per category (batch x n_category)
            scales_diff = (self.priors_z_scales.detach().T.unsqueeze(0) - src_scales.unsqueeze(1).unsqueeze(2)).abs()
            
            # assign the correct scale prediction.
            # (the others are not used / thrown away)
            assignments = scales_diff.argmin(1)

            # select FG, category, and correct cluster
            cube_z = cube_z[fg_inds, :, box_classes, :][fg_inds, assignments[fg_inds, box_classes]]

        elif cube_z is not None:

            # if z is available, collect the per-category predictions.
            cube_z = cube_z[fg_inds, box_classes, :]
            
        cube_dims = cube_dims[fg_inds, box_classes, :]
        cube_pose = cube_pose[fg_inds, box_classes, :, :]

        if self.use_confidence:
            
            # if uncertainty is available, collect the per-category predictions.
            cube_uncert = cube_uncert[fg_inds, box_classes]
        
        cube_2d_deltas = cube_2d_deltas[fg_inds, box_classes, :]
        
        # apply our predicted deltas based on src boxes.
        cube_x = src_ctr_x + src_widths * cube_2d_deltas[:, 0]
        cube_y = src_ctr_y + src_heights * cube_2d_deltas[:, 1]
        
        cube_xy = torch.cat((cube_x.unsqueeze(1), cube_y.unsqueeze(1)), dim=1)

        cube_dims_norm = cube_dims
        
        if self.dims_priors_enabled:

            # gather prior dimensions
            prior_dims = self.priors_dims_per_cat.detach().repeat([n, 1, 1, 1])[fg_inds, box_classes]
            prior_dims_mean = prior_dims[:, 0, :]
            prior_dims_std = prior_dims[:, 1, :]

            if self.dims_priors_func == 'sigmoid':
                prior_dims_min = (prior_dims_mean - 3*prior_dims_std).clip(0.0)
                prior_dims_max = (prior_dims_mean + 3*prior_dims_std)
                cube_dims = util.scaled_sigmoid(cube_dims_norm, min=prior_dims_min, max=prior_dims_max)
            elif self.dims_priors_func == 'exp':
                cube_dims = torch.exp(cube_dims_norm.clip(max=5)) * prior_dims_mean

        else:
            # no priors are used
            cube_dims = torch.exp(cube_dims_norm.clip(max=5))
        
        if self.allocentric_pose:
            
            # To compare with GTs, we need the pose to be egocentric, not allocentric
            cube_pose_allocentric = cube_pose
            cube_pose = util.R_from_allocentric(Ks_scaled_per_box, cube_pose, u=cube_x.detach(), v=cube_y.detach())
            
        cube_z = cube_z.squeeze()
        
        if self.z_type =='sigmoid':    
            cube_z_norm = torch.sigmoid(cube_z)
            cube_z = cube_z_norm * 100

        elif self.z_type == 'log':
            cube_z_norm = cube_z
            cube_z = torch.exp(cube_z)

        elif self.z_type == 'clusters':
            
            # gather the mean depth, same operation as above, for a n x c result
            z_means = self.priors_z_stats[:, :, 0].T.unsqueeze(0).repeat([n, 1, 1])
            z_means = torch.gather(z_means, 1, assignments.unsqueeze(1)).squeeze(1)

            # gather the std depth, same operation as above, for a n x c result
            z_stds = self.priors_z_stats[:, :, 1].T.unsqueeze(0).repeat([n, 1, 1])
            z_stds = torch.gather(z_stds, 1, assignments.unsqueeze(1)).squeeze(1)

            # do not learn these, they are static
            z_means = z_means.detach()
            z_stds = z_stds.detach()

            z_means = z_means[fg_inds, box_classes]
            z_stds = z_stds[fg_inds, box_classes]

            z_mins = (z_means - 3*z_stds).clip(0)
            z_maxs = (z_means + 3*z_stds)

            cube_z_norm = cube_z
            cube_z = util.scaled_sigmoid(cube_z, min=z_mins, max=z_maxs)

        if self.virtual_depth:
            cube_z = (cube_z * virtual_to_real)

        if self.training:

            prefix = 'Cube/'
            storage = get_event_storage()

            # Pull off necessary GT information
            # let lowercase->2D and uppercase->3D
            # [x, y, Z, W, H, L] 
            gt_2d = gt_boxes3D[:, :2]
            gt_z = gt_boxes3D[:, 2]
            gt_dims = gt_boxes3D[:, 3:6]

            # this box may have been mirrored and scaled so
            # we need to recompute XYZ in 3D by backprojecting.
            gt_x3d = gt_z * (gt_2d[:, 0] - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
            gt_y3d = gt_z * (gt_2d[:, 1] - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
            gt_3d = torch.stack((gt_x3d, gt_y3d, gt_z)).T

            # put together the GT boxes
            gt_box3d = torch.cat((gt_3d, gt_dims), dim=1)

            # These are the corners which will be the target for all losses!!
            gt_corners = util.get_cuboid_verts_faces(gt_box3d, gt_poses)[0]

            # project GT corners
            gt_proj_boxes = torch.bmm(Ks_scaled_per_box, gt_corners.transpose(1,2))
            gt_proj_boxes /= gt_proj_boxes[:, -1, :].clone().unsqueeze(1)

            gt_proj_x1 = gt_proj_boxes[:, 0, :].min(1)[0]
            gt_proj_y1 = gt_proj_boxes[:, 1, :].min(1)[0]
            gt_proj_x2 = gt_proj_boxes[:, 0, :].max(1)[0]
            gt_proj_y2 = gt_proj_boxes[:, 1, :].max(1)[0]

            gt_widths = gt_proj_x2 - gt_proj_x1
            gt_heights = gt_proj_y2 - gt_proj_y1
            gt_x = gt_proj_x1 + 0.5 * gt_widths
            gt_y = gt_proj_y1 + 0.5 * gt_heights

            gt_proj_boxes = torch.stack((gt_proj_x1, gt_proj_y1, gt_proj_x2, gt_proj_y2), dim=1)
            
            if self.disentangled_loss:
                '''
                Disentangled loss compares each varaible group to the 
                cuboid corners, which is generally more robust to hyperparams.
                '''
                    
                # compute disentangled Z corners
                cube_dis_x3d_from_z = cube_z * (gt_2d[:, 0] - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d_from_z = cube_z * (gt_2d[:, 1] - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_z = torch.cat((torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z)).T, gt_dims), dim=1)
                dis_z_corners = util.get_cuboid_verts_faces(cube_dis_z, gt_poses)[0]
                
                # compute disentangled XY corners
                cube_dis_x3d = gt_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d = gt_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_XY = torch.cat((torch.stack((cube_dis_x3d, cube_dis_y3d, gt_z)).T, gt_dims), dim=1)
                dis_XY_corners = util.get_cuboid_verts_faces(cube_dis_XY, gt_poses)[0]
                loss_xy = self.l1_loss(dis_XY_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
                    
                # Pose
                dis_pose_corners = util.get_cuboid_verts_faces(gt_box3d, cube_pose)[0]
                
                # Dims
                dis_dims_corners = util.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses)[0]

                # Loss dims
                loss_dims = self.l1_loss(dis_dims_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)

                # Loss z
                loss_z = self.l1_loss(dis_z_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
    
                # Rotation uses chamfer or l1 like others
                if self.chamfer_pose:
                    loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)

                else:
                    loss_pose = self.l1_loss(dis_pose_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
                
            # Non-disentangled training losses
            else:
                '''
                These loss functions are fairly arbitrarily designed. 
                Generally, they are in some normalized space but there
                are many alternative implementations for most functions.
                '''

                # XY
                gt_deltas = (gt_2d.clone() - torch.cat((src_ctr_x.unsqueeze(1), src_ctr_y.unsqueeze(1)), dim=1)) \
                            / torch.cat((src_widths.unsqueeze(1), src_heights.unsqueeze(1)), dim=1)
                
                loss_xy = self.l1_loss(cube_2d_deltas, gt_deltas).mean(1) 

                # Dims
                if self.dims_priors_enabled:
                    cube_dims_gt_normspace = torch.log(gt_dims/prior_dims)
                    loss_dims = self.l1_loss(cube_dims_norm, cube_dims_gt_normspace).mean(1) 

                else:
                    loss_dims = self.l1_loss(cube_dims_norm, torch.log(gt_dims)).mean(1)
                
                # Pose
                try:
                    if self.allocentric_pose:
                        gt_poses_allocentric = util.R_to_allocentric(Ks_scaled_per_box, gt_poses, u=cube_x.detach(), v=cube_y.detach())
                        loss_pose = 1-so3_relative_angle(cube_pose_allocentric, gt_poses_allocentric, eps=0.1, cos_angle=True)
                    else:
                        loss_pose = 1-so3_relative_angle(cube_pose, gt_poses, eps=0.1, cos_angle=True)
                
                # Can fail with bad EPS values/instability
                except:
                    loss_pose = None

                if self.z_type == 'direct':
                    loss_z = self.l1_loss(cube_z, gt_z)

                elif self.z_type == 'sigmoid':
                    loss_z = self.l1_loss(cube_z_norm, (gt_z * real_to_virtual / 100).clip(0, 1))
                    
                elif self.z_type == 'log':
                    loss_z = self.l1_loss(cube_z_norm, torch.log((gt_z * real_to_virtual).clip(0.01)))

                elif self.z_type == 'clusters':
                    loss_z = self.l1_loss(cube_z_norm, (((gt_z * real_to_virtual) - z_means)/(z_stds)))
            
            total_3D_loss_for_reporting = loss_dims*self.loss_w_dims

            if not loss_pose is None:
                total_3D_loss_for_reporting += loss_pose*self.loss_w_pose

            if not cube_2d_deltas is None:
                total_3D_loss_for_reporting += loss_xy*self.loss_w_xy

            if not loss_z is None:
                total_3D_loss_for_reporting += loss_z*self.loss_w_z
            
            # reporting does not need gradients
            total_3D_loss_for_reporting = total_3D_loss_for_reporting.detach()

            if self.loss_w_joint > 0:
                '''
                If we are using joint [entangled] loss, then we also need to pair all 
                predictions together and compute a chamfer or l1 loss vs. cube corners.
                '''
                
                cube_dis_x3d_from_z = cube_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
                cube_dis_y3d_from_z = cube_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
                cube_dis_z = torch.cat((torch.stack((cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z)).T, cube_dims), dim=1)
                dis_z_corners_joint = util.get_cuboid_verts_faces(cube_dis_z, cube_pose)[0]
                
                if self.chamfer_pose and self.disentangled_loss:
                    loss_joint = self.chamfer_loss(dis_z_corners_joint, gt_corners)

                else:
                    loss_joint = self.l1_loss(dis_z_corners_joint, gt_corners).contiguous().view(n, -1).mean(dim=1)

                valid_joint = loss_joint < np.inf
                total_3D_loss_for_reporting += (loss_joint*self.loss_w_joint).detach()

            # compute errors for tracking purposes
            z_error = (cube_z - gt_z).detach().abs()
            dims_error = (cube_dims - gt_dims).detach().abs()
            xy_error = (cube_xy - gt_2d).detach().abs()

            storage.put_scalar(prefix + 'z_error', z_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'dims_error', dims_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'xy_error', xy_error.mean().item(), smoothing_hint=False)
            storage.put_scalar(prefix + 'z_close', (z_error<0.20).float().mean().item(), smoothing_hint=False)
            
            storage.put_scalar(prefix + 'total_3D_loss', self.loss_w_3d * self.safely_reduce_losses(total_3D_loss_for_reporting), smoothing_hint=False)

            if self.inverse_z_weight:
                '''
                Weights all losses to prioritize close up boxes.
                '''

                gt_z = gt_boxes3D[:, 2]

                inverse_z_w = 1/torch.log(gt_z.clip(E_CONSTANT))
                
                loss_dims *= inverse_z_w

                # scale based on log, but clip at e
                if not cube_2d_deltas is None:
                    loss_xy *= inverse_z_w
                
                if loss_z is not None:
                    loss_z *= inverse_z_w

                if loss_pose is not None:
                    loss_pose *= inverse_z_w
    
                if self.loss_w_joint > 0:
                    loss_joint *= inverse_z_w

            if self.use_confidence > 0:
                
                uncert_sf = SQRT_2_CONSTANT * torch.exp(-cube_uncert)
                
                loss_dims *= uncert_sf

                if not cube_2d_deltas is None:
                    loss_xy *= uncert_sf

                if not loss_z is None:
                    loss_z *= uncert_sf

                if loss_pose is not None:
                    loss_pose *= uncert_sf
    
                if self.loss_w_joint > 0:
                    loss_joint *= uncert_sf

                losses.update({prefix + 'uncert': self.use_confidence*self.safely_reduce_losses(cube_uncert.clone())})
                storage.put_scalar(prefix + 'conf', torch.exp(-cube_uncert).mean().item(), smoothing_hint=False)

            # store per batch loss stats temporarily
            self.batch_losses = [batch_losses.mean().item() for batch_losses in total_3D_loss_for_reporting.split(num_boxes_per_image)]
            
            if self.loss_w_dims > 0:
                losses.update({
                    prefix + 'loss_dims': self.safely_reduce_losses(loss_dims) * self.loss_w_dims * self.loss_w_3d,
                })

            if not cube_2d_deltas is None:
                losses.update({
                    prefix + 'loss_xy': self.safely_reduce_losses(loss_xy) * self.loss_w_xy * self.loss_w_3d,
                })

            if not loss_z is None:
                losses.update({
                    prefix + 'loss_z': self.safely_reduce_losses(loss_z) * self.loss_w_z * self.loss_w_3d,
                })

            if loss_pose is not None:
                
                losses.update({
                    prefix + 'loss_pose': self.safely_reduce_losses(loss_pose) * self.loss_w_pose * self.loss_w_3d, 
                })

            if self.loss_w_joint > 0:
                if valid_joint.any():
                    losses.update({prefix + 'loss_joint': self.safely_reduce_losses(loss_joint[valid_joint]) * self.loss_w_joint * self.loss_w_3d})

            
        '''
        Inference
        '''
        if len(cube_z.shape) == 0:
            cube_z = cube_z.unsqueeze(0)

        # inference
        cube_x3d = cube_z * (cube_x - Ks_scaled_per_box[:, 0, 2])/Ks_scaled_per_box[:, 0, 0]
        cube_y3d = cube_z * (cube_y - Ks_scaled_per_box[:, 1, 2])/Ks_scaled_per_box[:, 1, 1]
        cube_3D = torch.cat((torch.stack((cube_x3d, cube_y3d, cube_z)).T, cube_dims, cube_xy*im_ratios_per_box.unsqueeze(1)), dim=1)

        if self.use_confidence:
            cube_conf = torch.exp(-cube_uncert)
            cube_3D = torch.cat((cube_3D, cube_conf.unsqueeze(1)), dim=1)

        # convert the predictions to intances per image
        cube_3D = cube_3D.split(num_boxes_per_image)
        cube_pose = cube_pose.split(num_boxes_per_image)
        box_classes = box_classes.split(num_boxes_per_image)
        
        pred_instances = None
        
        pred_instances = instances if not self.training else \
            [Instances(image_size) for image_size in im_current_dims]

        for cube_3D_i, cube_pose_i, instances_i, K, im_dim, im_scale_ratio, box_classes_i, pred_boxes_i in \
            zip(cube_3D, cube_pose, pred_instances, Ks, im_current_dims, im_scales_ratio, box_classes, pred_boxes):
            
            # merge scores if they already exist
            if hasattr(instances_i, 'scores'):
                instances_i.scores = (instances_i.scores * cube_3D_i[:, -1])**(1/2)
            
            # assign scores if none are present
            else:
                instances_i.scores = cube_3D_i[:, -1]
            
            # assign box classes if none exist
            if not hasattr(instances_i, 'pred_classes'):
                instances_i.pred_classes = box_classes_i

            # assign predicted boxes if none exist    
            if not hasattr(instances_i, 'pred_boxes'):
                instances_i.pred_boxes = pred_boxes_i

            instances_i.pred_bbox3D = util.get_cuboid_verts_faces(cube_3D_i[:, :6], cube_pose_i)[0]
            instances_i.pred_center_cam = cube_3D_i[:, :3]
            instances_i.pred_center_2D = cube_3D_i[:, 6:8]
            instances_i.pred_dimensions = cube_3D_i[:, 3:6]
            instances_i.pred_pose = cube_pose_i

        if self.training:
            return pred_instances, losses
        else:
            return pred_instances

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, matched_ious=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes, matched_ious=matched_ious
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]
    
    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        
        #separate valid and ignore gts
        targets_ign = [target[target.gt_classes < 0] for target in targets]
        targets = [target[target.gt_classes >= 0] for target in targets]
        
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []

        for proposals_per_image, targets_per_image, targets_ign_per_image in zip(proposals, targets, targets_ign):
            
            has_gt = len(targets_per_image) > 0
            
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            try:
                if len(targets_ign_per_image) > 0:

                    # compute the quality matrix, only on subset of background
                    background_inds = (matched_labels == 0).nonzero().squeeze()

                    # determine the boxes inside ignore regions with sufficient threshold
                    if background_inds.numel() > 1:
                        match_quality_matrix_ign = pairwise_ioa(targets_ign_per_image.gt_boxes, proposals_per_image.proposal_boxes[background_inds])
                        matched_labels[background_inds[match_quality_matrix_ign.max(0)[0] >= self.ignore_thresh]] = -1
                    
                        del match_quality_matrix_ign
            except:
                pass
            
            gt_arange = torch.arange(match_quality_matrix.shape[1]).to(matched_idxs.device)
            matched_ious = match_quality_matrix[matched_idxs, gt_arange]
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes, matched_ious=matched_ious)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


    def safely_reduce_losses(self, loss):

        valid = (~(loss.isinf())) & (~(loss.isnan()))

        if valid.any():
            return loss[valid].mean()
        else:
            # no valid losses, simply zero out
            return loss.mean()*0.0
        