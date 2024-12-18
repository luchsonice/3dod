# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import logging
from detectron2.config.config import configurable
from detectron2.data.transforms.augmentation import AugmentationList
import torch
import numpy as np
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import (
    DatasetMapper
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)

from typing import List, Optional, Union

from PIL import Image

class DatasetMapper3D(DatasetMapper):

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        mode:str=None,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        only_2d: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            mode: 'get_depth_maps' (default), 'cube_rcnn'
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.only_2d                = only_2d
        self.mode                   = mode
        # fmt: on
        logger = logging.getLogger(__name__)
        mode_out = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode_out}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, mode='get_depth_maps'):
        augs = detection_utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "mode": mode,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "only_2d": cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = detection_utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        # state = torch.get_rng_state()
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w

        # dont load ground map and depth map when 
        if not self.only_2d:
            if 'depth_image_path' in dataset_dict:
                dp_img = Image.fromarray(np.load(dataset_dict["depth_image_path"])['depth'])
                dp_img = np.array(dp_img.resize(image.shape[:2][::-1], Image.NEAREST))
                aug_input_dp = T.AugInput(dp_img)
                aug_only_flip = AugmentationList(transforms[-1:])
                # torch.set_rng_state(state)
                #transforms_dp = aug_only_flip(aug_input_dp)
                dp_image = aug_input_dp.image
                dataset_dict["depth_map"] = torch.as_tensor(np.ascontiguousarray(dp_image))
            else:
                dataset_dict["depth_map"] = None

            #  ground image
            if 'ground_image_path' in dataset_dict:
                ground_img = Image.fromarray(np.load(dataset_dict["ground_image_path"])['mask'])
                ground_img = np.array(ground_img.resize(image.shape[:2][::-1], Image.NEAREST))
                aug_input_gr = T.AugInput(ground_img)
                #transforms_gr = aug_only_flip(aug_input_gr)
                gr_image = aug_input_gr.image
                dataset_dict["ground_map"] = torch.as_tensor(np.ascontiguousarray(gr_image))
            else:
                dataset_dict["ground_map"] = None

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # no need for additional processing at inference
        if not self.is_train:
            return dataset_dict

        if "annotations" in dataset_dict:

            dataset_id = dataset_dict['dataset_id']
            K = np.array(dataset_dict['K'])

            unknown_categories = self.dataset_id_to_unknown_cats[dataset_id]

            # transform and pop off annotations
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]

            # convert to instance format
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict

'''
Cached for mirroring annotations
'''
_M1 = np.array([
    [1, 0, 0], 
    [0, -1, 0],
    [0, 0, -1]
])
_M2 = np.array([
    [-1.,  0.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0.,  1.]
])


def transform_instance_annotations(annotation, transforms, *, K):
    
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))[0]
    
    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if annotation['center_cam'][2] != 0:

        # project the 3D box annotation XYZ_3D to screen 
        point3D = annotation['center_cam']
        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        # apply coords transforms to 2D box
        annotation["center_cam_proj"][0:2] = transforms.apply_coords(
            point2D[np.newaxis][:, :2]
        )[0].tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]
        
        if annotation['ignore']:
            # all keypoints marked as not visible 
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:
            
            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        # in place
        transforms.apply_coords(keypoints[:, :2])
        annotation["keypoints"] = keypoints.tolist()

        # manually apply mirror for pose
        for transform in transforms:

            # horrizontal flip?
            if isinstance(transform, T.HFlipTransform):

                pose = _M1 @ np.array(annotation["pose"]) @ _M2
                annotation["pose"] = pose.tolist()
                annotation["R_cam"] = pose.tolist()

    return annotation


def annotations_to_instances(annos, image_size, unknown_categories):

    # init
    target = Instances(image_size)
    
    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor([int(obj["category_id"]) for obj in annos], dtype=torch.int64)
    target.gt_boxes = Boxes([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos])
    target.gt_boxes3D = torch.FloatTensor([anno['center_cam_proj'] + anno['dimensions'] + anno['center_cam'] for anno in annos])
    target.gt_poses = torch.FloatTensor([anno['pose'] for anno in annos])
    
    n = len(target.gt_classes)

    # do keypoints?
    target.gt_keypoints = Keypoints(torch.FloatTensor([anno['keypoints'] for anno in annos]))

    gt_unknown_category_mask = torch.zeros(max(unknown_categories)+1, dtype=bool)
    gt_unknown_category_mask[torch.tensor(list(unknown_categories))] = True

    # include available category indices as tensor with GTs
    target.gt_unknown_category_mask = gt_unknown_category_mask.unsqueeze(0).repeat([n, 1])

    return target
