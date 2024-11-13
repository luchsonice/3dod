# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer
from cubercnn import data


from cubercnn.evaluation.omni3d_evaluation import Omni3DEvaluationHelper, instances_to_coco_json


logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
from tqdm import tqdm

def do_test(args, cfg, model):

    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')

    model.eval()
    
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)
    print('saving to', output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    name = 'KITTI' # 'KITTI' or 'SUNRGBD'
    split = 'test'
    dataset_paths_to_json = [f'datasets/Omni3D/{name}_{split}.json',]

    # Example 1. load all images
    dataset = data.Omni3D(dataset_paths_to_json)
    imgIds = dataset.getImgIds()
    imgs = dataset.loadImgs(imgIds)

    inference_json = [] 

    for i, path in enumerate(tqdm(imgs)):
        path = 'datasets/'+path['file_path']
        im_name = util.file_parts(path)[1]
        im = util.imread(path)

        if im is None:
            continue

        img_id = imgs[i]['id']
        K = np.array(imgs[i]['K'])
        
        image_shape = im.shape[:2]  # h, w

        h, w = image_shape
        
        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2

        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point

        # K = np.array([
        #     [focal_length, 0.0, px], 
        #     [0.0, focal_length, py], 
        #     [0.0, 0.0, 1.0]
        # ])
        # is_ground = os.path.exists(f'datasets/ground_maps/{im_name}.npz')
        # if is_ground:
            # ground_map = np.load(f'datasets/ground_maps/{im_name}.npz')['mask']
        # depth_map = np.load(f'datasets/depth_maps/{im_name}.npz')['depth']
        is_ground = False

        aug_input = T.AugInput(im)
        tfms = augmentations(aug_input)
        image = aug_input.image
        if is_ground:
            ground_map = tfms.apply_image(ground_map*1.0)
            ground_map = torch.as_tensor(ground_map)
        else:
            ground_map = None
        depth_map = None

        # batched = [{
        #     'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
        #     'height': image_shape[0], 'width': image_shape[1], 'K': K
        # }]
        # first you must run the scripts to get the ground and depth map for the images
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))), 
            'depth_map': depth_map,
            'ground_map': ground_map,
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]
        dets = model(batched)[0]['instances']

        n_det = len(dets)

        meshes_pred = []
        meshes_text = []
        bboxes = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx, bbox) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes, dets.pred_boxes
                )):

                # skip
                if score < thres:
                    continue
                
                cat = cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes_pred.append(box_mesh)
                bboxes.append(bbox.tolist())

        if args.display:
            if len(meshes_pred) > 0:

                # find the img which has im_name in the file_path

                # Example 2. load annotations for image index 0
                annIds = dataset.getAnnIds(imgIds=img_id)
                anns = dataset.loadAnns(annIds)

                meshes_gt = []
                # append gt mesh to meshes
                for ann in anns:
                    bbox3D = ann['bbox3D_cam']
                    pose = ann['R_cam']
                    dimensions = ann['dimensions']
                    cube = ann['center_cam'] + dimensions
                    if ann['category_name'] == 'dontcare':
                        continue
                    box_mesh = util.mesh_cuboid(cube, pose, color=color)
                    meshes_gt.append(box_mesh)
                    meshes_text.append(ann['category_name'])
                colors_pred = [np.array([0, 1, 0, 0.6]) for _ in range(len(meshes_pred))]
                colors_gt   = [np.array([0, 0, 1, 0.6]) for _ in range(len(meshes_gt))]

                meshes = meshes_pred + meshes_gt
                colors = colors_pred + colors_gt

                im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85, colors=None)
                
                # our bboxes coordinates are (top left, bottom right)
                # bboxes = Boxes(bboxes)
                # v_pred = Visualizer(im_drawn_rgb, None)
                # v_pred = v_pred.overlay_instances(
                #     boxes=bboxes.tensor.cpu().numpy()
                # )
                # im_drawn_rgb = v_pred.get_image()
                util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
                util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))

        # save detections
        prediction = {
                    "image_id": im_name,
                    "K": K,
                    "width": w,
                    "height": h,
                }

        # convert to json format
        # filter out dets with score < thres  
        dets = dets[dets.scores > thres]
        
        prediction["instances"] = instances_to_coco_json(dets.to('cpu'), im_name)

        # store in overall predictions
        inference_json.append(prediction)


    # save 
    eval_helper = Omni3DEvaluationHelper(
        ['KITTI_pred'], 
        None, 
        output_dir, 
        iter_label='final',
    )

    eval_helper.add_predictions('KITTI_pred', inference_json)
    eval_helper.save_predictions('KITTI_pred')
        
        

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
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )