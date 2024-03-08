import pickle
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers.nms import batched_nms
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
import numpy as np
import torch
from cubercnn import util, vis
from transformers import AutoImageProcessor, DepthAnythingConfig, DepthAnythingForDepthEstimation
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch

def depth_of_image(image):
    '''Use Depth anything from huggingface
    might need to install pip install git+https://github.com/huggingface/transformers.git
    '''

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # configuration = DepthAnythingConfig()
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = DepthAnythingForDepthEstimation.from_pretrained('LiheYoung/depth-anything-small-hf')

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.shape[:-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return formatted, output


def plot_3dbox_in2d(image, pred2d, pred3d):

    

    return

def make_random_boxes(n_boxes=10):
    # rotation_matrix = torch.rand(3,3)*2*torch.pi

    rotation_matrix = torch.eye(3) # no rotation
    
    # need xyz, whl, and pose (R)
    # whl = torch.rand(3)*0.5
    whl = torch.tensor([0.3, 0.3, 0.3])
    xyz = torch.tensor([-0.1, 0, 1.7])
    # xyz = torch.rand(3)*1
    return xyz, whl, rotation_matrix


def proposals_3d_from_2d(image, pred2d):

    with open('3dboxes/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)
    
    n_boxes = 1
    pred_xyz, pred_whl, pred_pose = make_random_boxes(n_boxes=n_boxes)
    pred_xyzwhl = torch.cat((pred_xyz, pred_whl), dim=0)

    pred_colors = torch.tensor([util.get_color(i) for i in range(n_boxes)])/255.0

    pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

    input_format = 'BGR'
    img = batched_inputs[0]['image']
    img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
    img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
    input = batched_inputs[0]
    K = torch.tensor(input['K'])
    scale = input['height']/img.shape[0]
    
    K_scaled = torch.tensor(
        [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
        dtype=torch.float32) @ K
    # convert to lists
    pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]

    # horizontal stack 3D GT and pred left/right


    # 2 box
    box_size = min(len(proposals[0].proposal_boxes), 2)
    v_pred = Visualizer(img, None)
    v_pred = v_pred.overlay_instances(
        boxes=proposals[0].proposal_boxes[0:box_size].tensor.cpu().numpy()
    )
    prop_img = v_pred.get_image()
    img_3DPR = vis.draw_scene_view(prop_img, K_scaled.cpu().numpy(), pred_meshes, text=['3d box'], mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
    # vis_img_3d = img_3DPR[:, :, [2, 1, 0]] # RGB
    vis_img_3d = img_3DPR.astype(np.uint8)
    fig, ax = plt.subplots(); ax.imshow(vis_img_3d); ax.axis('off')
    
    plt.savefig(f'3dboxes/proposals/figs/pred.png', bbox_inches='tight', dpi=300)

    # visualize(batched_inputs, proposals, instances)


    return

def visualize(batched_inputs, proposals, instances):
    # taken from the class ROIHeads3D
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 top-scoring predicted
    object proposals on the original image. Users can implement different
    visualization functions for different models.
    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
        instances (list): a list that contains predicted RoIhead instances. Both
            batched_inputs and proposals should have the same length.
    """
    max_vis_prop = 2

    device = 'cpu'
    input_format = 'BGR'

    # thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    thing_classes = ['pedestrian', 'car', 'cyclist', 'van', 'truck', 'traffic cone', 'barrier', 'motorcycle', 'bicycle', 'bus', 'trailer', 'books', 'bottle', 'camera', 'cereal box', 'chair', 'cup', 'laptop', 'shoes', 'towel', 'blinds', 'window', 'lamp', 'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet', 'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed', 'night stand', 'pillow', 'sofa', 'television', 'floor mat', 'curtain', 'clothes', 'stationery', 'refrigerator', 'bin', 'stove', 'oven', 'machine']
    num_classes = len(thing_classes)

    for i, (input, prop, instances_i) in enumerate(zip(batched_inputs, proposals, instances)):

        img = input["image"]            
        img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
        img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
        img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR

        '''
        Visualize the 2D GT and proposal predictions
        '''
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        box_size = min(len(prop.proposal_boxes), max_vis_prop)
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
        )
        prop_img = v_pred.get_image()
        vis_img_rpn = np.concatenate((anno_img, prop_img), axis=1)
        # fig, ax = plt.subplots(); ax.imshow(vis_img_rpn); ax.axis('off')
        # plt.savefig(f'3dboxes/proposals/figs/vis_img_rpn_{i}.png', bbox_inches='tight', dpi=300)

        '''
        Visualize the 3D GT and predictions
        '''
        K = torch.tensor(input['K'], device=device)
        scale = input['height']/img.shape[0]
        fx, sx = (val.item()/scale for val in K[0, [0, 2]])
        fy, sy = (val.item()/scale for val in K[1, [1, 2]])
        
        K_scaled = torch.tensor(
            [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
            dtype=torch.float32, device=device
        ) @ K

        gts_per_image = input["instances"]

        gt_classes = gts_per_image.gt_classes
        
        # Filter out irrelevant groundtruth
        fg_selection_mask = (gt_classes != -1) & (gt_classes < num_classes)

        gt_classes = gt_classes[fg_selection_mask]
        gt_class_names = [thing_classes[cls_idx] for cls_idx in gt_classes]
        gt_boxes   = gts_per_image.gt_boxes.tensor[fg_selection_mask]  # 2D boxes
        gt_poses   = gts_per_image.gt_poses[fg_selection_mask]         # GT poses

        # projected 2D center, depth, w, h, l, 3D center
        gt_boxes3D = gts_per_image.gt_boxes3D[fg_selection_mask]

        # this box may have been mirrored and scaled so
        # we need to recompute XYZ in 3D by backprojecting.
        gt_z = gt_boxes3D[:, 2]

        gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx)/fx
        gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy)/fy
        
        # put together the GT boxes
        gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
        gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)

        gt_colors = torch.tensor(
            [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], 
            device=device
        )/255.0

        gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

        # perform a simple NMS, which is not cls dependent. 
        keep = batched_nms(
            instances_i.pred_boxes.tensor, 
            instances_i.scores, 
            torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device), 
            0.5 # this should come from roi_heads.nms_thresh
        )
        
        keep = keep[:max_vis_prop]
        num_to_visualize = len(keep)

        pred_xyzwhl = torch.cat((instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1)
        pred_pose = instances_i.pred_pose[keep]

        pred_colors = torch.tensor(
            [util.get_color(i) for i in range(num_to_visualize)], 
            device=device
        )/255.0

        pred_boxes = instances_i.pred_boxes[keep]
        pred_scores = instances_i.scores[keep]
        pred_classes = instances_i.pred_classes[keep]
        pred_class_names = ['{} {:.2f}'.format(thing_classes[cls_idx], score) for cls_idx, score in zip(pred_classes, pred_scores)]
        pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)
        # print(pred_xyzwhl)
        # convert to lists
        pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
        gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]
        
        img_3DPR = vis.draw_scene_view(anno_img, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
        img_3DGT = vis.draw_scene_view(img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)

        # horizontal stack 3D GT and pred left/right
        img_3DGT = img_3DGT[:, :, [2, 1, 0]] # RGB
        vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)
        vis_img_3d = vis_img_3d.astype(np.uint8)
        fig, ax = plt.subplots(); ax.imshow(vis_img_3d); ax.axis('off')
        plt.savefig(f'3dboxes/proposals/figs/vis_img_3d_{i}.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    # proposals_3d_from_2d(None, None)

    with open('3dboxes/proposals/network_out.pkl', 'rb') as f:
        batched_inputs, images, features, proposals, Ks, gt_instances, im_scales_ratio, instances = pickle.load(f)
    
    n_boxes = 1
    pred_xyz, pred_whl, pred_pose = make_random_boxes(n_boxes=n_boxes)
    pred_xyzwhl = torch.cat((pred_xyz, pred_whl), dim=0)

    pred_colors = torch.tensor([util.get_color(i) for i in range(n_boxes)])/255.0

    pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

    input_format = 'BGR'
    img = batched_inputs[0]['image']
    img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
    img_depth, prediction = depth_of_image(img)
    # plt.imshow(img_depth)
    # plt.show()
    plt.figure()
    plt.matshow(prediction, cmap='magma')
    plt.show()