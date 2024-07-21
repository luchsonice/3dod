import numpy as np
import gradio as gr
from io import BytesIO
import matplotlib.pyplot as plt

import logging
import os
import sys
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from cubercnn.data.generate_depth_maps import depth_of_images, setup_depth_model
from cubercnn.data.generate_ground_segmentations import find_ground, init_segmentation, load_image2
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from groundingdino.util.inference import load_model

sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis

def generate_imshow_plot(depth_map):
    # Generate a dummy depth map for demonstration
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    
    # Display the depth map using imshow
    cax = ax.imshow(depth_map, cmap='viridis')
    
    # Add a colorbar to the plot
    fig.colorbar(cax)
    return fig

def do_test(im, threshold, model_str):
    if im is None:
        return None, None
    
    model.eval()
    
    thres = threshold

    min_size = 512
    max_size = 4096
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    category_path = 'configs/category_meta.json'
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    
    image_shape = im.shape[:2]  # h, w

    h, w = image_shape
    
    focal_length_ndc = 4.0
    focal_length = focal_length_ndc * h / 2

    px, py = w/2, h/2

    K = np.array([
        [focal_length, 0.0, px], 
        [0.0, focal_length, py], 
        [0.0, 0.0, 1.0]
    ])
    # is_ground = os.path.exists(f'datasets/ground_maps/{im_name}.jpg.npz')
    # if is_ground:
    #     ground_map = np.load(f'datasets/ground_maps/{im_name}.jpg.npz')['mask']
    # depth_map = np.load(f'datasets/depth_maps/{im_name}.jpg.npz')['depth']

    # dummy
    # model.to(device)

    image_source, image_tensor =  load_image2(im, device=device)
    ground_map = find_ground(image_source, image_tensor, ground_model, segmentor, device=device)
    if ground_map is not None: is_ground = True 
    else: is_ground = False
    depth_map = depth_of_images(im, depth_model)


    aug_input = T.AugInput(im)
    tfms = augmentations(aug_input)
    image = aug_input.image
    if is_ground:
        ground_map = tfms.apply_image(ground_map*1.0)
        ground_map = torch.as_tensor(ground_map)
    else:
        ground_map = None
    depth_map = tfms.apply_image(depth_map)

    # first you must run the scripts to get the ground and depth map for the images
    batched = [{
        'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))), 
        'depth_map': torch.as_tensor(depth_map),
        'ground_map': ground_map,
        'height': image_shape[0], 'width': image_shape[1], 'K': K
    }]
    with torch.no_grad():
        dets = model(batched)[0]['instances']

    n_det = len(dets)

    meshes = []
    meshes_text = []

    if n_det > 0:
        for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                dets.pred_pose, dets.scores, dets.pred_classes
            )):

            # skip
            if score < thres:
                continue
            
            cat = cats[cat_idx]

            bbox3D = center_cam.tolist() + dimensions.tolist()
            meshes_text.append('{} {:.2f}'.format(cat, score))
            color = [c/255.0 for c in util.get_color(idx)]
            box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
            meshes.append(box_mesh)
    
    print('File with {} dets'.format(len(meshes)))

    if len(meshes) > 0:
        im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
        im_drawn_rgb, im_topdown = im_drawn_rgb.astype(np.uint8), im_topdown.astype(np.uint8)
    else:
        im_drawn_rgb, im_topdown = im.astype(np.uint8), None
    if ground_map is None:
        ground_map = torch.zeros(image_shape[0], image_shape[1])
    
    return im_drawn_rgb, im_topdown, generate_imshow_plot(depth_map), ground_map.numpy()

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def main(config_file, weigths=None):
    cfg = setup(config_file)
    model = build_model(cfg)
    
    DetectionCheckpointer(model).resume_or_load(
        weigths, resume=True
    )
    return cfg, model


if __name__ == "__main__":
    
    config_file =  "configs/BoxNet.yaml"
    MODEL_WEIGHTS = "output/Baseline_sgd/model_final.pth"
    cfg, model = main(config_file, MODEL_WEIGHTS) 

    depth_model = 'zoedepth'
    pretrained_resource = 'local::depth/checkpoints/depth_anything_metric_depth_indoor.pt'
    depth_model = setup_depth_model(depth_model, pretrained_resource)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ground_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth", device=device)
    segmentor = init_segmentation(device=device)

    demo = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image", value="datasets/examples/ex1.jpg"), 
            gr.Slider(0, 1, value=0.25, label="Threshold"),
            gr.Radio(["Proposal method", "Pseudo GT method", "Weak Cube R-CNN", "Time-equalized Cube R-CNN", "Cube R-CNN"], label="Method", value="Proposal method"),
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view"), gr.Plot(label="Depth map"), gr.Image(label="Ground map")],
            title="3D cube prediction",
            description="This showcases the different models, developed for our thesis \"weakly supervised 3D object detection\"", 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[]],)
    demo.launch()

    # io1 = gr.Interface(lambda x:x, "textbox", "textbox")
    # io2 = gr.Interface(lambda x:x, "image", "image")

    # def show_row(value):
    #     if value=="Interface 1":
    #         return (gr.update(visible=True), gr.update(visible=False))  
    #     if value=="Interface 2":
    #         return (gr.update(visible=False), gr.update(visible=True))
    #     return (gr.update(visible=False), gr.update(visible=False))

    # with gr.Blocks() as demo:
    #     d = gr.Dropdown(["Interface 1", "Interface 2"])
    #     with gr.Row(visible=False) as r1:
    #         io1.render()
    #     with gr.Row(visible=False) as r2:
    #         io2.render()
    #     d.change(show_row, d, [r1, r2])
        
    # demo.launch()