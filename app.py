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

model_loaded = None
model, depth_model, ground_model, segmentor = None, None, None, None

def generate_imshow_plot(depth_map):
    # Generate a dummy depth map for demonstration
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(dpi=200, tight_layout=True)
    
    # Display the depth map using imshow
    cax = ax.imshow(depth_map, cmap='viridis')
    
    # Add a colorbar to the plot
    fig.colorbar(cax, shrink=0.8)
    return fig

def do_test(im, threshold, model_str):
    if im is None:
        return None, None
    # have to do this small workaround to only load the models once
    global model_loaded
    global model, depth_model, ground_model, segmentor
    if model_loaded != model_str:
        model, depth_model, ground_model, segmentor = load_model_config(model_str)
        model_loaded = model_str
    
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

    # dummy
    aug_input = T.AugInput(im)
    tfms = augmentations(aug_input)
    image = aug_input.image
    # model.to(device)
    if model_str == "Proposal method":
        image_source, image_tensor =  load_image2(im, device='cpu')
        ground_map = find_ground(image_source, image_tensor, ground_model, segmentor, device='cpu')
        if ground_map is not None: is_ground = True 
        else: is_ground = False
        depth_map = depth_of_images(im, depth_model)


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
    else:
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))), 
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
    if model_str == "Proposal method":
        if ground_map is None:
            ground_map = torch.zeros(image_shape[0], image_shape[1])
        
        return im_drawn_rgb, im_topdown, generate_imshow_plot(depth_map), ground_map.numpy()
    return im_drawn_rgb, im_topdown

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
    def load_model_config(model_str):
        if model_str == "Proposal method":
            config_file =  "configs/BoxNet.yaml"
            MODEL_WEIGHTS = "output/Baseline_sgd/model_final.pth"
            cfg, model = main(config_file, MODEL_WEIGHTS) 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            depth_model = 'zoedepth'
            pretrained_resource = 'local::depth/checkpoints/depth_anything_metric_depth_indoor.pt'
            depth_model = setup_depth_model(depth_model, pretrained_resource)
            ground_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth", device=device)
            segmentor = init_segmentation(device=device)
            return model, depth_model, ground_model, segmentor
        elif model_str == "Pseudo GT method":
            config_file =  "configs/Base_Omni3D.yaml"
            MODEL_WEIGHTS = "output/omni_pseudo_gt/model_final.pth"
            cfg, model = main(config_file, MODEL_WEIGHTS) 
            return model, None, None, None
        elif model_str == "Weak Cube R-CNN":
            config_file =  "configs/Omni_combined.yaml"
            MODEL_WEIGHTS = "output/exp_10_iou_zpseudogt_dims_depthrange_rotalign_ground/model_recent.pth"
            cfg, model = main(config_file, MODEL_WEIGHTS) 
            return model, None, None, None
        elif model_str == "Time-equalized Cube R-CNN":
            config_file =  "configs/Base_Omni3D.yaml"
            MODEL_WEIGHTS = "output/omni_equalised/model_final.pth"
            cfg, model = main(config_file, MODEL_WEIGHTS) 
            return model, None, None, None
        elif model_str == "Cube R-CNN":
            config_file =  "configs/Base_Omni3D.yaml"
            MODEL_WEIGHTS = "output/Baseline_sgd/model_final.pth"
            cfg, model = main(config_file, MODEL_WEIGHTS) 
            return model, None, None, None
        else:
            raise ValueError("Model not found")


    title = None
    description = "This showcases the different models, developed for our thesis \"weakly supervised 3D object detection\". You can choose between the different methods by selecting the tabs above. \n Upload an image, or use one of the example images below. \n We have created three methods using 2D box annotations: A **proposal-and-scoring method**, a **pseudo-ground-truth method**, and a **weak Cube R-CNN**. The proposal method generates 1000 cubes per object and scores them. The prediction of this method is used as a pseudo ground truth in the [[`Cube R-CNN framework`](https://garrickbrazil.com/omni3d)]. To create a weak Cube RCNN, we modify the framework by replacing its 3D loss functions with ones based solely on 2D annotations. Our methods rely heavily on external, strong generalised deep learning models to infer spatial information in scenes. Experimental results show that all models perform comparably to an annotation time-equalised Cube R-CNN, whereof the pseudo ground truth method achieves the highest accuracy. The results show the methods' ability to understand scenes in 3D, providing satisfactory visual results. Although not precise enough for centimetre accurate measurements, the methods provide a solid foundation for further research. \n Check out the code on [GitHub](https://github.com/luchsonice/3dod)"

    proposal = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image"), 
            gr.Slider(0, 1, value=0.25, label="Threshold", info="Only show predictions with a score above this threshold"),
            gr.Textbox(value="Proposal method", visible=False, render=False)
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view"), gr.Plot(label="Depth map"), gr.Image(label="Ground map")],
            title=title,
            description=description + "Note that the proposal method is very dependent on the finding the ground. You can see in the bottom two images how the ground is detected.", 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[],["datasets/examples/ex1.jpg"]],)

    pseudo_gt = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image"), 
            gr.Slider(0, 1, value=0.25, label="Threshold", info="Only show predictions with a confidence above this threshold"),
            gr.Textbox(value="Pseudo GT method", visible=False, render=False)
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view")],
            title=title,
            description=description, 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[],["datasets/examples/ex1.jpg"]],)
    
    
    weak_cube = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image"), 
            gr.Slider(0, 1, value=0.25, label="Threshold", info="Only show predictions with a confidence above this threshold"),
            gr.Textbox(value="Weak Cube R-CNN", visible=False, render=False)
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view")],
            title=title,
            description=description, 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[],["datasets/examples/ex1.jpg"]],)
    
    time_cube = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image"), 
            gr.Slider(0, 1, value=0.25, label="Threshold", info="Only show predictions with a confidence above this threshold"),
            gr.Textbox(value="Time-equalized Cube R-CNN", visible=False, render=False)
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view")],
            title=title,
            description=description, 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[],["datasets/examples/ex1.jpg"]],)
    cube_rcnn = gr.Interface(
        fn=do_test, 
        inputs=[
            gr.Image(label="Input Image"), 
            gr.Slider(0, 1, value=0.25, label="Threshold", info="Only show predictions with a confidence above this threshold"),
            gr.Textbox(value="Cube R-CNN", visible=False, render=False)
            ],
        outputs=[gr.Image(label="Predictions"), gr.Image(label="Top view")],
            title=title,
            description=description, 
            allow_flagging='never',
            examples=[["datasets/examples/ex2.jpg"],[],[],["datasets/examples/ex1.jpg"]],)


    demo = gr.TabbedInterface([pseudo_gt, proposal, weak_cube, time_cube, cube_rcnn], ["Pseudo GT method", "Proposal method", "Weak Cube R-CNN", "Time-equalized Cube R-CNN", "Cube R-CNN"], title="Weakly supervised 3D Cube Prediction")
    
    demo.launch(server_name="0.0.0.0", server_port=7860)