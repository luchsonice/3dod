from segment_anything.utils.transforms import ResizeLongestSide
from groundingdino.util.inference import load_image, load_model, predict
from torchvision.ops import box_convert
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import os
import torchvision.transforms as T2
import groundingdino.datasets.transforms as T
from PIL import Image

def init_segmentation(device='cpu') -> Sam:
    # 1) first cd into the segment_anything and pip install -e .
    # to get the model stary in the root foler folder and run the download_model.sh 
    # 2) chmod +x download_model.sh && ./download_model.sh
    # the largest model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # this is the smallest model
    if os.path.exists('sam-hq/sam_hq_vit_b.pth'):
        sam_checkpoint = "sam-hq/sam_hq_vit_b.pth"
        model_type = "vit_b"
    else:
        sam_checkpoint = "sam-hq/sam_hq_vit_tiny.pth"
        model_type = "vit_tiny"
    print(f'SAM device: {device}, model_type: {model_type}')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def find_ground(image_source, image, model, segmentor, device='cpu', TEXT_PROMPT="ground", BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25):
    boxes, logits, _ = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
    if len(boxes) == 0:
        return None
    # only want box corresponding to max logit
    max_logit_idx = torch.argmax(logits)
    box = boxes[max_logit_idx].unsqueeze(0)

    _, h, w = image_source.shape
    box = box * torch.tensor([w, h, w, h], device=device)
    xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy")

    image = image.unsqueeze(0)
    org_shape = image.shape[-2:]
    resize_transform = ResizeLongestSide(segmentor.image_encoder.img_size)
    batched_input = []
    images = resize_transform.apply_image_torch(image*1.0)# .permute(2, 0, 1).contiguous()
    for image, boxes in zip(images, xyxy):
        transformed_boxes = resize_transform.apply_boxes_torch(boxes, org_shape) # Bx4
        batched_input.append({'image': image, 'boxes': transformed_boxes, 'original_size':org_shape})

    seg_out = segmentor(batched_input, multimask_output=False)
    mask_per_image = seg_out[0]['masks']
    return mask_per_image[0,0,:,:].cpu().numpy()

def load_image2(image:np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform2 = T2.ToTensor()
    image_source = Image.fromarray(image).convert("RGB")
    image = transform2(image_source).to(device)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


if __name__ == '__main__':
    import pandas as pd
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from cubercnn import data
    from detectron2.data.catalog import MetadataCatalog
    from priors import get_config_and_filter_settings
    import supervision as sv
    
    def init_dataset():
        ''' dataloader stuff.
        currently not used anywhere, because I'm not sure what the difference between the omni3d dataset and load omni3D json functions are. this is a 3rd alternative to this. The train script calls something similar to this.'''
        cfg, filter_settings = get_config_and_filter_settings()

        dataset_names = ['SUNRGBD_train','SUNRGBD_val','SUNRGBD_test']
        dataset_paths_to_json = ['datasets/Omni3D/'+dataset_name+'.json' for dataset_name in dataset_names]
        # for dataset_name in dataset_names:
        #     simple_register(dataset_name, filter_settings, filter_empty=True)

        # Get Image and annotations
        datasets = data.Omni3D(dataset_paths_to_json, filter_settings=filter_settings)
        data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)


        thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
        dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

        infos = datasets.dataset['info']

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

        return datasets

    def load_image(image_path: str, device) -> tuple[torch.Tensor, torch.Tensor]:
        transform = T.Compose(
            [
                # T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform2 = T2.ToTensor()
        image_source = Image.open(image_path).convert("RGB")
        image = transform2(image_source).to(device)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed.to(device)


    def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: list[str]) -> np.ndarray:
        """    
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (np.ndarray): The source image to be annotated.
        boxes (torch.Tensor): A tensor containing bounding box coordinates.
        logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
        phrases (List[str]): A list of labels for each bounding box.

        Returns:
        np.ndarray: The annotated image.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]

        box_annotator = sv.BoxAnnotator()
        # annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        annotated_frame = image_source.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame


    # datasets = init_dataset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)

    segmentor = init_segmentation(device=device)

    os.makedirs('datasets/ground_maps', exist_ok=True)
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth", device=device)
    TEXT_PROMPT = "ground"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    noground = 0
    no_ground_idx = []

    #  **** to annotate full dataset ****
    # for img_id, img_info in tqdm(datasets.imgs.items()):
    #     file_path = img_info['file_path']
    #     width = img_info['width']
    #     height = img_info['height']
    #  **** to annotate full dataset ****
    #  **** to annotate demo images ****
    for img_id in tqdm(os.listdir('datasets/coco_examples')):
        file_path = 'coco_examples/'+img_id
        image_source, image = load_image('datasets/'+file_path, device=device)
    #  **** to annotate demo images ****

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
        if len(boxes) == 0:
            print(f"No ground found for {img_id}")
            noground += 1
            # save a ground map that is all zeros
            no_ground_idx.append(img_id)
            continue
        # only want box corresponding to max logit
        max_logit_idx = torch.argmax(logits)
        logit = logits[max_logit_idx].unsqueeze(0)
        box = boxes[max_logit_idx].unsqueeze(0)
        phrase = [phrases[max_logit_idx]]

        _, h, w = image_source.shape
        box = box * torch.tensor([w, h, w, h], device=device)
        xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy")

        image = image.unsqueeze(0)
        org_shape = image.shape[-2:]
        resize_transform = ResizeLongestSide(segmentor.image_encoder.img_size)
        batched_input = []
        images = resize_transform.apply_image_torch(image*1.0)# .permute(2, 0, 1).contiguous()
        for image, boxes in zip(images, xyxy):
            transformed_boxes = resize_transform.apply_boxes_torch(boxes, org_shape) # Bx4
            batched_input.append({'image': image, 'boxes': transformed_boxes, 'original_size':org_shape})

        seg_out = segmentor(batched_input, multimask_output=False)
        mask_per_image = seg_out[0]['masks']

        nnz = torch.count_nonzero(mask_per_image, dim=(-2, -1))
        indices = torch.nonzero(nnz <= 1000).flatten()
        if len(indices) > 0:
            noground += 1
            # save a ground map that is all zeros
            no_ground_idx.append(img_id)

        np.savez_compressed(f'datasets/ground_maps/{img_id}.npz', mask=mask_per_image.cpu()[0,0,:,:].numpy())

    print(f"Could not find ground for {noground} images")


    df = pd.DataFrame(no_ground_idx, columns=['img_id'])
    df.to_csv('datasets/no_ground_idx.csv', index=False)