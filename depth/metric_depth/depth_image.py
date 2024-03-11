# Born out of Issue 36. 
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import matplotlib.pyplot as plt

# Global settings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 256
FINAL_WIDTH = 256
INPUT_DIR = 'metric_depth/input'
OUTPUT_DIR = 'metric_depth/output'
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

def process_images(model):
    print('output:', OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        # try:
        color_image = Image.open(image_path).convert('RGB')
        original_width, original_height = color_image.size
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        pred = model(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.squeeze().detach().cpu().numpy()

        # Resize color image and depth to final size
        # resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
        resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)

    # resized_pred is the image shaped to the original image size, depth is in meters
    return np.array(resized_pred)

def setup_depth_model(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    model = setup_depth_model(args.model, args.pretrained_resource)

    resized_pred = process_images(model)