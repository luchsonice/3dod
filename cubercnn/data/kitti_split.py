import os
import shutil
from tqdm import tqdm

# Define the paths
val_file = "datasets/val.txt"
src_folder = "datasets/KITTI_object/training/image_2"
dest_folder = "datasets/KITTI_object/val/image_2"

os.makedirs(dest_folder, exist_ok=True)

# Read the val.txt file
with open(val_file, "r") as file:
    for line in tqdm(file):
        # Get the corresponding file name
        file_name = line.strip() + ".png"
        
        # Check if the file exists in the source folder
        if os.path.exists(os.path.join(src_folder, file_name)):
            # Copy the file to the destination folder
            shutil.copy(os.path.join(src_folder, file_name), dest_folder)