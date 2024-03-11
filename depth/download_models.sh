# Check if the "checkpoints" directory exists
if [ ! -d "checkpoints" ]; then
    # Create the "checkpoints" directory
    mkdir checkpoints
fi

# Download the pre-trained models

cd checkpoints

wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt?download=true -O depth_anything_metric_depth_indoor.pt
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true -O depth_anything_vitl14.pth