# Check if the "checkpoints" directory exists
if [ ! -d "checkpoints" ]; then
    # Create the "checkpoints" directory
    mkdir checkpoints
fi

# Download the pre-trained models

cd checkpoints

wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true -O depth_anything_v2_metric_hypersim_vitl.pth