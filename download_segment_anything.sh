# execute this from within the segment-anything directory
cd segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b_01ec64.pth
cd ..

cd sam-hq
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth -O sam_hq_vit_b.pth
cd ..