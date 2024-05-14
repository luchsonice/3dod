git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e . --user
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

cd sam-hq
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth -O sam_hq_vit_b.pth
cd ..

cd depth
./download_models.sh
cd ..