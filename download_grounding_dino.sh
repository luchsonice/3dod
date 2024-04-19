git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e . --user
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
