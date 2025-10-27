# install additional dependencies for track-anything and depth-anything
pip install -r requirements_extra.txt

# install droid-slam
echo "Installing DROID-SLAM..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

cd AutoMask/sam2
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \
cd ..


# install tapnet
echo "Installing TAPNet..."
cd tapnet
pip install .
cd ..

echo "Downloading checkpoints..."
mkdir checkpoints
cd checkpoints
# sam_vit_h checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# tapir checkpoint
wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
echo "Done downloading checkpoints"


