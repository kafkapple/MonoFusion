# MonoFusion Preprocessing
**[Project Page](https://ImNotPrepared.github.io/research/25_DSR/index.html) | [Arxiv](https://arxiv.org/abs/2507.23782) | [Data](https://drive.google.com/drive/folders/18H8OOOZLv7OmOen8pGbSLWwu8AvZAZro?usp=sharing)**

This folder packages every dependency needed by `process_custom.py`: AutoMask (SAM2 + XMem), Bootstapir/TAPIR tracks, DINOv2 features, DUSt3R, and MoGE depth alignment.

## Installation
```bash
cd preproc
./setup_dependencies.sh  # installs AutoMask, TapNet, GroundingDINO, checkpoints
```
Run the script from inside the `monofusion` conda env created in the top-level README.

## Usage
### 1. Arrange your capture root
```
../raw_data/_scene_token/
├── images/
│   ├── cam01/*.png
│   ├── cam02/*.png
│   └── ...
├── videos/ (optional mp4 copies for AutoMask)
└── metadata/... (Dy_train_meta.json, etc.)
```
Every path passed to `--img-dirs` must include the literal `images` component; the helper derives sibling folders (`sam_v2_dyn_mask`, `bootstapir`, `dust3r`, …) by swapping this segment.

### 2. Run the batch driver
```bash
cd preproc
python process_custom.py \
  --img-dirs ../raw_data/_scene_token/images/cam0* \
  --gpus 0 1
```
- Expand `--img-dirs` with globs to cover every camera. The script assigns workers across the GPU list.
- Override prompts or roots via flags such as `--automask-prompt`, `--dust3r-raw-root`, or `--dust3r-seq-names scene_cam01 scene_cam02`.
- The DUSt3R step auto-resolves `_raw_data/<SEQ_NAME>` (preferred) or falls back to `../raw_data/<SEQ_NAME>` when mapping short names.

### 3. What `process_custom.py` runs per sequence
- **AutoMask** (`AutoMask/custom_mask.py` + `visualize_masks.py`) → `sam_v2_dyn_mask/<SEQ_NAME>/camXX/` PNG masks driven by the default `person` prompt (`--automask-prompt` to change).
- **Bootstapir (TAPIR-Torch)** (`compute_tracks_torch.py`) → `bootstapir/<SEQ_NAME>/camXX` track files for foreground points; switch to the JAX backend with `--tapir-torch False`.
- **DINOv2 Features** (`compute_dinofeatures.py`) → `dinov2_features/<SEQ_NAME>/` tensors plus PCA visualizations under `<...>/pca_viz/`. Use flags like `--feature-max-frames 200` to trim runtime.
- **MoGE raw depth** (`compute_raw_moge_depth.py`) → `raw_moge_depth/<SEQ_NAME>/` storing raw metric depth, intrinsics, and optional point clouds.
- **DUSt3R alignment** (`compute_dust3r.py`) → `dust3r/<SEQ_NAME>/dust_true_depth_conf3/` containing fused depth + scene graphs built from the corresponding `_raw_data` take.
- **Aligned MoGE depth** (`compute_aligned_moge_depth.py`) → `aligned_moge_depth/<SEQ_NAME>/` masks, confidence maps, and filtered clouds. These outputs feed directly into the training configs.

### 4. Resulting folder snapshot
```
../raw_data/_scene_token/
├── images/cam0*/frame_*.png
├── sam_v2_dyn_mask/cam0*/mask_*.png
├── bootstapir/cam0*/tracks_*.npz
├── dinov2_features/cam0*/{features.npy,pca_viz/*.png}
├── raw_moge_depth/cam0*/depth_*.npy
├── dust3r/cam0*/dust_true_depth_conf3/*.npz
└── aligned_moge_depth/cam0*/aligned_depth_*.npz
```
Mirror or rsync these folders into `MonoFusion/data/<SEQ_NAME>/` before launching training if you keep raw data outside the repo root.

### 5. Re-running individual modules
Each helper script accepts the same `--img-dir` conventions used above:
```bash
CUDA_VISIBLE_DEVICES=0 python AutoMask/custom_mask.py --video_dir ../raw_data/_scene_token/videos \\
  --save_dir ../raw_data/_scene_token/sam_v2_dyn_mask --seq cam01 --text_prompt person
CUDA_VISIBLE_DEVICES=0 python compute_tracks_torch.py --image_dir ../raw_data/_scene_token/images/cam01 \\
  --mask_dir ../raw_data/_scene_token/sam_v2_dyn_mask/cam01 --model_type bootstapir --out_dir ../raw_data/_scene_token/bootstapir/cam01
CUDA_VISIBLE_DEVICES=0 python compute_dinofeatures.py --img_dir ../raw_data/_scene_token/images/cam01 --out_dir ../raw_data/_scene_token/dinov2_features/cam01 ...
CUDA_VISIBLE_DEVICES=0 python compute_raw_moge_depth.py --seq-name _scene_token --image-root ../raw_data/_scene_token/images \\
  --output-root ../raw_data/_scene_token/raw_moge_depth --raw-root ../_raw_data
CUDA_VISIBLE_DEVICES=0 python compute_dust3r.py --seq-name _scene_token --image-root ../raw_data/_scene_token/images \\
  --output-root ../raw_data/_scene_token/dust3r --raw-root ../_raw_data
CUDA_VISIBLE_DEVICES=0 python compute_aligned_moge_depth.py --seq-name _scene_token --raw-moge-root ../raw_data/_scene_token/raw_moge_depth \\
  --dust3r-root ../raw_data/_scene_token/dust3r --output-root ../raw_data/_scene_token/aligned_moge_depth
```
Use these when debugging a single stage or when tweaking parameters without re-running the entire pipeline.
