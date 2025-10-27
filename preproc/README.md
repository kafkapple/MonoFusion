# MonoFusion Preprocessing
**[Project Page](https://ImNotPrepared.github.io/research/25_DSR/index.html) | [Arxiv](https://arxiv.org/abs/2507.23782) | [Data](https://drive.google.com/drive/folders/18H8OOOZLv7OmOen8pGbSLWwu8AvZAZro?usp=sharing)**

We rely on the following upstream projects:
- Metric depth — [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- Monocular depth — [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- Masks — [Track-Anything](https://github.com/gaomingqi/Track-Anything) (Segment Anything + XMem)
- Cameras — [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- Tracks — [TAPIR](https://github.com/google-deepmind/tapnet)

## Installation
```bash
cd preproc
./setup_dependencies.sh  # installs extra wheels, clones third-party repos, downloads checkpoints
```
The script expects you to run it inside the `monofusion` conda env from the main README. Re-run only when dependencies change.

## Usage
### 1. Prepare a data root
Organize raw assets before touching the scripts:
```
<data_root>
├── videos/
│   ├── seq1.mp4
│   └── seq2.mp4
└── images/
    ├── seq1
    └── seq2
```
Both `videos` and `images` are optional; mix them as needed.

### 2. Extract frames + masks
```bash
python mask_app.py --root_dir <data_root>
```
The Gradio UI lets you sample frames, draw foreground masks with Segment Anything, and refine them with XMem. Save the outputs per sequence (e.g., `images/SEQ_NAME`, `sam_v2_dyn_mask/SEQ_NAME`).

### 3. Generate priors
```bash
python process_custom.py --img-dirs <data_root>/images/** --gpus 0 1
```
This single entry point launches metric depth, mono depth, Track-Anything, TAPIR, DUSt3R, and SLAM back-to-back. Expect the following structure on completion:
```
<data_root>/
├── images/SEQ_NAME
├── sam_v2_dyn_mask/SEQ_NAME
├── unidepth_disp/SEQ_NAME
├── unidepth_intrins/SEQ_NAME
├── depth_anything/SEQ_NAME
├── aligned_depth_anything/SEQ_NAME
├── droid_recon/SEQ_NAME
└── bootstapir/SEQ_NAME
```
Point your main training run at the mirrored copy under `MonoFusion/data/SEQ_NAME` if you prefer keeping derived assets inside the repo.

### 4. Optional single modules
Every stage can be re-run independently:
```bash
python launch_metric_depth.py --img-dirs <data_root>/images/** --gpus 0
python launch_depth.py        --img-dirs <data_root>/images/** --gpus 0
python launch_slam.py         --img-dirs <data_root>/images/** --gpus 0
python launch_tracks.py       --img-dirs <data_root>/images/** --gpus 0
```
Use these if one component fails or if you need to tweak parameters without touching the all-in-one script.

### 5. TAPIR backends
- Default: `tapnet_torch` (PyTorch) ships with this repo and works out of the box.
- Optional: `tapnet` (JAX) lives as a submodule. For faster inference install TAPIR per its [official instructions](https://github.com/google-deepmind/tapnet) and run `compute_tracks_jax.py` instead of the torch implementation.

Keep raw captures under `../raw_data/` or any path you pass to the scripts. Outputs are idempotent; re-running a stage overwrites the previous results.
