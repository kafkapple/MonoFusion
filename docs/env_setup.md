# MonoFusion Environment Setup

> **DEPRECATED**: This document describes the initial single-env approach.
> Superseded by `mouse_m5t2/envs/SETUP_GUIDE.md` (3-env multi-CUDA approach).
> Kept for historical reference only.

## ~~Conda Environment: `monofusion`~~ (REVOKED)

~~Single env 전략 — training + preprocessing 통합.~~

### Installed Components

```
Python:      3.10
PyTorch:     2.1.0+cu118
gsplat:      1.5.3 (rasterization verified on A6000 sm_86)
CUDA:        11.8 (conda nvidia toolkit, env-isolated)
nvcc:        ~/anaconda3/envs/monofusion/bin/nvcc
CUDA_HOME:   ~/anaconda3/envs/monofusion/
numpy:       1.26.4 (< 2.0 for PyTorch 2.1 compatibility)
```

### Activation

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate monofusion
export CUDA_VISIBLE_DEVICES=4  # A6000
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"
```

### Verification (2026-03-27)

```
gsplat rasterization: PASSED (output shape [1, 512, 512, 3])
GPU: NVIDIA RTX A6000, sm_86
CUDA_HOME: conda env internal (/home/joon/anaconda3/envs/monofusion)
nvcc: 11.8 (conda, not system)
```

## Preprocessing Submodule Installation Plan

### Priority Order (M5t2 dataset — what we already have vs need)

| Step | Tool | Status | What it produces | M5t2 has? |
|------|------|--------|-----------------|-----------|
| 0 | RGBA unpack | TODO | RGB + alpha mask | alpha=YES |
| 1 | DUSt3R | TODO | Multi-view depth | NO → need |
| 2 | MoGe | TODO | Monocular depth + alignment | NO → need |
| 3 | DINOv2 | TODO | Feature maps (32D) | NO → need |
| 4 | TAPNet | TODO | 2D/3D tracks | NO → need |
| - | SAM2 | SKIP | Foreground masks | YES (alpha) |
| - | DROID-SLAM | SKIP | Camera poses | YES (calibrated) |
| - | SEA-RAFT | SKIP (CUDA 12.2) | Optical flow | Optional |

### DUSt3R Installation

```bash
conda activate monofusion
cd ~/dev/MonoFusion/preproc/Dust3R
pip install -e .
# If CUDA compilation needed:
# TORCH_CUDA_ARCH_LIST="8.6" pip install -e .
```

### MoGe Installation

```bash
cd ~/dev/MonoFusion/preproc/MoGE
pip install -e .
```

### DINOv2 Features

```bash
# DINOv2 via torch.hub — no separate install needed
# XFormers for efficient attention:
pip install xformers==0.0.23+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### TAPNet (PyTorch port)

```bash
cd ~/dev/MonoFusion/preproc/tapnet
pip install -e .
# Download BootsTAPIR checkpoint
```

## Data Pipeline

```
M5t2 RGBA images (512×512, 6 cams, 3600 frames)
    │
    ├─ [unpack] → RGB images + alpha masks
    │
    ├─ [DUSt3R] → multi-view depth maps (with known poses as constraint)
    │
    ├─ [MoGe] → monocular depth → aligned with DUSt3R
    │
    ├─ [DINOv2] → 32D feature maps per frame
    │
    ├─ [TAPNet] → FG point tracks (mouse tracking)
    │
    └─ [MonoFusion training] → 4D Gaussian Splatting model
```

---

*MonoFusion Env Setup | 2026-03-27*
