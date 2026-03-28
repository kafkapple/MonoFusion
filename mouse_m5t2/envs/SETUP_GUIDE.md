# MonoFusion Multi-Environment Setup Guide

## Overview

MonoFusion requires 2-3 isolated conda environments due to framework conflicts
(JAX vs PyTorch) and GPU architecture compatibility.

## GPU Architecture on gpu03

| GPU | Type | Architecture | Compatible Envs |
|-----|------|-------------|-----------------|
| 0-3 | RTX PRO 6000 Blackwell | sm_120 | `monofusion_pytorch` only |
| 4-7 | RTX A6000 | sm_86 | Both envs |

## Environments

| Env | CUDA | Framework | Components | GPU |
|-----|------|-----------|------------|-----|
| `monofusion_pytorch` | 12.8 (cu128) | PyTorch 2.11 | DUSt3R, MoGe, DINOv2, gsplat, training | GPU 0-7 (all) |
| `monofusion_jax` | 12.x | JAX 0.4.30 | TAPNet/BootsTAPIR | **GPU 4-7 only** (A6000) |
| `monofusion_raft` | 12.2 | PyTorch | SEA-RAFT (optional) | — |

## Installation

```bash
# 0. Data format preparation (run ONCE before pipeline)
conda run -n monofusion_pytorch python mouse_m5t2/scripts/prepare_data_format.py \
    --data_root /node_data/joon/data/monofusion/m5t2_poc --seq_name m5t2

# 1. PyTorch environment (main — install first)
conda env create -f monofusion_pytorch.yml
conda activate monofusion_pytorch
# Verify on Blackwell GPU:
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_name(0))"
# Verify gsplat:
python -c "import gsplat; print(f'gsplat {gsplat.__version__}')"

# 2. JAX environment (TAPNet)
conda env create -f monofusion_jax.yml
conda activate monofusion_jax
# Verify JAX GPU (MUST use A6000, NOT Blackwell):
CUDA_VISIBLE_DEVICES=7 python -c "import jax; print(jax.devices())"
# Download BootsTAPIR checkpoint:
mkdir -p ~/dev/MonoFusion/checkpoints
wget -O ~/dev/MonoFusion/checkpoints/bootstapir_checkpoint_v2.npy \
    https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy

# 3. SEA-RAFT environment (optional)
conda env create -f monofusion_raft.yml
```

## Pipeline Execution

```bash
DATA=/node_data/joon/data/monofusion/m5t2_poc
PREPROC=~/dev/MonoFusion/preproc

# Stage 1: DUSt3R (env: monofusion_pytorch, GPU 0-3)
conda activate monofusion_pytorch
cd $PREPROC && CUDA_VISIBLE_DEVICES=0 python compute_dust3r.py \
    --seq_name m5t2 \
    --image_root $DATA/images --mask_root $DATA/masks \
    --raw_root $DATA/_raw_data --processing_root $DATA \
    --frame_step 1 --frame_count 60

# Stage 2: MoGe raw depth (env: monofusion_pytorch, GPU 1)
for cam in 00 01 02 03; do
    CUDA_VISIBLE_DEVICES=1 python compute_moge.py \
        --img_dir $DATA/images/m5t2_cam${cam} \
        --out_dir $DATA/raw_moge_depth/m5t2_cam${cam}
done

# Stage 3: MoGe alignment (CPU, after Stage 1+2)
python compute_aligned_moge_depth.py --seq_name m5t2 \
    --raw_moge_root $DATA/raw_moge_depth --dust3r_root $DATA/dust3r \
    --mask_root $DATA/masks --image_root $DATA/images \
    --output_root $DATA/aligned_moge_depth --raw_root $DATA/_raw_data

# Stage 4: TAPNet (env: monofusion_jax, GPU 4-7 ONLY)
conda activate monofusion_jax
for cam in 00 01 02 03; do
    CUDA_VISIBLE_DEVICES=7 python compute_tracks_jax.py \
        --image_dir $DATA/images/m5t2_cam${cam} \
        --mask_dir $DATA/masks/m5t2_cam${cam} \
        --out_dir $DATA/tapir/m5t2_cam${cam} \
        --model_type bootstapir --ckpt_dir ~/dev/MonoFusion/checkpoints
done

# Stage 5: DINOv2 features (env: monofusion_pytorch, GPU 2-3)
conda activate monofusion_pytorch
for cam in 00 01 02 03; do
    CUDA_VISIBLE_DEVICES=3 python compute_dinofeatures.py \
        --img_dir $DATA/images/m5t2_cam${cam} \
        --out_dir $DATA/dinov2_features/m5t2_cam${cam} \
        --mask_dir $DATA/masks/m5t2_cam${cam} \
        --frame_step 1 --output_height 512 --output_width 512
done

# Stage 6: Training (env: monofusion_pytorch)
# conda run -n monofusion_pytorch python mouse_m5t2/train_m5t2.py
```

## Data Exchange

- All intermediate data: `.npy` files (numpy, cross-env compatible)
- NumPy version: **1.26.4** in ALL environments (mandatory)
- No torch tensors or JAX arrays across env boundaries
- See `docs/data_compatibility.md` for format conversion details

## Verification Checklist

```bash
# Check all envs
for env in monofusion_jax monofusion_pytorch; do
    echo "=== $env ==="
    conda run -n $env python -c "import numpy; print(f'numpy {numpy.__version__}')"
done

# JAX GPU check (A6000 only!)
CUDA_VISIBLE_DEVICES=7 conda run -n monofusion_jax python -c "import jax; print(jax.devices())"

# PyTorch Blackwell check
CUDA_VISIBLE_DEVICES=0 conda run -n monofusion_pytorch python -c \
    "import torch; print(f'{torch.__version__} on {torch.cuda.get_device_name(0)}')"

# gsplat check
conda run -n monofusion_pytorch python -c "import gsplat; print(gsplat.__version__)"
```

---

*MonoFusion Multi-Env Setup Guide | 2026-03-27 (updated: cu128 + Blackwell)*
