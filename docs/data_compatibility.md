# Data Format Compatibility Guide

Mapping between m5t2 mouse dataset format and MonoFusion's expected format.
Every conversion is done via symlinks or format translation — no pipeline code modified.

↑ [docs/README.md](README.md) | ↔ [SETUP_GUIDE](../mouse_m5t2/envs/SETUP_GUIDE.md) | ↓ [prepare_data_format.py](../mouse_m5t2/scripts/prepare_data_format.py)

## Overview

| Aspect | m5t2 Mouse Format | MonoFusion Expected | Solution |
|--------|-------------------|---------------------|----------|
| **Image naming** | `000000.png` (6-digit) | `00000.png` (5-digit) | Symlink: `00000.png → 000000.png` |
| **Mask naming** | `000000.npz` (6-digit) | `dyn_mask_00000.npz` | Symlink: `dyn_mask_00000.npz → 000000.npz` |
| **Mask directory** | `masks/` | `sam_v2_dyn_mask/` | Symlink: `sam_v2_dyn_mask → masks` |
| **Camera dir naming** | `m5t2_cam00/` | `m5t2_undist_cam00/` | Symlink in images/, masks/, raw_moge_depth/ |
| **MoGe depth naming** | `000000.npy` (6-digit) | `00000.npy` (5-digit) | Symlink: `00000.npy → 000000.npy` in depth/ |
| **Symlink duplication** | N/A | N/A | Image symlinks cause 120 frames seen (60 real + 60 dup). Affects MoGe, DINOv2, TAPNet — doubles compute but harmless for PoC. **Optimization**: point glob-based scripts to dirs without 5-digit symlinks. |
| **Calibration** | `Dy_train_meta.json` | `gopro_calibs.csv` | Format conversion script |
| **Frame start index** | N/A | `timestep.txt` | Generated (value: 0) |

## Calibration Format Conversion

### Source: `Dy_train_meta.json`
```json
{
  "hw": [[512, 512], ...],           // (4, 2) — height, width per camera
  "k":  [[[fx,0,cx],[0,fy,cy],...]], // (60, 4, 3, 3) — per-frame intrinsics
  "w2c": [[[[R|t],[0001]],...]]      // (60, 4, 4, 4) — per-frame world-to-camera
}
```

### Target: `gopro_calibs.csv`
```
cam_uid, tx/ty/tz_world_cam, qw/qx/qy/qz_world_cam, image_width/height, intrinsics_0-3
```

### Conversion Logic
1. `w2c[0]` (frame 0) → invert → `c2w` (camera-to-world pose)
2. Extract translation `(tx, ty, tz)` from `c2w[:3, 3]`
3. Extract quaternion `(qw, qx, qy, qz)` from `c2w[:3, :3]` via `scipy.spatial.transform.Rotation`
4. Extract `fx, fy, cx, cy` from `K[0]` (frame 0 intrinsics)
5. `image_width/height` from `hw` array

### Key Assumptions
- **Static cameras**: DUSt3R uses fixed camera rig poses. We use frame 0's w2c matrices.
  Our multi-camera rig has static extrinsics, so this is valid.
- **Quaternion convention**: DUSt3R's `org_utils.py` uses scipy `(qx, qy, qz, qw)` for input,
  stores as `(qw, qx, qy, qz)` in CSV. Our conversion matches this.
- **Intrinsic scaling**: DUSt3R's `get_preset_any` applies `ratio = size/3840`. Our images are
  512×512 (not 3840×2160 GoPro), so DUSt3R will rescale intrinsics. This is handled by the
  `image_width/height` columns in the CSV.

## Directory Structure After Preparation

```
/node_data/joon/data/monofusion/m5t2_poc/
├── images/
│   ├── m5t2/                    # Base sequence (not per-camera)
│   ├── m5t2_cam00/              # Camera 0: 60 frames
│   │   ├── 000000.png           # Original (6-digit)
│   │   ├── 00000.png → 000000.png  # Symlink (5-digit)
│   │   └── ...
│   ├── m5t2_cam01/
│   ├── m5t2_cam02/
│   └── m5t2_cam03/
├── masks/ (= sam_v2_dyn_mask via symlink)
│   ├── m5t2_cam00/
│   │   ├── 000000.npz           # Original
│   │   ├── dyn_mask_00000.npz → 000000.npz  # Symlink
│   │   └── ...
│   └── ...
├── sam_v2_dyn_mask → masks      # Directory symlink
├── _raw_data/
│   └── m5t2/
│       ├── trajectory/
│       │   ├── Dy_train_meta.json          # Original (source)
│       │   ├── Dy_train_meta_cam00.json    # Per-camera (source)
│       │   └── gopro_calibs.csv            # Generated
│       └── frame_aligned_videos/
│           └── timestep.txt                # Generated (value: 0)
├── dust3r/          # Output: DUSt3R
├── raw_moge_depth/  # Output: MoGe raw
├── aligned_moge_depth/  # Output: MoGe aligned
├── tapir/           # Output: TAPNet tracks
└── dinov2_features/ # Output: DINOv2
```

## Pipeline Stage I/O

### Stage 1: DUSt3R
- **Input**: `images/{seq_name}_cam*/`, `masks/{seq_name}_cam*/dyn_mask_*.npz`, `gopro_calibs.csv`
- **Output**: `dust3r/{seq_name}/` — depth maps, confidence, point clouds per camera
- **Env**: `monofusion_pytorch`
- **GPU Compat**: Blackwell sm_120 → PyTorch 2.11+cu128

### Stage 2: MoGe Raw Depth
- **Input**: `images/{seq_name}_cam*/`
- **Output**: `raw_moge_depth/{seq_name}_cam*/depth/*.npy`
- **Env**: `monofusion_pytorch`

### Stage 3: MoGe Alignment
- **Input**: `raw_moge_depth/`, `dust3r/`, `gopro_calibs.csv`
- **Output**: `aligned_moge_depth/{seq_name}/`
- **Env**: `monofusion_pytorch`

### Stage 4: TAPNet (BootsTAPIR)
- **Input**: `images/{seq_name}_cam*/` (6-digit OK — uses glob), `masks/{seq_name}_cam*/` (6-digit OK)
- **Output**: `tapir/{seq_name}_cam*/` — pairwise track .npy files
- **Env**: `monofusion_jax` (JAX + CUDA 12)
- **Checkpoint**: `checkpoints/bootstapir_checkpoint_v2.npy`
- **NOTE**: TAPNet uses `glob("*")` for frames, so 6-digit naming works directly. No symlinks needed.

### Stage 5: DINOv2 Features
- **Input**: `images/{seq_name}_cam*/`
- **Output**: `dinov2_features/{seq_name}_cam*/`
- **Env**: `monofusion_pytorch`
- **NOTE**: Uses direct directory listing, so 6-digit naming works. Mask prefix configurable.

## Runtime Dependencies (discovered during setup)

| Env | Package | Required By | Notes |
|-----|---------|-------------|-------|
| `monofusion_jax` | `tensorflow-cpu` | `tapnet.__init__` | tapnet unconditionally imports `tensorflow` |
| `monofusion_jax` | `tensorflow-datasets` | `tapnet.tapvid` | Same unconditional import chain |
| `monofusion_pytorch` | `imageio` | DUSt3R `org_utils.py` | Not in requirements.txt |
| `monofusion_pytorch` | `pandas` | DUSt3R `org_utils.py` (CSV parsing) | Not in requirements.txt |

## GPU Compatibility Matrix

| GPU | Architecture | CUDA CC | PyTorch | Notes |
|-----|-------------|---------|---------|-------|
| RTX PRO 6000 Blackwell (GPU 0-3) | sm_120 | 13.0 | 2.11+cu128 | `monofusion_pytorch` only |
| RTX A6000 (GPU 4-7) | sm_86 | 13.0 | 2.11+cu128 or 2.1+cu118 | Both envs work |

**IMPORTANT**:
- `monofusion` (old env, cu118) does NOT work on Blackwell GPUs.
- `monofusion_jax` (JAX 0.4.30, sm_90a PTX) does NOT work on Blackwell GPUs.
  JAX ptxas cannot compile sm_90a → sm_120. **TAPNet must run on A6000 (GPU 4-7).**
- Blackwell-compatible JAX would require JAX 0.5+ with CUDA 12.8+ (not yet available as of 2026-03-27).

## Preparation Script

```bash
# Run once before pipeline execution
conda run -n monofusion_pytorch python mouse_m5t2/scripts/prepare_data_format.py \
    --data_root /node_data/joon/data/monofusion/m5t2_poc \
    --seq_name m5t2
```

Creates all symlinks and format conversions. Idempotent — safe to re-run.

---

*MonoFusion Data Compatibility Guide | 2026-03-27*
