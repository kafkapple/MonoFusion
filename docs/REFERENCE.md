# MonoFusion M5t2 — Technical Reference

> Consolidated from: core_architecture, data_compatibility, training_guide, theory/*, hardcoding_issues, env_setup
> Last: 2026-04-02

---

## 1. Pipeline

```
Raw Multi-cam RGB
  ├─ [SAM2] → FG masks (float32 ±1.0, erosion border=0)
  ├─ [OpenCV calibration] → K (3×3), w2c (4×4) per cam per frame
  ├─ [MoGe / Depth-Anything-V2] → relative depth (512×512, unnormalized)
  ├─ [DINOv2 ViT-S/14] → dense features (37×37×384 → upsample 512×512)
  └─ [RAFT multi-query] → point tracks [N,T,4] = [x,y,occ_logit,dist]

  → casual_dataset.py (camera_convention flag, per-cam loader)
  → train_m5t2.py → dance_glb.py (multi-cam synchronized training)
  → SceneModel: canonical 3DGS + SE(3) motion bases
  → gsplat rasterization → losses → optimizer
```

## 2. SE(3) Motion Bases

MonoFusion은 MLP deformation이 아닌 **SE(3) basis trajectories** 사용:
- `motion_bases.params.rots`: [K, T, 6] continuous 6D rotation
- `motion_bases.params.transls`: [K, T, 3] translation
- `fg.params.motion_coefs`: [N, K] per-Gaussian mixing weights
- Scene flow: `compute_poses_fg(t+1) - compute_poses_fg(t)` (analytical, no MLP forward)

Multi-view consistency: all cameras share K basis trajectories → optimizer fits single motion model explaining all views.

## 3. Data Format (M5t2)

### Dy_train_meta.json
```json
{
  "hw": [[512, 512], ...],        // (n_cams, 2)
  "k": [...],                     // (T, n_cams, 3, 3)
  "w2c": [...],                   // (T, n_cams, 4, 4)
  "camera_convention": "w2c"      // ⚠️ CRITICAL: "w2c"=real w2c, "c2w"=DUSt3R
}
```

### Directory Structure
```
m5t2_v5/
├── images/m5t2_cam{00-03}/    # RGB 512×512 PNG
├── masks/m5t2_cam{00-03}/     # .npz with dyn_mask (±1.0)
├── raw_moge_depth/            # .npy float32 relative depth
├── dinov2_features/           # .npy float16 (37×37→512×512)
├── tapir/ → tapir_raft/       # RAFT tracks (multi-query)
└── Dy_train_meta.json
```

### Naming Conventions
- Images/masks: 6-digit (`000000.png`), with 5-digit symlinks for MonoFusion compat
- Tracks: `{query_frame}_{target_frame}.npy` shape [N, 4]
- `dyn_mask.astype(bool)` 금지 → `raw > 0` 사용 (float -1.0 → True bug)

## 4. Training Configuration (V5j)

```yaml
# Key hyperparameters
total_steps: 300 epochs (4500 steps, 15 steps/epoch)
bg_gaussians: 32
opacity_reset: every 30 controls, reset_mult=1.5 (must > cull threshold)
stop_control: 80% of training
w_mask: 1.0
w_feat: 0.75
w_depth: 0.0  # depth scale unaligned → disabled
densify: 40-60% range, grad_threshold=0.00015, max_gaussians=100k
pca: 384d → 32d + L2 norm
motion_bases: K=10
```

### Training Command (gpu03)
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate monofusion
CC=x86_64-conda-linux-gnu-gcc \
CUDA_VISIBLE_DEVICES=X \
PYTHONPATH=~/dev/MonoFusion:~/dev/MonoFusion/preproc/Dust3R \
python ~/dev/MonoFusion/mouse_m5t2/train_m5t2.py [args]
```

## 5. Densification & Pruning

3DGS adaptive density control (from `flow3d/trainer.py`):
- **Clone**: small Gaussians (scale < threshold) with high gradient → duplicate
- **Split**: large Gaussians with high gradient → split into 2 smaller
- **Prune**: opacity < cull_threshold → remove
- **Opacity reset**: periodically reset all opacities (⚠️ value must exceed cull threshold)

MonoFusion specifics:
- FG and BG densified independently
- `xys_grad_norm` shape: gsplat returns `[C, N, 2]` (2D radii) → `reshape(-1,2).norm(dim=-1)`

## 6. Known Constraints

| Item | Detail |
|------|--------|
| **FG coverage** | Mouse = 2.1% of image → full PSNR ceiling ~9.2dB |
| **FG-only PSNR** | Use `eval_v5g_comprehensive.py` for meaningful metric |
| **Depth** | MoGe = relative, not metric → cross-cam scale inconsistent |
| **RAFT drift** | Accumulates over time → multi-query (5Q) mitigates |
| **TAPNet failure** | Mouse too fast (6px/frame) → 80% tracks occluded |
| **gsplat + gcc** | nvcc+cu118 needs gcc≤11 → `conda install gcc_linux-64=11` |
| **DINOv2 native** | ViT-S/14 → 37×37 patches → must upsample to image resolution |

## 7. Environment

| Env | Purpose | GPU |
|-----|---------|-----|
| `monofusion` (cu118) | Training + gsplat | A6000 (sm_86) |
| `monofusion_pytorch` (cu128) | Blackwell compat | RTX PRO 6000 |
| `monofusion_jax` | TAPNet only | A6000 only |

---

*MonoFusion M5t2 PoC | Technical Reference | 2026-04-02*
