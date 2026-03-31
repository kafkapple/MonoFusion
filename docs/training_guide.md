# Training Guide — M5t2 PoC

> Pre-training checklist, training setup, monitoring, and post-training analysis.
> Last updated: 2026-03-29 (after MoA deliberation on training risks)

---

## Status

```
Preprocessing: ✅ COMPLETE
  - SAM2 masks:    ✅  (RGBA alpha → float32 ±1.0)
  - DUSt3R K/w2c:  ✅  (Dy_train_meta.json)
  - MoGe depth:    ✅  (aligned_moge_depth/)
  - DINOv2 feats:  ✅  (dinov2_features/)
  - RAFT tracks:   ✅  (tapir/ → symlink → tapir_raft/)

Training: ⏳ NOT STARTED
```

---

## Pre-Training Checklist (~1 hour)

Before launching the first training run, implement these 4 safeguards identified from MoA deliberation:

### 1. Visibility-Weighted Track Loss

Replace hard binary masking with sigmoid-weighted loss:

```python
# In: flow3d/losses.py or wherever L_track is computed
# Before: loss = w_track * L2(proj_pt - track_xy)
# After:
visibility_weight = torch.sigmoid(-2.0 * occ_logit)  # [N]: 0.27 to ~1.0
loss_track = w_track * (visibility_weight * L2(proj_pt - track_xy)).mean()
```

Why: Hard cutoff at occ_logit=0 is noisy (SAM2 mask jitter ±2px). Soft weighting degrades low-confidence tracks gracefully.

### 2. Mask Boundary Dilation (2px exclusion zone)

When seeding query points and during per-frame validation, exclude points within 2px of the FG mask boundary:

```python
from scipy.ndimage import binary_erosion, binary_dilation

def get_reliable_mask_region(mask_binary: np.ndarray) -> np.ndarray:
    """Erode mask by 2px to exclude boundary noise."""
    eroded = binary_erosion(mask_binary, iterations=2)
    return eroded
```

Why: Fur/background boundary has 1-2px classification uncertainty. Tracks seeded at boundary may be incorrectly validated as FG/BG at different frames → oscillating training signal.

### 3. Deformation MLP Bias Initialization

Initialize the final layer bias of the deformation MLP to near-zero:

```python
# In: flow3d/deformation_field.py (or equivalent)
model.deform_mlp.output_layer.bias.data.fill_(-0.01)
```

Why: Random initialization can produce large initial deformations (>10px), causing the track loss to pull Gaussians in wrong directions during early training. Near-zero init ensures F0 deformation ≈ 0, and the field learns incrementally.

### 4. Per-Camera Track Weighting

Weight track loss contributions based on camera visibility:

```python
# Based on M5t2 RAFT results (F30 visibility):
CAMERA_WEIGHTS = {
    "m5t2_cam00": 1.0,   # 45% visible at F30
    "m5t2_cam01": 0.6,   # 18% visible
    "m5t2_cam02": 0.0,   # 0% visible (mouse left FOV)
    "m5t2_cam03": 0.5,   # 15% visible
}
```

Why: cam02 has 0% FG tracks and will only contribute noise to the track loss. Disabling it prevents its background tracks from pulling Gaussians toward the wrong 3D positions.

---

## Training Setup

### Environment
```bash
ssh gpu03
source /home/joon/anaconda3/etc/profile.d/conda.sh && conda activate monofusion_pytorch
cd /home/joon/dev/MonoFusion
```

### Data Root
```
/node_data/joon/data/monofusion/m5t2_poc/
├── images/m5t2_cam{00-03}/{######}.png
├── masks/m5t2_cam{00-03}/{######}.npz         # dyn_mask float32 ±1.0
├── aligned_moge_depth/m5t2/m5t2_cam{00-03}/depth/{######}.npy
├── dinov2_features/m5t2_cam{00-03}/{######}.npy
└── tapir/ → tapir_raft/ (symlink)             # RAFT multi-query tracks
    └── m5t2_cam{00-03}/{query}_{target}.npy    # 300 files per camera
```

### Verify symlink before training
```bash
ls -la /node_data/joon/data/monofusion/m5t2_poc/tapir
# Should show: tapir -> /node_data/joon/data/monofusion/m5t2_poc/tapir_raft
```

### Launch Training (mf_001)
```bash
# Time-box: 4-6 hours. Monitor: see section below.
CUDA_VISIBLE_DEVICES=4 python mouse_m5t2/train_m5t2.py \
    --data_root /node_data/joon/data/monofusion/m5t2_poc \
    --result_dir /node_data/joon/data/monofusion/m5t2_poc/results/mf_001 \
    --num_frames 60 \
    --num_fg 5000 \
    --num_bg 10000 \
    --max_steps 5000 \
    2>&1 | tee /node_data/joon/data/monofusion/m5t2_poc/logs/mf_001.log
```

---

## Training Monitoring (Every ~500 steps)

### 3 Critical Metrics to Log

**1. Multi-View Reprojection Error**
```python
# For each visible track: triangulate from all cameras → reproject to each camera → measure error
# Expected: < 2px at convergence
# Danger: growing error = multi-view inconsistency (geometric tearing)
```

**2. Trajectory Smoothness (Jitter)**
```python
# Second-order temporal derivative of Gaussian positions
jitter = ||mu(t+1) - 2*mu(t) + mu(t-1)||_2
# Expected: decreasing as training progresses
# Danger: spikes or high steady-state = track noise dominating loss
```

**3. Visibility Flip Rate**
```python
# Count FG/BG label changes per tracked point over 60 frames
flip_rate = mean(sum(occ[t] != occ[t-1] for t in range(1, T)) for each track)
# Expected: < 0.05 (< 5% of frames have state flip)
# Danger: > 0.15 = mask instability is poisoning the training signal
```

### Early Stop Conditions
- L_track_per_point > 5.0 and not decreasing after step 500 → stop, check track quality
- Multi-view reprojection error > 5px at step 1000 → stop, check epipolar consistency
- Training NaN → restart with lower lr (1e-4 → 5e-5)

### Log Inspection
```bash
# Follow training log
tail -f /node_data/joon/data/monofusion/m5t2_poc/logs/mf_001.log | grep -E "step|loss|reprojection"
```

---

## Post-Training Analysis

### 3D Scene Flow Visualization
After training converges, visualize the learned deformation field:

```bash
# See docs/theory/scene_flow_and_tracking.md §5 for full code
python mouse_m5t2/scripts/viz_scene_flow.py \
    --ckpt /node_data/.../results/mf_001/checkpoint_final.pth \
    --output_dir /node_data/.../viz/scene_flow/
```

Expected outputs:
- `scene_flow_mag_f30.png` — Gaussians colored by 3D flow magnitude at F30
- `scene_flow_quiver.html` — Interactive 3D vector field (Open3D or Plotly)
- `scene_flow_trails.mp4` — Trajectory trails animation
- `flow_2d_projection_verify.png` — 3D flow projected to 2D vs RAFT comparison

### Quality Metrics (FG-Masked)
```
Primary metrics (mouse region only):
  PSNR_FG  ↑  (target: > 22 dB for PoC)
  SSIM_FG  ↑  (target: > 0.7)
  LPIPS_FG ↓  (target: < 0.3)
  Mask IoU ↑  (target: > 0.8)
```

---

## Improvement Roadmap (Post-Baseline)

### Tier 1 (If baseline shows track-related artifacts, 1-2 weeks)
1. **CoTracker** — Replace RAFT with Meta's long-range tracker (+10-20% expected)
2. **Temporal smoothness regularization** — Add `L_smooth = ||D(μ, t+1) - D(μ, t)||²` (weight 0.1-0.2)
3. **Epipolar regularization** — Cross-camera consistency loss (weight 0.05)

### Tier 2 (If Tier 1 insufficient, post-training refinement)
4. **DINOv2 post-training smoothing** — For Gaussians with high residual drift, use temporal median + DINOv2 feature-space correction (NOT pre-training — DINOv2 14px patches degrade RAFT 0.5px precision)

### Tier 3 (Research extension)
5. **Adaptive multi-scale query density** — High-motion regions (paws) get 2× query density
6. **4-view triangulation** — Direct 3D track supervision (eliminates depth ambiguity)

---

↑ MOC: `docs/README.md` | ↔ Related: `core_architecture.md`, `theory/monofusion_architecture.md`, `theory/scene_flow_and_tracking.md`

*Training Guide | MonoFusion M5t2 PoC | 2026-03-29*
