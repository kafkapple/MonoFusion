# MonoFusion Core Architecture

> Fact-checked against codebase: `flow3d/`, `mouse_m5t2/`, `preproc/` (2026-03-29)

---

## Pipeline Overview

```
Raw Video (multi-cam RGB)
    │
    ▼
[SAM2] ─────────────────── per-frame FG binary masks (dyn_mask float32 ±1.0)
    │
    ▼
[DUSt3R] ────────────────── camera intrinsics K, extrinsics w2c, scene depth
    │
    ▼
[MoGe] ──────────────────── monocular metric depth (aligned to DUSt3R scale)
    │
    ▼
[DINOv2 (ViT-L/14)] ──────── per-frame dense features [H, W, 1024]
    │
    ▼
[TAPNet / BootsTAPIR] ─────── point tracks [N, T, 4] = [x, y, occ_logit, dist]
    │
    ▼
[MonoFusion Training] ──────── 4D Gaussian Splatting (3DGS + temporal deformation)
    │
    ▼
Novel view + time synthesis
```

---

## Component Roles

### 1. SAM2 — Foreground Segmentation
- **Output**: `dyn_mask` float32 in `[-1, 1]` per frame (`+1.0 = FG`, `-1.0 = BG`)
- **Critical convention**: Use `mask = raw > 0`, NOT `raw.astype(bool)` (−1.0 converts to True)
- **Role**: Provides FG region for track seeding and validation; separates dynamic from static scene

### 2. DUSt3R — Multi-View Geometry
- **Output**: Camera parameters `K` [3×3], `w2c` [4×4], image dimensions `hw`
- **Stored in**: `_raw_data/m5t2/trajectory/Dy_train_meta.json`
- **Role**: Bootstrap camera calibration without known camera rigs; global point cloud alignment

### 3. MoGe — Metric Depth
- **Output**: Per-frame depth maps `.npy` [H, W], metric scale aligned to DUSt3R
- **Stored in**: `aligned_moge_depth/m5t2/{cam}/depth/{frame}.npy`
- **Role**: Provides depth signal for 3D Gaussian initialization and depth regularization loss

### 4. DINOv2 — Dense Features
- **Model**: ViT-S/14 (dinov2_vits14), patch size 14, output dim 384
- **Output**: `[H', W', 384]` features (37×37 for 512×512 input)
- **Stored in**: `dinov2_features/{cam}/{frame}.npy`
- **Role**: Feature similarity loss (`w_feat`) to enforce view consistency beyond RGB

### 5. TAPNet / BootsTAPIR — Point Tracking
- **Output format**: `{query}_{target}.npy` shape `[N, 4]` = `[x, y, occlusion_logit, expected_dist]`
- **Occlusion convention**: `logit ≤ 0.0` → visible, `logit > 0.0` → occluded (sigmoid)
- **Stored in**: `tapir/{cam}/`
- **Known issue**: For fast-moving foreground objects (e.g., mouse), marks FG-seeded tracks
  as occluded (80-89% at frame 30). Visible tracks shift to background.
- **Fix (Option B)**: RAFT optical flow + per-frame SAM2 mask validation → `tapir_raft/{cam}/`

---

## Training Loss Decomposition

```python
loss = w_track * L_track       # Point track consistency (ESSENTIAL)
     + w_feat  * L_feat        # DINOv2 feature similarity
     + w_depth_reg * L_depth   # MoGe depth regularization
```

| Loss | Default Weight | Function |
|------|---------------|----------|
| `L_track` | `w_track = 2.0` | 2D re-projection of tracked 4D Gaussians vs. track positions |
| `L_feat` | `w_feat = 1.5` | Cosine/L2 distance between rendered DINOv2 and precomputed features |
| `L_depth_reg` | `w_depth_reg = 0.0` | MoGe depth as soft constraint on Gaussian z-positions |

**`L_track` is non-negotiable** — MonoFusion's 4D deformation field is supervised primarily
through track consistency. Without valid foreground tracks, Gaussians learn background motion.

---

## Data Format Conventions

### File Structure (per-camera, per-frame)
```
m5t2_poc/
├── images/{cam}/{######}.png          # 6-digit frame names (e.g., 000000.png)
├── masks/{cam}/{######}.npz           # key: "dyn_mask", float32 ±1.0
├── aligned_moge_depth/m5t2/{cam}/depth/{######}.npy
├── dinov2_features/{cam}/{######}.npy
├── tapir/{cam}/{query}_{target}.npy   # [N, 4] original TAPNet tracks
└── tapir_raft/{cam}/{query}_{target}.npy  # [N, 4] RAFT+mask tracks (Option B)
```

### Track Array Format
```
track[N, 0] = x  (pixel column, float32)
track[N, 1] = y  (pixel row, float32)
track[N, 2] = occlusion_logit  (float32; ≤0 = visible, >0 = occluded)
track[N, 3] = expected_dist    (float32; 0.0 = neutral, used by MonoFusion depth coupling)
```

### Camera Meta JSON
```json
{
  "hw":  [[H, W], ...],          // per-camera image size
  "k":   [[[K_cam0], ...], ...], // intrinsics per frame per cam
  "w2c": [[[w2c_cam0], ...], ...]// world-to-camera 4x4 per frame per cam
}
```

---

## Tracking Fix Strategy (m5t2 PoC)

### Root Cause
BootsTAPIR was designed for slow-motion or near-static foreground. A mouse at ~1.5 m/s
generates large between-frame motion (~6px/frame at 1080p, 30fps). TAPNet responds by
marking these fast-moving points occluded, leaving only slow-moving background tracks visible.

### Option A — Post-filter (implemented)
```
TAPNet tracks → apply per-frame SAM2 mask → BG tracks set to occ_logit=+10
```
- Result: 0% visible at mid-frames for cam01 (correct — all TAPNet-visible tracks were BG)
- Not useful as MonoFusion input (too little signal)

### Option B — RAFT + Mask (implemented, recommended)
```
Original RGB → RAFT-small → consecutive flows [T-1, H, W, 2]
SAM2 query mask (F0) → seed N points in FG
Chain flows forward + backward → propagate positions
Per-frame SAM2 mask → mark out-of-mask points as occluded
```
- Result: cam00=45%, cam01=18%, cam02=0% (mask absent=mouse moved), cam03=15% at F30
- All visible tracks confirmed inside SAM2 FG region
- Scripts: `mouse_m5t2/scripts/generate_raft_tracks.py`

### RAFT Preprocessing Note (Critical Bug)
```python
# WRONG: float32 [0,255] causes ~10x flow magnification
frames = imgs.astype(np.float32)  # ❌

# CORRECT: uint8 + official OpticalFlow transform
imgs.append(torch.from_numpy(img).permute(2, 0, 1))  # uint8
src_t, dst_t = transform(src, dst)  # weights.transforms() → [-1, 1] float32
```

---

## 4D Gaussian Splatting (MonoFusion)

MonoFusion extends 3D Gaussian Splatting (3DGS) with **SE(3) motion bases** — NOT an MLP deformation field.

> See `docs/theory/monofusion_architecture.md` for full technical details.

```
Canonical Gaussians {μ_i, Σ_i, c_i, α_i}        (G Gaussians)
    │
    ▼  motion_coefs_i (G, K) — softmax blending weights
    │
    ▼  SE(3) Bases: rots(K,T,6), transls(K,T,3)   (K shared global bases)
    │     ↓ compute_transforms(t, coefs_i) → SE3_i(t)
    ▼
Time-t Gaussian i: μ_i(t) = SE3_i(t) @ [μ_i_canonical; 1]
    │
    ▼  Differentiable Rasterizer (gsplat) → Rendered image
    │
    ▼  Compare vs: RGB, DINOv2 features, track positions, MoGe depth
```

Track supervision: if rendered Gaussian `i` at time `t` doesn't match tracked pixel `(x_t, y_t)`,
the SE(3) basis parameters are corrected. All cameras share the **same K bases** → multi-view
consistency is structurally enforced, not regularized.

---

## Known Issues & Mitigations

| Issue | Symptom | Mitigation |
|-------|---------|------------|
| TAPNet FG tracking failure | 0% visible FG tracks at F>10 | Option B: RAFT+mask |
| RAFT drift accumulation | Tracks walk off mouse by F59 | Per-frame mask validation; windowed reinit (planned) |
| cam02 total occlusion | 0% visible F30-45 | Expected — mouse leaves cam02 FOV. MonoFusion handles missing cameras |
| dyn_mask sign convention | `.astype(bool)` → -1.0 treated as FG | Always use `raw > 0` |

---

## Training Integration Details

### Required Data Check (`train_m5t2.py:142`)
```python
required = ["images", "masks", "dinov2_features", "aligned_moge_depth", "tapir"]
for d in required:
    path = data_root / d
    if not path.exists():
        print(f"ERROR: Missing {path}")
        sys.exit(1)
```
**⚠️ Track directory is hardcoded as `"tapir"`**. To use RAFT tracks:
```bash
# Option 1: symlink (non-destructive)
ln -sfn /node_data/joon/data/monofusion/m5t2_poc/tapir_raft \
         /node_data/joon/data/monofusion/m5t2_poc/tapir

# Option 2: modify train_m5t2.py to accept --track_dir argument
```

### Track Loading (`casual_dataset.py:156-160`)
```python
for name in ("tapir", track_2d_type, "bootstapir"):
    if name and name not in track_candidates:
        track_candidates.append(name)
self.tracks_dir = _resolve_seq_path(track_candidates, "track directory")
```
RAFT is not a recognized track type — MonoFusion only knows `"tapir"` and `"bootstapir"`.
Our RAFT tracks use the **same [N,4] format**, so a symlink or rename is sufficient.

### Occlusion → Visibility Conversion (`data/utils.py:53-66`)
```python
def parse_tapir_track_info(occlusions, expected_dist):
    visiblility = 1 - F.sigmoid(occlusions)
    confidence = 1 - F.sigmoid(expected_dist)
    valid_visible = visiblility * confidence > 0.5
    valid_invisible = (1 - visiblility) * confidence > 0.5
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence
```
- `occ_logit = -2.0` (visible) → sigmoid(-2) = 0.12 → visibility = 0.88 ✅
- `occ_logit = +10.0` (occluded) → sigmoid(10) ≈ 1.0 → visibility ≈ 0.0 ✅

### Loss Weights (confirmed from `configs.py`)
| Loss | Weight | Disableable |
|------|--------|-------------|
| `w_track` | 2.0 | ❌ Essential |
| `w_feat` | 1.5 | ✅ (set to 0.0) |
| `w_depth_reg` | **0.0** | Already disabled |
| `w_depth_const` | **0.0** | Already disabled |

---

---

↑ MOC: `docs/README.md` | ↔ Related: `theory/monofusion_architecture.md`, `theory/scene_flow_and_tracking.md`, `training_guide.md`

*MonoFusion M5t2 PoC — Core Architecture | 2026-03-29*
