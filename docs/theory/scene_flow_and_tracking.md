# Scene Flow & Point Tracking — Theory

> Fact-checked against codebase + MoA deliberation: 2026-03-29

---

## 1. What is 3D Scene Flow?

**Definition**: Scene flow is a dense 3D vector field φ(X, t) that describes how every point X ∈ ℝ³ in a scene moves between time t and t+1.

```
φ(X, t) = X(t+1) − X(t)     [3D displacement per point per frame]
```

Compare with related concepts:

| Concept | Dimension | Output | Captures |
|---------|-----------|--------|---------|
| Optical flow | 2D image | (u, v) per pixel | Apparent motion projected onto image plane |
| Depth | 2D → 3D | Z per pixel | Static 3D structure |
| Scene flow | 3D + time | (dx, dy, dz) per 3D point | Full 3D motion |
| 4D Gaussian Splatting | 3D + time | Gaussian trajectories | Scene flow encoded as deformation field |

**Key insight**: Optical flow is a 2D *projection* of scene flow. Given camera intrinsics K and camera-space scene flow φ_c, the optical flow is:

```
(u, v) ≈ K · [φ_c / Z]     (approximate, valid for small displacements)
```

MonoFusion inverts this: supervised by 2D optical flow → learns 3D scene flow.

---

## 2. 4D Gaussian Splatting as Scene Flow

MonoFusion represents scene flow via **SE(3) motion bases** — NOT an MLP deformation field.

> ⚠️ Common misconception: MonoFusion is often described as having a "deformation MLP" (like D-NeRF).
> Reality: It uses K explicit SE(3) basis trajectories, shared globally across all Gaussians.
> See `docs/theory/monofusion_architecture.md` for full technical details.

### Architecture

```
Canonical Gaussians {μ_i, Σ_i, c_i, α_i}    (G Gaussians, learned 3D structure)
        │
        ▼  motion_coefs_i [K]                  (per-Gaussian softmax blending weights)
        │
        ▼  SE(3) Bases: rots[K,T,6], transls[K,T,3]   (K global basis trajectories)
        │     ↓ compute_transforms(t, coefs_i) → SE3_combined_i(t)
        ▼
Time-t Gaussian i: μ_i(t) = SE3_combined_i(t) @ [μ_i_canonical; 1]
        │
        ▼
Differentiable Rasterizer (gsplat)
        │
        ▼
Rendered image at (camera, time t)
```

**Scene flow of Gaussian i at time t**:
```
φ_i(t) = μ_i(t+1) − μ_i(t)
```

This is analytically exact — no MLP forward pass needed; just call `compute_poses_fg(t)` and `compute_poses_fg(t+1)`.

### Supervision Signal

2D track supervision closes the loop:

```
π(μ_i(t)) ≈ track_xy_i(t)
```

where π = camera projection. If Gaussian i's position at time t doesn't match the tracked pixel `(x_t, y_t)`, the SE(3) bases are corrected.

**Why motion bases matter for multi-view consistency**:
- Motion bases are **global** (not per-camera) → all cameras share the same K bases
- RAFT tracks from cam00/cam01/cam03 all supervise the same SE(3) parameters
- Multi-view consistency is **structurally guaranteed** (not just regularized)

---

## 3. Point Tracking Methods Compared

### 3.1 TAPNet / BootsTAPIR (Original MonoFusion)

**Architecture**: Transformer-based, trained on TAP-Vid benchmark. Designed for long-range, occlusion-aware point tracking.

**Occlusion convention** (critical for MonoFusion):
```
track[N, 2] = occlusion_logit
  occ_logit ≤ 0.0  →  visible   (sigmoid ≈ 0.0-0.5)
  occ_logit > 0.0   →  occluded  (sigmoid > 0.5)
```

**M5t2 failure analysis**:
- Mouse moves ~6px/frame at 1080p, 30fps
- TAPNet was trained on natural videos with slow-moving foreground (average 0.5-2px/frame)
- Fast motion triggers conservative occlusion prediction: 80-89% FG tracks marked occluded at F30
- Remaining "visible" tracks are all background (slow-moving)
- **Root cause**: TAPNet's occlusion prior is wrong for fast small objects

### 3.2 RAFT Optical Flow (Option B — Working)

**Architecture**: Recurrent All-Pairs Field Transforms. Designed for dense, frame-to-frame optical flow estimation.

**Key difference from TAPNet**: RAFT does NOT model occlusion. It outputs a flow vector `(u, v)` for every pixel, regardless of whether that pixel is occluded. This is its strength for M5t2: it produces motion estimates even when TAPNet would give up.

**Preprocessing requirement (critical)**:
```python
# WRONG: float32 [0, 255] → 10x flow magnification
frames = imgs.astype(np.float32)   # ❌

# CORRECT: uint8 + official transform
imgs = [torch.from_numpy(img).permute(2, 0, 1)]  # uint8
transform = Raft_Small_Weights.DEFAULT.transforms()
src_t, dst_t = transform(src_uint8, dst_uint8)  # → float32 [-1, 1]
```

**Track generation strategy**:
1. Seed N query points from SAM2 FG mask at query frame q
2. Compute consecutive RAFT flows: (T-1) frame pairs
3. Chain flows forward (q → T-1) and backward (q → 0)
4. Validate per-frame: points outside SAM2 FG mask → occ_logit = +10.0

**Result on M5t2**:
```
cam00: F30 = 45% visible (230/512 pts)
cam01: F30 = 18% visible  (92/512 pts)
cam02: F30 =  0% visible  (mouse left FOV — correct)
cam03: F30 = 15% visible  (77/512 pts)
```

### 3.3 Multi-Query RAFT (Current Best)

**Motivation**: Single-query RAFT accumulates drift. With query at F0 only, tracks drifted by F40-59 cover fewer FG pixels.

**Strategy**: Multiple query frames [0, 15, 30, 45, 59] each seed 512 new points. Each produces its own set of tracks that propagate forward + backward.

```
Query F0:  512 pts → 60 target files per camera
Query F15: 512 pts → 60 target files per camera
...
Query F59: 512 pts → 60 target files per camera
──────────────────────────────────────────────
Total: 5 queries × 60 targets = 300 files per camera
```

**Coverage improvement**:
```
Single-query:  cam01 F30 = 18% (92/512 pts)
Multi-query:   cam01 F30 ≈ 30% (154/2560 pts, best across 5 queries)
```

**MonoFusion compatibility**: `casual_dataset.py` supports multiple `query_idcs`. The dataset loader selects, for each target frame, all tracks where `query_idx` appears in the loaded track files. Multiple queries = more supervised points per frame.

### 3.4 CoTracker (Future — Not Implemented)

Meta's CoTracker (2023) was purpose-built for long-range point tracking. Key advantages over RAFT:
- Explicit occlusion modeling (unlike RAFT)
- Trained on diverse motion patterns (unlike TAPNet's conservative prior)
- Expected +10-20% visible tracks for M5t2

Not implemented yet. Parallel option to current RAFT pipeline.

---

## 4. Why RAFT+Mask Works: The Key Insight

**Distribution shift prevention**:
```
WRONG: Mask input RGB before RAFT → modifies pixel distribution → RAFT flow degraded
RIGHT: Feed original RGB to RAFT → use mask as POST-HOC validator
```

When mask is used as **input** (zero-out background pixels), RAFT's convolutional features see an unnatural distribution (hard black boundary where mouse fur meets black background). This causes flow artifacts near the mask boundary.

When mask is used as **validation only** (after RAFT produces flow), the original image statistics are preserved. Only the final occlusion label is affected, not the flow computation.

---

## 5. 3D Scene Flow Visualization (Post-Training)

After MonoFusion training converges, SE(3) motion bases encode the full 3D scene flow.
Extract via `model.compute_poses_fg(t)` — no MLP forward pass needed.

**Implementation**: `mouse_m5t2/scripts/viz_scene_flow.py` (matplotlib, no Open3D/GUI)

### Method A: Magnitude Heatmap (Easiest)
```python
# Extract per-Gaussian means at two frames (from checkpoint)
means_f0  = extract_means(model, frame=0)   # [G, 3]
means_f30 = extract_means(model, frame=30)  # [G, 3]

delta_mu = means_f30 - means_f0              # [G, 3] scene flow
flow_mag = np.linalg.norm(delta_mu, axis=1)  # [G] magnitude

# Scatter plot colored by magnitude (matplotlib, no Open3D needed)
fig = plt.figure(figsize=(12, 5))
for col_idx, (elev, azim, title) in enumerate([
        (30, 45, "Isometric"), (90, 0, "Top"), (0, 0, "Front")]):
    ax = fig.add_subplot(1, 3, col_idx+1, projection='3d')
    sc = ax.scatter(means_f0[:, 0], means_f0[:, 1], means_f0[:, 2],
                    c=flow_mag, cmap='viridis', s=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
plt.colorbar(sc, label='|φ| (scene flow magnitude)')
```

**Expected pattern**: Paws > body > head (in terms of motion magnitude)

### Method B: 3D Quiver Plot (Vector Field)
```python
# Plot flow arrows as line segments (matplotlib 3D)
# Subsample to top-K by magnitude for clarity
ax = fig.add_subplot(projection='3d')
top_k = np.argsort(flow_mag)[-200:]
ax.quiver(means_f0[top_k, 0], means_f0[top_k, 1], means_f0[top_k, 2],
          delta_mu[top_k, 0], delta_mu[top_k, 1], delta_mu[top_k, 2],
          length=0.05, normalize=True, color='red', alpha=0.6)
```

### Method C: Trajectory Trails
```python
# For each Gaussian, plot 3D path: μ_i(0), μ_i(1), ..., μ_i(59)
# Reveals kinematic chains (limbs, spine, tail)
# Color = time (blue=early, red=late)
for i in top_k:
    ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2],
            c=cm.plasma(np.linspace(0, 1, T)), linewidth=0.5)
```

### Method D: 2D Projection Verification
```python
# Project 3D scene flow → 2D and compare with RAFT flow
# Discrepancy = failure mode (depth ambiguity, multi-view inconsistency)
# K[3,3] intrinsics, Z = depth at point
uv_flow = (K[:2, :2] @ delta_mu[:, :2].T / means_f0[:, 2]).T  # (G, 2)
raft_uv  = load_raft_flow(cam_dir, frame_t)  # (N, 2) for seeded points
error = np.linalg.norm(uv_flow - raft_uv, axis=1)  # per-point 2D error
```

---

## 6. Training Monitoring Metrics (Critical)

Three quantitative metrics to detect silent training failures early:

### 6.1 Multi-View Reprojection Error
```python
# For each visible track point: triangulate 3D from all cameras, reproject to each camera
# Flag if reprojection error > 2px or growing over training iterations
reprojection_error = ||project(triangulate(tracks_all_cams)) - track_xy||_2
```
**What it catches**: Multi-view inconsistency → geometric tearing

### 6.2 Trajectory Smoothness (Jitter Metric)
```python
# Second-order derivative of per-Gaussian trajectories
jitter = ||mu(t+1) - 2*mu(t) + mu(t-1)||_2  # acceleration magnitude
```
**What it catches**: Noisy track supervision → jittery 4D reconstruction

### 6.3 Visibility Flip Rate
```python
# Count occlusion state changes per tracked point over 60 frames
flip_rate = count(occ_state[t] != occ_state[t-1]) / T
```
**What it catches**: Unstable SAM2 mask → oscillating track loss signal

---

## 7. Loss Function Reference

```python
loss = w_track    * L_track      # 2D reprojection of tracks (ESSENTIAL)
     + w_feat     * L_feat       # DINOv2 feature similarity
     + w_depth    * L_depth      # MoGe depth regularization (currently 0)
     + w_smooth   * L_smooth     # Temporal smoothness (recommended addition)
     + w_epipolar * L_epipolar   # Multi-view consistency (recommended addition)
```

| Term | Default | Status | Notes |
|------|---------|--------|-------|
| `w_track` | 2.0 | Essential | Non-negotiable |
| `w_feat` | 1.5 | Active | Can increase to 2.5-3.0 if tracks are sparse |
| `w_depth` | 0.0 | Disabled | MoGe scale not verified |
| `w_smooth` | 0.0 | **Recommended** | Add: `||D(μ, t+1) - D(μ, t)||²` |
| `w_epipolar` | 0.0 | **Recommended** | Add: cross-camera reprojection consistency |

---

↑ MOC: `docs/README.md` | ↔ Related: `monofusion_architecture.md`, `core_architecture.md`, `training_guide.md`

*Scene Flow & Tracking Theory | MonoFusion M5t2 PoC | 2026-03-29*
