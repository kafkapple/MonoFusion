# Depth Alignment Plan — Two Approaches

> Problem: MoGe relative depth has independent scale per camera → multi-view inconsistency → flat Gaussians
> Goal: Consistent metric depth across all cameras
> Date: 2026-04-02

---

## Approach A: Learned Per-Camera Depth Scale (Fast)

### Concept
Add 2 learnable parameters per camera (scale `s`, shift `b`) to align MoGe relative depth to predicted depth during training:
```
aligned_depth = s_cam * moge_depth + b_cam
```
The optimizer jointly learns scene reconstruction AND depth alignment.

### Implementation

**1. Add camera_id to batch** (`casual_dataset.py`):
```python
# In __getitem__():
data["camera_id"] = self.camera_id  # set during dataset init
```

**2. Add depth_scale parameters** (`trainer.py` or `scene_model.py`):
```python
# Per-camera scale and shift (learnable)
self.depth_scales = nn.Parameter(torch.ones(n_cameras))
self.depth_shifts = nn.Parameter(torch.zeros(n_cameras))
```

**3. Apply in depth loss** (`trainer.py`):
```python
# Before computing depth loss:
cam_ids = torch.cat([b["camera_id"] for b in batch])
scales = self.depth_scales[cam_ids].view(-1, 1, 1, 1)
shifts = self.depth_shifts[cam_ids].view(-1, 1, 1, 1)
aligned_tgt_disp = 1.0 / (scales * depths[..., None] + shifts + 1e-5)
```

### Pros/Cons
- **Pro**: No additional preprocessing, 2 params/camera, instant implementation
- **Pro**: Optimizer finds best alignment automatically
- **Con**: Scale may not converge if initial depth is very wrong
- **Con**: Requires w_depth > 0 to be effective (currently 0)

### Config Changes
```yaml
w_depth_reg: 0.1    # was 0.0
w_depth_grad: 0.05  # was 0.0
# w_depth_const: keep 0 initially
```

### Estimated Time: 2-3 hours implementation + training

---

## Approach B: COLMAP MVS Consistent Depth (Robust)

### Concept
Use ground-truth camera poses with COLMAP to generate geometrically consistent dense depth maps across all cameras. Replace MoGe depth with MVS depth.

### Pipeline
```
GT cameras (opencv_cameras.json)
    → Convert to COLMAP format (cameras.txt, images.txt, points3D.txt)
    → COLMAP feature_extractor + exhaustive_matcher
    → COLMAP triangulator (using GT poses, not SfM)
    → COLMAP patch_match_stereo (dense MVS)
    → COLMAP stereo_fusion → fused point cloud
    → Per-view depth map extraction
```

### Implementation

**1. Install COLMAP** (gpu03):
```bash
conda install -c conda-forge colmap  # or build from source
# Alternative: pip install pycolmap
```

**2. GT pose → COLMAP format converter**:
```python
# cameras.txt: PINHOLE model, fx fy cx cy
# images.txt: quaternion + translation per image
# points3D.txt: empty (let COLMAP triangulate)
```

**3. Run MVS**:
```bash
colmap feature_extractor --image_path images/ --database_path colmap.db
colmap exhaustive_matcher --database_path colmap.db
colmap point_triangulator --database_path colmap.db \
    --image_path images/ --input_path sparse/0/ --output_path sparse/0/
colmap patch_match_stereo --workspace_path dense/ \
    --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path dense/ --output_path fused.ply
```

**4. Extract per-view depth maps**:
```python
# From fused point cloud + GT cameras → per-view depth
# Or directly from patch_match_stereo output (depth_maps/)
```

**5. Replace MoGe depth in dataset**:
```python
# depth_type = "colmap_mvs"
# Load from colmap_depth/ instead of aligned_moge_depth/
```

### Pros/Cons
- **Pro**: Geometrically consistent across ALL cameras by construction
- **Pro**: Metric scale (not relative) → w_depth can be enabled confidently
- **Pro**: Also generates dense point cloud for BG Gaussian init (solves 32 BG problem!)
- **Con**: Requires COLMAP installation
- **Con**: MVS can fail on textureless/reflective regions
- **Con**: Additional preprocessing time (1-4 hours)

### Estimated Time: 4-8 hours (install + run + integrate)

---

## Execution Plan (Parallel)

```
Timeline:
  Hour 0-2: [A] Implement learned depth scale (code changes)
            [B] Install COLMAP, write GT→COLMAP converter
  Hour 2-4: [A] V5k training with depth scale (50ep quick test)
            [B] Run COLMAP MVS pipeline
  Hour 4-6: [A] Evaluate V5k novel views
            [B] Extract depth maps, replace MoGe
  Hour 6-8: Compare A vs B results → pick winner → full training
```

### Success Criteria
- Cross-camera projection error: < 5px (currently 43-129px)
- Novel view PSNR within 3dB of training view
- No "flat photo stacking" in interpolated views

---

*MonoFusion M5t2 PoC | Depth Alignment Plan | 2026-04-02*
