# MonoFusion M5t2 PoC — Lessons Learned

> First successful 3D reconstruction: V5j (loss 2.06, PSNR 13.36, 204k FG Gaussians)
> 8 failed experiments (V5b-V5h) due to geometry bug, not hyperparameter issues
> 3-model MoA deliberation verified (2026-04-02)

---

## Core Principle

### 1. Geometry First, Always

**Rule**: Verify geometric correctness before ANY hyperparameter tuning or large-scale training.

**Evidence**: V5b-V5h (8 experiments, dozens of GPU hours) were ALL invalidated by a single camera convention bug. Loss/PSNR numbers were meaningless because the 3D coordinate system was wrong. Only after fixing geometry did loss drop from 7.24 to 2.06.

**When to apply**: Before the FIRST training run on ANY new dataset or pipeline modification. Not after 8 failed experiments.

---

## Verification Tactics

### 2. Visualize, Don't Assume

**Rule**: Implement visual debugging tools from day one. Render camera frustums, sparse point clouds, and projected points in 3D before training.

**Evidence**: The camera convention bug (w2c stored as c2w) would have been caught in minutes with a 3D camera pose visualization. Instead, it persisted across 8 experiments because only scalar metrics (loss, PSNR) were monitored.

**Checklist for new datasets**:
- [ ] Plot camera positions in 3D — do they form expected configuration?
- [ ] Project scene center into all cameras — all in-frame?
- [ ] Back-project FG from cam0 → project to cam1-N — error < 5px?

### 3. Trace Operations, Not Variable Names

**Rule**: Never trust variable names like `w2c` or `c2w`. Trace the actual math: `inv()`, `einsum()`, `@` operators.

**Evidence**: `casual_dataset.py:373` stored `inv(md['w2c'])` into `self.w2cs`. The variable name said "w2c" but the value was c2w (because DUSt3R stores c2w in the 'w2c' field). The inv() converted it to real w2c for DUSt3R data, but double-inverted our real w2c back to c2w.

**Dual hypothesis test**: When uncertain about convention, compute both interpretations and check which produces physically plausible results (camera spread, projection accuracy).

### 4. Convention Verification at Pipeline Boundaries

**Rule**: When converting data between systems (DUSt3R → OpenCV → MonoFusion → gsplat), explicitly document each coordinate convention and verify at every boundary.

**Evidence**: DUSt3R stores c2w in a field named 'w2c'. OpenCV calibration stores real w2c. This single mismatch cascaded through the entire pipeline.

**Apply to**: Any cross-library integration — ROS↔OpenCV, COLMAP↔NeRF Studio, Unity↔OpenGL, etc.

---

## Assessment Principles

### 5. "It Runs" ≠ "It Works"

**Rule**: A crash-free training run is not success. Define quantitative AND qualitative pass criteria before training.

**Evidence**: V5e ran 300 epochs without crash, achieved loss 7.24, PSNR 9.47. It was declared "first convergent run." But the rendered output was a blurry blob — the pipeline was broken, not working. V5j's PSNR 13.36 is still low by 3DGS standards (good: 25+).

**Pass criteria for 3DGS**:
- Novel view PSNR within 3dB of training view
- No "flat photo stacking" in interpolated views
- FG object shape recognizable in rendered output

### 6. Diagnose Per-Component, Not Per-Scalar

**Rule**: When quality is poor, measure per-camera, per-region (FG/BG), per-frame — not just average PSNR.

**Evidence**: Average PSNR 13.36 hides that cam00=18.2 and cam03=9.0. This 9dB gap reveals cross-camera inconsistency, not general quality. Without per-camera breakdown, the real problem (multi-view alignment) is invisible.

---

## System Architecture

### 7. Pipeline Components Are Interdependent

**Rule**: Before removing or replacing any pipeline component, document what implicit contracts it provides to downstream modules.

**Evidence**: Removing DUSt3R (because calibrated cameras were available) also removed the **metric scale anchor** that MoGe depth depends on. MoGe alone produces per-view relative depth with independent scale — fine for DUSt3R-aligned pipeline, catastrophic without it. Result: `w_depth=0` forced, depth supervision lost, Gaussians collapsed to 2D billboards.

**Dependency chain that broke**:
```
DUSt3R → metric scene scale → MoGe depth alignment → w_depth loss → 3D structure
         ↑ REMOVED                                    ↑ FORCED TO 0
```

### 8. Relative Depth Requires Scale Anchoring for Multi-View

**Rule**: Per-view relative depth (MoGe, Depth-Anything, MiDaS) cannot enforce multi-view 3D consistency alone. It needs a shared metric scale from: DUSt3R, COLMAP MVS, known camera baselines, or learned per-camera scale parameters.

**Evidence**: MoGe generates depth independently per camera. cam00 depth scale ≠ cam01 depth scale. When fused with absolute camera poses, Gaussians receive contradictory 3D instructions → mathematically optimal solution = 2D billboard per view. This is the root cause of "flat photo stacking" in novel views.

**Solutions (by effort)**:
1. Learned per-camera scale/shift (cheapest, add 2 params per camera)
2. MoGe scale alignment using triangulated sparse points
3. COLMAP MVS dense depth (most robust, 4-8 hours compute)
4. DUSt3R reintroduction (original pipeline, most reliable)

---

## Quick Reference Checklist

```
New Dataset / Pipeline Integration:
  □ Camera pose visualization (3D frustums)
  □ Cross-camera projection test (< 5px error)
  □ Convention dual-hypothesis test (w2c vs c2w)
  □ Depth scale consistency check across views
  □ Single-view overfit test (sanity, before multi-view)

Before Training:
  □ Define pass/fail criteria (not just "it runs")
  □ Per-camera, per-region diagnostic plan
  □ Component dependency map documented

After First Results:
  □ Novel view rendering (not just training view)
  □ Per-camera PSNR breakdown
  □ 3D point cloud shape inspection
```

---

*MonoFusion M5t2 PoC | Lessons Learned | 2026-04-02*
*Validated by 3-model MoA deliberation (Claude/Gemini/GPT)*
