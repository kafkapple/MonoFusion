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

## Experiment Design (Added 2026-04-05)

### 9. Never Change Multiple Variables Simultaneously

**Rule**: Each experiment should change exactly ONE variable from a controlled baseline.

**Evidence**: V7c changed 4 variables at once (BG LR, FG count, BG count, motion bases). Result: loss ~48 at ep127 (worse than V7b's 7.71), but impossible to attribute the regression to any specific variable. 5-iteration 3-model deliberation confirmed: "scientifically worthless if both succeed and fail."

**Apply**: Use V8-style isolation (V8a baseline → E1/E2/E3 single-variable) with pre-defined success criteria (>1dB PSNR improvement = significant).

### 10. Bare `except:` Is a Silent Killer in ML Code

**Rule**: Never use bare `except:` in training code. Always catch specific exceptions.

**Evidence**: `trainer.py` had `try: stats["feat_loss"] = feat_loss.item() / except: stats = {...without feat_loss}`. This silently swallowed `NameError` when `feat_loss` was undefined (has_bg=False path), making it appear that feat_loss was always computed when it wasn't. This pattern was present in 2 locations in compute_losses and compute_stat_losses.

**Pattern**: For conditional variables, use explicit check:
```python
stats = {guaranteed_keys}
try:
    stats["optional_key"] = optional_var.item()
except NameError:
    pass
```

### 11. PSNR Mask Must Match Evaluation Region

**Rule**: When computing FG/BG PSNR, verify the mask variable hasn't been reassigned.

**Evidence**: `compute_stat_losses` had `masks = valid_masks` (line 614) that overwrote the FG mask defined earlier. This made all PSNR computations use the full-frame valid mask instead of the FG-only mask, inflating PSNR by including easy-to-predict static background pixels.

**Check**: `masks` used in PSNR should be `b["masks"] * valid_masks` (FG segmentation × frame validity), NOT `valid_masks` alone.

### 12. Define Success Criteria Before Running Experiments

**Rule**: Write down quantitative success/failure thresholds before executing.

**Evidence**: V5b-V7c: 11+ experiments with no pre-defined "this PSNR means success." Post-hoc interpretation led to ambiguous conclusions ("loss decreased, but is 7.71 good or bad?").

**Template**: `>1.0 dB improvement = significant, 0.3-1.0 = marginal, <0.3 = no effect`.

### 13. BG Learning Is Required for FG Quality (V8 Breakthrough)

**Rule**: When mask compositing renders both FG and BG, you MUST allow BG Gaussians to learn. Frozen BG poisons FG gradients.

**Evidence**: V8 single-variable isolation (2026-04-07):
- V8a (BG frozen): full PSNR 9.07 dB
- E1 (BG unfrozen, only change): full PSNR **25.80 dB** = **+16.73 dB**
- E2 (FG 18K, BG frozen): 6.44 dB — actually WORSE
- E3 (bases 28, BG frozen): 7.07 dB — actually WORSE

V5–V7 series spent 11+ experiments tuning FG side (count, motion bases, features) while BG was frozen at LR=1e-9. The "FG ghost" symptom was a downstream effect of broken BG, not insufficient FG capacity.

**Mechanism**: rendered = FG·mask + BG·(1−mask). If BG never updates from initialization noise:
- BG term is persistent error → loss can never reach 0
- FG over-corrects to compensate → ghosting artifacts
- Adding more FG capacity (E2) wastes parameters without addressing root cause

**Apply**: For any alpha-composited rendering pipeline, treat BG learning rate as a critical hyperparameter, not a "set-and-forget" detail. Verify BG actually updates by checking loss on BG-only regions before assuming the FG side is the bottleneck.

### 14. Capacity ≠ Quality — Add Capacity Only After Fixing Bottlenecks

**Rule**: Don't scale up model capacity until you've identified and fixed the actual bottleneck.

**Evidence**: V8 E2 added 13K Gaussians (5K → 18K) and got WORSE PSNR (9.07 → 6.44). E3 added 18 motion bases (10 → 28) and also got WORSE (9.07 → 7.07). Both added compute and memory without touching the BG bottleneck, and the extra parameters likely interfered with the broken pipeline.

**Apply**: When tempted to "throw more capacity at it" — first verify your isolation experiment shows that capacity is the bound. If a single hyperparameter (E1 = 1 line change) gives 16 dB, capacity scaling is meaningless until that fix is applied.

### 15. Hidden Parameter Interactions Can Confound Single-Variable Isolation

**Rule**: Before designing an experiment that changes initialization counts (FG/BG), trace the interaction with downstream caps (`max_gaussians`, `stop_densify_steps`, `opacity_reset`). A single overlooked cap can convert your single-variable isolation into a confounded multi-variable test.

**Evidence**: V9a (2026-04-07) intended to test "BG capacity 5×" by setting `num_bg=50000` (was 10000), with all other params = E1. The actual behavior:
- Init: 5K + 50K = 55K Gaussians
- First densification: jumped to **940,831** Gaussians (V7c-instability regime)
- `max_gaussians=100K` cap kicked in immediately, freezing the count
- Loss spiked to 69 at ep1 (V7c signature)

The intended single-variable change ("BG seed count") was not the actual variable. The actual variable was "first-densification-overshoot magnitude, gated by max_gaussians cap behavior."

**Apply**: Before any experiment that changes a count parameter (FG/BG/bases), ask:
1. What does the densifier do with this initial count after step 1?
2. Does the result of step 1 cross any cap?
3. Will the cap then change the optimization trajectory permanently?

If yes to any → the experiment is confounded. Either fix the cap interaction or design a different test.

### 16. Parameter Names Lie — Trace Actual Code Behavior

**Rule**: Never assume a parameter does what its name suggests. Always read the code that uses the parameter before designing an experiment around it.

**Evidence**: `max_gaussians` (2026-04-07) sounds like a hard ceiling. Actual behavior in `_densify_control_step`:
```python
if cfg.max_gaussians > 0 and self.model.num_gaussians >= cfg.max_gaussians:
    return  # skip densification
```

This is a **"skip if already over"** trigger, not a hard cap. Once `num_gaussians` exceeds `max_gaussians`, densification stops forever. But a single densification step can produce arbitrary growth (E1 went 15K → 297K in epoch 0).

Result: setting `max_gaussians=100K` and `max_gaussians=200K` produces the **same final count** (~297K), because the first densify step overshoots whichever value is set. V9b discovered this by aborting at ep5 when no "Skipping densification" log appeared until well after the cap was exceeded.

The parameter name suggests "ceiling at N gaussians" but the behavior is "trigger to disable densification once N is exceeded once." These are completely different semantics.

**Apply**:
- For `max_gaussians`: it does NOT cap at the value; it triggers a permanent skip after first overshoot. To actually constrain capacity, modify the densifier itself (not this parameter).
- For any newly encountered parameter: grep the codebase for its uses, read the surrounding 20 lines of code, and verify the actual semantics before designing a test.

This is a generalization of the "variable names lie" lesson from the camera convention bug (LESSONS §1-§3): **all parameter names are advisory, only the code is authoritative.**

### 17. Full-Image PSNR Lies for Small-Object Reconstruction (Goodhart's Law)

**Rule**: When the object of interest occupies < 10% of pixels, full-image PSNR is dominated by background quality and **cannot** distinguish "good FG + good BG" from "good BG + zero FG". Always report FG-only and BG-only PSNR separately.

**Evidence**: V8/V9 killer test (2026-04-07):

| Exp | Full PSNR | FG PSNR | BG PSNR | Gap |
|-----|----------:|--------:|--------:|----:|
| V8a | 7.07 | 7.07 | 7.09 | 0.02 (uniformly broken) |
| **E1** | **22.06** | **11.01** | **24.94** | **13.92** |
| **V9c** | **22.00** | **10.84** | **24.97** | **14.13** |

E1's celebrated "+16.73 dB breakthrough" decomposed as **+17.85 dB BG, +3.94 dB FG**. The mouse (the entire purpose of the project) stayed at PSNR ~11 (barely visible), while the static scene reached PSNR ~25. Since BG occupies ~95% of pixels and FG only ~3-5%, the full-image PSNR was dominated by BG.

The visual symptom: BG looks like a "blurry temporal average of the scene including faint mouse trails", FG looks like a "translucent ghost". The metric could not see this failure because it averaged over the whole image.

**Math**: For 95% BG at PSNR 25 and 5% FG at PSNR 11:
- BG MSE = 10^(-25/10) = 0.00316
- FG MSE = 10^(-11/10) = 0.0794
- Weighted MSE = 0.95 × 0.00316 + 0.05 × 0.0794 = 0.00697
- Full PSNR = -10 log₁₀(0.00697) = **21.6 dB**

This means even **PSNR 0 on the entire FG region** would only reduce full PSNR by 1-2 dB. The metric is structurally incapable of penalizing FG failures when FG is small.

**Apply**:
- For any task where the object of interest is < 10% of pixels:
  1. Report FG-only PSNR, BG-only PSNR, and full PSNR separately
  2. Define success criteria on FG PSNR primarily, not full PSNR
  3. Visualize a few frames manually before trusting metric improvements
  4. Run a "swap test": render frame t, evaluate against frame t+10. Real reconstruction should give much higher loss when frames are mismatched. If loss is flat, the model is producing a temporal average.
- For MonoFusion specifically: full-image PSNR is the **wrong** primary metric. Use FG-only PSNR.

**Goodhart's Law generalization**: "When a measure becomes a target, it ceases to be a good measure." The pre-defined V8 success criterion (>1 dB full PSNR = significant) was rigorous methodology in form but wrong in substance — it optimized for a measure that didn't capture what we cared about. Pre-defining criteria is necessary but not sufficient; the criteria themselves must be the right ones.

---

*MonoFusion M5t2 PoC | Lessons Learned | 2026-04-07*
*Validated by 3-model MoA deliberation + V8/V9 isolation + killer test post-mortem*
