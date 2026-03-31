# MonoFusion M5t2 — V5e Results & Next Steps

## V5e Config (First Successful Convergent Run)
- **Features**: PCA 32d + L2 norm (from ViT-S 384d, 87.5% variance)
- **Loss**: w_feat=0.5 (ramp ep5-50), w_depth_reg=0.0, w_rgb=5.0
- **Model**: num_fg=5000→23,845 (densified), num_bg=32, max_gaussians=50k
- **Optimizer**: stop_densify=40%, stop_control=80%, reset_opacity_mult=1.5
- **Data**: 80 frames, 4 cameras, 512×512, OMP=8 per-process

## V5e Metrics
| Camera | F0 PSNR | F15 | F30 | F45 | Avg SSIM |
|--------|---------|-----|-----|-----|----------|
| cam00 | 9.78 | 10.80 | 10.55 | 10.19 | 0.405 |
| cam01 | 9.45 | 10.27 | 9.45 | 9.59 | 0.373 |
| cam02 | 7.92 | 8.44 | 8.37 | 9.07 | 0.329 |
| cam03 | 9.82 | 9.42 | 9.19 | 9.15 | 0.401 |
| **Avg** | **9.24** | **9.73** | **9.39** | **9.50** | **0.377** |

## Training History (V5b→V5e)
| Version | Final Loss | PSNR | Status | Root Cause of Failure |
|---------|-----------|------|--------|----------------------|
| V5b | 42,391 | — | DIVERGED | 384d MSE 12× scale |
| V5c | 36,597 | — | DIVERGED | Hard warmup gradient shock |
| V5d | NaN@step2380 | — | CRASHED | Opacity reset < cull threshold |
| **V5e** | **7.24** | **9.47** | **CONVERGED** | — |

## Key Findings This Session
1. **384d raw DINOv2 → PCA 32d + L2 norm is mandatory** (L2 norm ~38 pre-norm)
2. **Gradual feat ramp** (not hard warmup) prevents gradient shock on RGB-optimized structure
3. **w_depth_reg=0.0** (paper default) — nonzero was confirmed destabilizer
4. **num_bg>0** needed to absorb mask imperfection gradients
5. **reset_opacity_multiplier > 1.0** (1.5) to survive cull threshold
6. **OMP per-process** (not global) — FaceLift num_workers=8 × OMP=8 = 72 cores
7. **stop_control_steps must scale** with total_steps (80%, not hardcoded 4000)

## Remaining Paper Deviations (7/14 unfixed)
| # | Deviation | Impact | Fix Effort |
|---|-----------|--------|------------|
| 2 | w_feat 0.5 (paper 1.5) | Medium | Config change |
| 5 | RAFT-small (paper BootsTAPIR) | High | Install + reprocess |
| 6 | Track format (modified logits) | Low | Comes with #5 |
| 7 | Camera opencv frozen (paper DUSt3R) | High | Learnable focal (~10L) or DUSt3R |
| 8 | Mask RGBA (paper SAM2) | Medium | Alpha binarize (1L) or SAM2 |
| 13 | 80fr/4cam (paper 200+/5-9) | Medium | Data limitation |

## Next Experiment Plan

### V5f — Diagnostic + Quick Wins (Next Session Priority)
**Goal**: Improve rendering quality without infrastructure changes
- Per-component loss logging (feat vs photo ratio)
- Binarize RGBA alpha masks (1 line)
- w_feat: 0.5 → 0.75 (incremental, not 1.5 directly)
- Extend densification to 60% of training (more Gaussians)
- Gaussian scale/opacity diagnostics
- **Gate**: If photo/feat ratio > 3.0, reduce w_feat before continuing

### V5g — Capacity Increase (If V5f renders sharper)
- 450 epochs, max_gaussians=100k
- Only proceed if V5f confirms photometric convergence

### V6 — Track Quality (If V5f still blurry)
- CoTracker or BootsTAPIR (long-range tracks)
- Prioritize over V5g if rendering quality plateaus
- Requires: installation, track regeneration for 80 frames × 4 cameras

### V7 — Full Paper Alignment (Long-term)
- DINOv2 ViT-L (1024d→PCA 32d)
- DUSt3R camera optimization or learnable focal length
- SAM2 masks
- More cameras/frames if available

## Git Commits This Session
- `ac4b6bb`: V5d config + PCA pipeline + framework fixes (28 files)
- `0e58111`: Opacity reset crash fix (3 files)

---
*MonoFusion M5t2 PoC | 2026-03-31*
