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

## Experiment Progression (V5e→V5h)

### V5f — Densification + w_feat (Completed)
- w_feat 0.5→0.75, densify 40→60%, grad 0.0002→0.00015, max_gaussians 50k→100k
- **Result**: Full PSNR 9.35, FG PSNR 10.28 — no improvement over V5e
- **Finding**: Gaussian count 2.6× but PSNR flat. Loss ≠ PSNR (Key Insight #2).

### V5g — w_mask Fix + Opacity Reset OFF (Completed)
- w_mask 7.0→1.0, opacity reset OFF, w_feat→1.5, 750ep
- **Result**: FG PSNR 15.04 (ep13, best!) → 10.36 (ep480, degraded)
- **Finding**: w_mask reduction effective, but opacity reset OFF caused Gaussian bloat (16k→105k)
- **Critical Discovery**: Full-image PSNR ceiling ~9.2dB due to bg_color mismatch. FG-only PSNR is the correct metric.

### V5h — Controlled num_bg Comparison (Running)
- V5h-1: num_bg=0 (FG-only), w_mask=7.0, opacity reset ON, w_feat=1.5, 300ep
- V5h-2: num_bg=32 (FG+BG), w_mask=7.0, opacity reset ON, w_feat=1.5, 300ep
- **Hypothesis**: Isolate BG effect. Winner gets 750ep extension.

### Future Directions
- V6: CoTracker/BootsTAPIR (track quality upgrade)
- V7: DINOv2 ViT-L, DUSt3R cameras, SAM2 masks (full paper alignment)

## Git Commits
- `ac4b6bb`: V5d config + PCA pipeline + framework fixes (28 files)
- `0e58111`: Opacity reset crash fix (3 files)
- `5192d2c`: V5e results doc + fix render script weights_only

---
*MonoFusion M5t2 PoC | Updated 2026-04-01*
