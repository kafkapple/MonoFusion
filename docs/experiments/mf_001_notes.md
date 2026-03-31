# Experiment: mf_001_baseline
Date: 2026-03-27
Status: **SUPERSEDED** → see mf_001_raft below
Reason: Preprocessing substitutions invalidated (RAFT fallback tracks, relative depth, etc.)

---

# Experiment: mf_001_raft — Baseline Training (RAFT+mask tracks)
Date: 2026-03-29
Status: **PENDING** (launch after pre-training safeguards)

## Config (RAFT tracks)
- Dataset: M5t2 PoC (60 frames × 4 cameras, 512×512)
- FG Gaussians: 5,000
- BG Gaussians: 10,000
- Motion bases: 10
- GPU: A6000 (CUDA_VISIBLE_DEVICES=4)
- Env: monofusion_pytorch (cu128, Blackwell-compatible)
- Track dir: `tapir/` → symlink → `tapir_raft/` (RAFT multi-query, 5Q × 512pts)
- Tracks: RAFT+mask, multi-query [F0, F15, F30, F45, F59], 512 pts/camera/query

## Pre-Training Safeguards (from MoA deliberation 2026-03-29)
- [ ] Visibility-weighted loss: `w_track * sigmoid(-2 * occ_logit) * L2`
- [ ] Mask boundary 2px erosion (exclude uncertain edge pixels)
- [ ] Deformation MLP bias init = -0.01 (near-zero initial deformation)
- [ ] Per-camera weights: cam00=1.0, cam01=0.6, cam02=0.0, cam03=0.5

## Expected Track Coverage (RAFT multi-query, F30)
| Camera | F30 visible | Weight |
|--------|------------|--------|
| cam00 | 45% (230/512) | 1.0 |
| cam01 | 18% (92/512) | 0.6 |
| cam02 | 0% (mouse exited FOV) | 0.0 |
| cam03 | 15% (77/512) | 0.5 |

## Monitoring Metrics
| Step | L_track | Reprojection error | Jitter | Flip rate | Notes |
|------|---------|--------------------|--------|-----------|-------|
| (pending) | | | | | |

## Results
*(fill after training)*
| Metric | Train | Novel | Notes |
|--------|-------|-------|-------|
| PSNR_FG | — | — | |
| SSIM_FG | — | — | |
| LPIPS_FG | — | — | |
| Mask IoU | — | — | |
| Training time | — | — | |

## Post-Training Analysis TODOs
- [ ] 3D scene flow visualization (quiver + trails + 2D projection verify)
- [ ] Novel view synthesis video
- [ ] Tier 1 decision: proceed with CoTracker or improve safeguards?

---

# Experiment: mf_001_abandoned (original)

## Config
- Dataset: M5t2 PoC (60 frames × 4 cameras, 512×512)
- FG Gaussians: 5,000
- BG Gaussians: 10,000
- Motion bases: 10
- GPU: A6000 (48GB)
- Env: monofusion (cu118 + gsplat 1.5.3)

## Config Changes from Default MonoFusion
- Views: 4 (M5t2 cams 0,1,2,5) instead of Panoptic [3,21,23,25]
- Resolution: 512×512 (no crop, M5t2 native)
- Frame stride: 1 (pre-sampled from 3600 at stride 10)
- Depth: Depth-Anything-V2 (relative, not MoGe)
- Tracks: RAFT optical flow (not TAPNet) — visibility 2-3%
- Features: DINOv2 vits14, upsampled 37×37→512×512
- Masks: RGBA alpha channel (not SAM2)

## Known Issues Before Training
1. Track quality low (RAFT drift, 2-3% visibility) — may cause poor FG init
2. Depth is relative (not metric) — BG point cloud scale may be off
3. Feature upsampling introduces interpolation artifacts
4. FG init: 7357 valid tracks from 4 cameras (decent)

## Preprocessing Verification (2026-03-27)
- Reprojection check: 4/4 cameras OK
- Mask FG coverage: 2.0-2.8%
- DINOv2 PCA: semantically consistent across cameras
- Depth: structure visible, mouse region distinct
- Tracks: query points on mouse, drift in later frames

## Key Results
| Metric | Train | Novel | Notes |
|--------|-------|-------|-------|
| PSNR_FG | — | — | pending |
| SSIM_FG | — | — | pending |
| Mask IoU | — | — | pending |
| FG Gaussian count | 7357 init | — | post-densification |

## Observations
- (pending training completion)

## Decision
- (pending)

---

*mf_001 | MonoFusion M5t2 PoC | 2026-03-27*
