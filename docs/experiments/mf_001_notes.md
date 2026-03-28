# Experiment: mf_001_baseline
Date: 2026-03-27
Status: **ABANDONED** (audit: preprocessing substitutions invalidated results)

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
