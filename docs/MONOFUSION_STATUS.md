# MonoFusion M5t2 PoC — Project Status

> Last: 2026-04-07 PM | Phase: 🚨 **V8/V9 audit revealed PSNR artifact** → V10 architectural fix in progress

## ⚠️ CRITICAL UPDATE (2026-04-07 PM)

The V8 E1 result reported here as a "+16.73 dB breakthrough" was **invalidated by killer test**:

| Exp | Full PSNR | **FG PSNR (mouse)** | BG PSNR | Gap |
|-----|----------:|--------------------:|--------:|----:|
| V8a (BG frozen) | 7.07 | 7.07 | 7.09 | uniform broken |
| **E1 "breakthrough"** | **22.06** | **11.01** | **24.94** | **13.92 dB** |
| V9c (cap=1M) | 22.00 | 10.84 | 24.97 | 14.13 dB |

**The full PSNR jump was BG learning the static scene, not mouse reconstruction.** FG (the mouse — the entire purpose) stayed at PSNR ~11 (barely above noise) in all experiments. Full forensic analysis: [`~/results/MonoFusion/AUDIT_REPORT_V8V9_ARTIFACT.md`](../../results/MonoFusion/AUDIT_REPORT_V8V9_ARTIFACT.md). Root cause: standard L1 loss is BG-dominated when FG occupies <10% of pixels (LESSONS §17).

## 1. Current State

| Item | Status | Detail |
|------|--------|--------|
| **Dataset** | **markerless_v7** | Raw data, per-camera intrinsic, 4cam×80f, 512×512 |
| **Best model (FG metric)** | E1 / V9c (tied) | **FG PSNR ~11 dB** — mouse barely reconstructed in any experiment |
| **Camera convention** | FIXED | metadata flag `camera_convention: w2c` |
| **V8 verdict** | 🚨 Metric artifact | Full PSNR was BG-dominated; FG never properly learned |
| **Next** | V10a (rgb_loss_mode=balanced) | Architectural fix: balance L1 between FG/BG regions |
| **Git** | origin=kafkapple, upstream=Z1hanW | Fork safety established |

### ⚠️ CRITICAL: Dataset Switch (2026-04-02)

**m5t2_v5 (FaceLift전처리) → markerless_v7 (원본 raw) 전환.**

m5t2_v5의 모든 실험(V5b~V5l)은 FaceLift가 per-camera intrinsic을 동일 값(fx=549, cx=cy=256)으로 정규화한 데이터를 사용. 원본 카메라는 fx=1557~1637, cx=583~642, cy=417~552로 모두 다름. 이 정규화가 multi-view geometry를 파괴하여 novel view "flat photo stacking" 발생.

markerless_v7은 원본 raw에서 직접 변환, per-camera intrinsic 보존.

### V5j Metrics (render_and_evaluate.py)

| Camera | F0 | F15 | F30 | F45 | Avg PSNR |
|--------|-----|------|------|------|----------|
| cam00 | 17.3 | 18.0 | 18.7 | 18.9 | **18.2** |
| cam01 | 11.5 | 12.1 | 15.3 | 15.4 | 13.6 |
| cam02 | 10.8 | 12.9 | 12.2 | 14.6 | 12.6 |
| cam03 | 9.4 | 9.7 | 8.4 | 8.6 | 9.0 |

## 2. Breakthrough: Camera Convention Fix

### Problem
V5b~V5h (8 experiments) — FG Gaussian이 GT와 18,737px 오차. Loss 7+ plateau.

### Root Cause
**Data format mismatch** (원본 코드 무결):
- MonoFusion `load_known_cameras`: DUSt3R convention — `md['w2c']`에 c2w 저장 → `inv()` 적용하여 real w2c 획득
- `convert_m5t2.py`: OpenCV convention — `md['w2c']`에 real w2c 저장
- 결과: `inv(real_w2c) = c2w` → `self.w2cs = c2w` → 모든 projection 역전

### Fix
`camera_convention` metadata flag:
```python
# convert_m5t2.py: "camera_convention": "w2c"
# casual_dataset.py: flag에 따라 inv() skip
cam_convention = md.get('camera_convention', 'c2w')
if cam_convention == 'w2c':
    k, w2c = md['k'][t][c], np.array(md['w2c'][t][c])
else:
    k, w2c = md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
```

### Evidence

| Test | Fixed (real w2c) | Bug (inv → c2w) |
|------|-----------------|-----------------|
| Camera spread | 6.97m ring | 0.39m clustered |
| cam0 back-projection | **0.4px** | 18,737px |
| Best loss (300ep) | **2.06** | 7.24 |

### Lesson
> **변수명 `w2c`/`c2w`를 신뢰하지 말 것.**
> 실제 연산(inv, einsum)을 추적하고, dual hypothesis test로 독립 검증.
> Hyperparameter 튜닝 전에 geometry 검증 선행 필수.

## 3. Experiment History

| Exp | Epochs | Loss | Key Change | Result |
|-----|--------|------|------------|--------|
| V1-V3 | 50-600 | 6.9-7.0 | Initial pipeline | BUG-002 (cam params overwrite) |
| V5b | 300 | ~9 | PCA 384d | MSE 12× divergence |
| V5c | 300 | ~8 | w_feat=0.01 | Still divergent |
| V5d | 300 | 8.5→NaN | PCA32d+L2norm+bg32 | NaN at step 2380 (opacity reset < cull) |
| V5e | 300 | **7.24** | reset_mult=1.5+stop_control | First convergent (PSNR 9.47) |
| V5f-h | 50-300 | 7+ | Densify/track/hyperparams | All had camera convention bug |
| **V5i** | 50 | **2.53** | **Convention fix** | Sparse but correct position |
| **V5j** | 300 | **2.06** | Fix + bg32 + opacity reset | **First successful reconstruction** |
| V7a | 100 | 8.91 | markerless_v7 dataset | New data, BG frozen |
| V7b | 300 | 7.71 | 300ep on markerless_v7 | BG frozen (LR~1e-9) |
| V7c | 300 | ~48@ep127 | All params → paper spec | Unstable, confounded |

## 4. Critical Bugs Resolved

| Bug | Impact | Root Cause | Fix |
|-----|--------|-----------|-----|
| Camera convention | V5b-V5h 전실험 실패 | DUSt3R vs OpenCV w2c mismatch | metadata flag |
| Cam params overwrite | V1-V3 렌더링 단일시점 | loop variable overwrite in dance_glb.py | Ks_fuse, w2cs_fuse 누적 |
| Opacity reset < cull | V5d NaN crash | reset 0.08 < cull threshold 0.1 | reset_mult=1.5 |
| RAFT track discard | 0% visible tracks | expected_dist=0.0 → sigmoid=0.5 | CONFIDENT_DIST=-2.0 |
| DINOv2 resolution | IndexError crash | ViT-S/14 → 37×37 vs 512×512 | F.interpolate upsample |
| dyn_mask bool cast | Wrong FG/BG | float32 -1.0 → True | `raw > 0` |

## 5. V7 Series & Phase 0 (2026-04-05)

### V7 Experiments (markerless_v7 dataset)

| Exp | FG | BG | Bases | BG LR | Epochs | Loss | Notes |
|-----|-----|------|-------|-------|--------|------|-------|
| V7a | 5K | 10K | 10 | frozen | 100 | 8.91 | Quick test |
| V7b | 5K | 10K | 10 | frozen | 300 | 7.71 | Full run, no PSNR |
| V7c | 18K | 100K→985K | 28 | **paper spec** | 300 (running) | ~48@ep127 | Unstable, all params changed |

### Phase 0 Code Fixes (MoA+Audit identified, 6 fixes applied)

| Fix | File | Issue |
|-----|------|-------|
| BG LR selectable (--bg_lr_config) | train_m5t2.py | BGLRConfig(~1e-9) was default, paper spec is BGLRGTConfig |
| bare except→except NameError | trainer.py ×2 | Silent error masking |
| wandb double-log removed | trainer.py ×2 | x-axis desync in wandb charts |
| depth_scales/shifts ckpt save+load | trainer.py + train_m5t2.py | Params lost on resume |
| PSNR mask corruption fixed | trainer.py | masks=valid_masks was overwriting FG mask |
| Seed control (--seed) | train_m5t2.py | 5-source deterministic: torch+cuda+numpy+random+cudnn |

### V8 Isolation Plan (→ [detail](experiments/mf_v8_isolation_plan.md))

V8a (baseline, seed=42, BG frozen) → E1 (BG LR only) → E2 (FG 18K) → E3 (bases 28)
- Success: PSNR >1dB improvement = variable matters
- 5-iteration deliberation (3 models): EXECUTE verdict

### Open Issues

- Depth alignment: MoGe relative depth → per-camera scale (learned params added, [plan](DEPTH_ALIGNMENT_PLAN.md))
- V7c loss instability: increasing after ep30 minimum — may indicate BG LR too aggressive at 985K scale
- H100 incompatibility: torch 2.1+cu118 lacks sm_90 → A40 only

## 6. File Locations

### Results (Local only)
```
~/results/MonoFusion/
├── v5j_viz/          # R1 GT vs Rendered, R4 4D Gaussians, R5 Scene flow
├── v5j_previews/     # Training epoch snapshots
└── CAMERA_FIX_SUMMARY.md
```

### Server (gpu03)
```
/node_data/joon/data/monofusion/markerless_v7/    ← CURRENT (raw data, correct intrinsics)
├── images/markerless_cam00~05/   # 6cam × 150frames, 1152×1024
├── masks/                        # FG masks
├── aligned_moge_depth/m5t2/      # MoGe depth per camera
├── dinov2_features/              # DINOv2 features
├── tapir/                        # RAFT tracks
└── _raw_data/.../Dy_train_meta.json  # camera_convention: w2c, per-cam K

/node_data/joon/data/monofusion/m5t2_v5/          ← DEPRECATED (FaceLift intrinsics wrong)
└── results_v5j/                  # reference only
```

### Verification Scripts
```
mouse_m5t2/scripts/
├── verify_camera_convention.py    # Dual hypothesis test
├── verify_gaussian_projection.py  # FG→GT projection error
├── render_and_evaluate.py         # R1+R4+R5 static viz + metrics
└── render_video_novel.py          # R2+R3 video rendering
```

## 7. Architecture Summary

```
RGB (4cam×80f) → [RAFT tracks + MoGe depth + DINOv2 feats + SAM2 masks]
    → casual_dataset.py (camera_convention flag)
    → 4D Gaussian Splatting: canonical 3DGS + SE(3) motion bases (K=10 or 28)
    → gsplat rasterization → L_rgb + L_track + L_mask + L_feat + L_depth
```

Key design: SE(3) basis trajectories (not MLP). K shared rigid transforms, per-Gaussian linear combination. Multi-view consistent by construction.

### Key Training Args (train_m5t2.py)

| Arg | Default | Description |
|-----|---------|-------------|
| `--seed N` | None | Reproducibility (torch+cuda+numpy+random+cudnn) |
| `--bg_lr_config gt\|frozen` | gt | BG LR: paper spec (gt) or ~1e-9 (frozen) |
| `--num_fg` | 5000 | FG Gaussian count (paper: 18K) |
| `--num_bg` | 10000 | BG Gaussian seed count (paper: 1.2M) |
| `--num_motion_bases` | 10 | Motion bases (paper: 28) |
| `--w_feat` | 0.75 | Feature loss weight (paper: 1.5 with PCA32) |
| `--max_gaussians` | 100000 | Hard cap on densification (0=uncapped) |

---

*MonoFusion M5t2 PoC | Project Status | 2026-04-05*
