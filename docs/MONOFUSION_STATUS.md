# MonoFusion M5t2 PoC — Project Status

> Last: 2026-04-02 | V5j = first successful reconstruction

## 1. Current State

| Item | Status | Detail |
|------|--------|--------|
| **Dataset** | **markerless_v7** | Raw data, per-camera intrinsic, 4cam, 512×512 |
| **Best model** | V5j 300ep (m5t2_v5) | loss 2.06, PSNR 13.36 — BUT used wrong dataset (FaceLift) |
| **Camera convention** | FIXED | metadata flag `camera_convention: w2c` |
| **Next** | V7a on markerless_v7 (512×512, per-cam K, 4cam, 80f) |

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

## 4. Critical Bugs Resolved

| Bug | Impact | Root Cause | Fix |
|-----|--------|-----------|-----|
| Camera convention | V5b-V5h 전실험 실패 | DUSt3R vs OpenCV w2c mismatch | metadata flag |
| Cam params overwrite | V1-V3 렌더링 단일시점 | loop variable overwrite in dance_glb.py | Ks_fuse, w2cs_fuse 누적 |
| Opacity reset < cull | V5d NaN crash | reset 0.08 < cull threshold 0.1 | reset_mult=1.5 |
| RAFT track discard | 0% visible tracks | expected_dist=0.0 → sigmoid=0.5 | CONFIDENT_DIST=-2.0 |
| DINOv2 resolution | IndexError crash | ViT-S/14 → 37×37 vs 512×512 | F.interpolate upsample |
| dyn_mask bool cast | Wrong FG/BG | float32 -1.0 → True | `raw > 0` |

## 5. Open Issues & Next Steps

### Depth Alignment (Priority 1)
- MoGe relative depth → cross-camera scale 불일치 (cam1-3: 43-129px 투영 오차)
- 해결: SfM(COLMAP) 기반 metric scale alignment

### BG Separation (Priority 2)
- Option A: BG pixel을 mask out (black) → FG 집중
- Option B: 2-stage (BG 50ep freeze → FG joint)

### Quality Improvements
- ViT-S → ViT-B/L (feature quality)
- w_depth_reg 활성화 (depth 정합 후)
- CoTracker (RAFT drift 대체)

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
    → 4D Gaussian Splatting: canonical 3DGS + SE(3) motion bases (K=10)
    → gsplat rasterization → L_rgb + L_track + L_mask + L_feat
```

Key design: SE(3) basis trajectories (not MLP). K shared rigid transforms, per-Gaussian linear combination. Multi-view consistent by construction.

---

*MonoFusion M5t2 PoC | Project Status | 2026-04-02*
