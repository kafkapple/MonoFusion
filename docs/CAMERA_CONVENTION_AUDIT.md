# MonoFusion M5t2 — Camera Convention Audit & Fix Plan

> Critical: 모든 V5 실험의 FG Gaussian 위치가 GT와 불일치하는 근본 원인.
> Date: 2026-04-01 | Status: **FIXED** (2026-04-01)
> Fix: metadata `camera_convention` flag + conditional inv() skip

## 1. 문제 요약

**증상**: 모든 V5 실험(V5e~V5h)에서 렌더링된 mouse 위치와 GT mouse 위치가 불일치.
색상/톤은 수렴하지만 형상과 위치가 맞지 않음.

**정량적 증거** (V5e, cam0 frame0):

| | X (pixel) | Y (pixel) |
|---|-----------|-----------|
| FG Gaussian 중심 | 211 | 178 |
| GT Mouse 중심 | 328 | 306 |
| **오차** | −117 | −128 |

4개 카메라 모두에서 Gaussian projection이 **이미지 밖**으로 나감:
- cam00: (932, −758) vs GT (328, 306)
- cam01: (−116, −280) vs GT (151, 313)
- cam02: (233, −65) vs GT (170, 304)
- cam03: (624, 2384) vs GT (325, 272)

## 2. Root Cause: w2c/c2w Convention 혼동

MonoFusion 원본 코드에서 camera matrix naming이 뒤집혀 있음.

### Convention 추적 (7개 지점)

```
[1] Dy_train_meta.json
    md['w2c'][t][c] = 진짜 world-to-camera (4×4)
    ↓
[2] casual_dataset.py:373 — load_known_cameras()
    w2c = np.linalg.inv(md['w2c'][t][c])
    → 결과: c2w (camera-to-world). 변수명 w2c = WRONG NAME
    ↓
[3] casual_dataset.py:418-430
    c2ws.append(w2c)  → list 이름은 맞음 (c2ws)
    return traj_c2w    → 반환값도 c2w
    ↓
[4] casual_dataset.py:464
    self.w2cs = w2cs   → c2w를 'w2cs'로 저장 (WRONG NAME)
    ↓ ↓
    ↓ [6] dance_glb.py:426
    ↓     w2cs = train_dataset.get_w2cs()  → c2w를 받음
    ↓     w2cs_fuse.append(w2cs)
    ↓     ↓
    ↓ [6a] dance_glb.py:433
    ↓     run_initial_optim(..., w2cs_fuse[0])  → c2w를 전달
    ↓     ↓
    ↓ [6b] init_utils.py:512
    ↓     project_2d_tracks(tracks, Ks, w2cs)  → T_cw 파라미터로 사용
    ↓     einsum("tij,tnj->tni", T_cw, world_pts)  → c2w로 world→cam?
    ↓     ↓
    ↓ [6c] dance_glb.py:441
    ↓     SceneModel(Ks_fuse, w2cs_fuse, ...)  → model.w2cs = c2w
    ↓     ↓
    ↓ [7] scene_model.py:296
    ↓     gsplat.rasterization(viewmats=w2cs)  → gsplat은 w2c 기대, c2w 수신
    ↓
[5] casual_dataset.py:629 — get_tracks_3d()
    c2ws = torch.linalg.inv(self.w2cs)
    → inv(c2w_actual) = w2c_actual. 변수명 c2ws = WRONG
    ↓
[5a] data/utils.py:123
    tracks_3d = einsum(c2ws, cam_points)  → c2w 기대, w2c 수신!
    → 3D track 위치가 잘못됨 ← ROOT CAUSE
```

### 핵심 분석

| 지점 | 실제 값 | 변수명 | 소비자 기대 | 일치? |
|------|--------|--------|------------|------|
| [4] self.w2cs | c2w | w2cs | — | ❌ 이름 |
| [5] c2ws (inv) | w2c | c2ws | c2w (utils.py) | ❌ **BUG** |
| [6b] w2cs → T_cw | c2w | w2cs/T_cw | ? (검증필요) | ❓ |
| [7] gsplat viewmats | c2w | w2cs | w2c | ❌ 또는 내부보상? |

### 왜 렌더링은 "어느 정도" 작동하는가

gsplat이 c2w를 받아도 완전 붕괴하지 않는 이유:
- Gaussian 3D 위치도 같은 뒤집힌 convention으로 초기화됨
- 렌더링(gsplat)과 초기화(init)가 **같은 잘못된 convention**을 공유
- 결과: 내부적으로 일관되지만, **GT 이미지 좌표와 불일치**

### V5i 수정 시도 결과

`casual_dataset.py:629`에서 `inv()` 제거 → **NaN at step 5**.
이유: get_tracks_3d만 수정하면 init_optim의 project_2d_tracks와 불일치.

## 3. 수정 전략

### 옵션 A: Source에서 수정 (권장)

`casual_dataset.py:373`에서 inv()를 제거하여 self.w2cs에 진짜 w2c를 저장.
**모든 downstream을 일괄 수정해야 함.**

변경 필요 지점:

| 파일:라인 | 현재 | 수정 후 |
|----------|------|--------|
| casual_dataset.py:373 | `w2c = inv(md['w2c'])` | `w2c = md['w2c']` (inv 제거) |
| casual_dataset.py:629 | `c2ws = inv(self.w2cs)` | `c2ws = inv(self.w2cs)` (유지 — 이제 진짜 c2w) |
| scene_model.py render | viewmats=w2cs | 유지 (이제 진짜 w2c) |
| init_utils.py:512 | project_2d_tracks(Ks, w2cs) | **검증 필요** — T_cw가 w2c여야 |
| trainer.py compute_losses | 동일 | **검증 필요** |

**리스크**: 원본 MonoFusion의 다른 데이터셋(EgoExo, Panoptic)과 호환성 깨질 수 있음.
그 데이터셋에서 `md['w2c']`가 실제로 c2w를 저장하고 있을 수 있음 (DUSt3R 출력).

### 옵션 B: 데이터 변환에서 수정

`convert_m5t2.py`에서 Dy_train_meta.json 생성 시 w2c 대신 c2w를 저장.
MonoFusion 원본 코드를 수정하지 않고, 입력 데이터를 맞춤.

변경 필요 지점:

| 파일 | 현재 | 수정 후 |
|------|------|--------|
| convert_m5t2.py | `"w2c": w2c` 저장 | `"w2c": c2w` 저장 (inv(w2c)) |
| 전처리 재실행 | — | 필수 (새 meta 생성) |

**리스크**: 직관에 반함 (w2c 키에 c2w 저장). 하지만 원본 코드와 호환.

### 옵션 C: 원본 동작 검증 후 결정

1. MonoFusion 원본 데이터(Panoptic Studio)로 원본 코드 실행
2. 원본에서 Gaussian이 GT 위치와 일치하는지 확인
3. 일치 → 옵션 B (데이터 변환 문제). 불일치 → 옵션 A (코드 버그).

## 4. 검증 방법 (다음 세션)

### Quick Check (15분)

```python
# MonoFusion 원본 데이터의 md['w2c']가 진짜 w2c인지 c2w인지 확인
# Panoptic Studio 데이터의 camera center = -R^T @ t (w2c) or [:3,3] (c2w)
import json, numpy as np
md = json.load(open("Panoptic_Dy_train_meta.json"))
w2c = np.array(md['w2c'][0][0])
# If w2c is really w2c: camera center = -R^T @ t
center_if_w2c = -w2c[:3,:3].T @ w2c[:3,3]
# If w2c is actually c2w: camera center = w2c[:3,3]
center_if_c2w = w2c[:3,3]
# Check which one gives a reasonable camera position (~meters from origin)
```

### Visual Check (30분)

수정 후 init만 실행 (학습 없이) → FG Gaussian을 GT 이미지에 투영 → 위치 일치 확인.

## 5. 영향 범위

| 항목 | 영향 |
|------|------|
| 모든 V5 실험 (V5e~V5h) | Gaussian 위치 잘못됨 → PSNR 한계 |
| 이전 hyperparameter 튜닝 | 잘못된 geometry에서의 최적화 → 실제 효과 불명 |
| FG-only PSNR metric 도입 | 여전히 유효 (올바른 평가 방법) |
| w_mask 분석 | 여전히 유효 (mask loss 지배는 실제 문제) |
| Opacity reset 분석 | 여전히 유효 (Gaussian bloat 문제) |

## 6. Fix Applied (2026-04-01)

### Root Cause (Corrected Understanding)

The bug is NOT in the original MonoFusion code. It is a **data format mismatch**:
- Original MonoFusion's `load_known_cameras` was designed for DUSt3R, which stores c2w in the 'w2c' field
- The inv() at line 373 correctly converts DUSt3R's c2w → real w2c
- Our `convert_m5t2.py` stores **real w2c** from OpenCV calibration
- inv(real_w2c) = c2w → self.w2cs = c2w → all downstream consumers receive wrong convention

### Evidence

| Test | Hypothesis A (real w2c) | Hypothesis B (c2w) |
|------|------------------------|-------------------|
| Camera spread | 6.97m (ring ✅) | 0.39m (clustered ❌) |
| Projection (4 cams) | 4/4 IN FRAME ✅ | 3/4 OUT, 1 BEHIND ❌ |
| FaceLift docs | opencv_cameras.json = w2c ✅ | — |

### Fix: Metadata-driven convention flag

Files modified:
1. `convert_m5t2.py` — add `"camera_convention": "w2c"` to Dy_train_meta.json
2. `casual_dataset.py:361-373` — read flag, skip inv() when convention='w2c'
3. `Dy_train_meta.json` (m5t2_v5, m5t2_poc) — added field

### Verification Results

| Metric | Target | V5i (fix) | V5e (buggy) |
|--------|--------|-----------|-------------|
| NaN within 60 steps | None | ✅ None | ✅ None |
| Loss (ep2) | — | **1.83** | 7.24 |
| self.w2cs = real w2c | Match JSON | ✅ | ❌ (was c2w) |
| Scene center projection | IN FRAME | ✅ 4/4 | ❌ OUT |

### Next: V5i 50ep run (GPU4), then V5j 300ep full run

---

*MonoFusion M5t2 PoC | Camera Convention Audit | 2026-04-01 (FIXED)*
