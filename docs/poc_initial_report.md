# MonoFusion PoC Initial Report

## Overview

MonoFusion을 M5t2 markerless mouse 데이터셋에 적용하기 위한 초기 분석 보고서.
MoA (Mixture-of-Agents) 3-model deliberation + 코드 분석 기반.

- **Paper**: https://arxiv.org/html/2507.23782v1
- **Repo**: https://github.com/Z1hanW/MonoFusion
- **Server**: gpu03 (Blackwell 98GB × 4 + A6000 48GB × 4)
- **Dataset**: M5t2 (3600 frames × 6 cams, 512×512 RGBA)

---

## 1. MonoFusion Method Summary

- **Task**: 4개 sparse-view 카메라로부터 4D dynamic scene 복원
- **Representation**: Canonical frame 3D Gaussians + SE(3) motion bases (weighted linear combination)
- **Preprocessing**: DUSt3R (multi-view depth) + MoGe (monocular depth) + SAM2 (masks) + DINOv2 (features) + TAPNet (tracking)
- **Training**: ~30min on single A6000, 17K FG + 47K BG Gaussians, 28 motion bases
- **Rendering**: gsplat 기반 ~30fps

## 2. Data Format Compatibility Analysis

### M5t2 Dataset (Source)

| Aspect | Value |
|--------|-------|
| Frames | 3600 (train 0-1197, val 1198-2395, test 2396-3599) |
| Cameras | 6 (cam_000 ~ cam_005) |
| Resolution | 512×512 |
| Format | RGBA PNG |
| Camera params | opencv_cameras.json per frame (w2c 4×4, fx=fy=549, cx=cy=256) |
| Structure | Per-frame dirs: `{frame_id}/images/cam_XXX.png` |
| FPS | 100fps (DANNCE origin) |
| Location | `/node_data/joon/data/preprocessed/FaceLift_mouse/M5` |

### MonoFusion Expected Format

| Aspect | Value |
|--------|-------|
| Frames | Flexible (start/end/stride configurable) |
| Cameras | **4 hardcoded** in Panoptic loader (indices [3,21,23,25]) |
| Resolution | **Flexible** (dynamic from image, Panoptic default 512×288) |
| Format | RGB (PNG/JPG) |
| Camera params | w2c 4×4 matrix + K 3×3 intrinsics (Dy_train_meta.json) |
| Structure | Per-sequence: `images/{seq_name}/{frame_id}.jpg` |
| FPS | Flexible (from metadata, default 15fps casual) |

### Conversion Requirements

| Item | M5t2 → MonoFusion | Difficulty |
|------|-------------------|-----------|
| RGBA → RGB | Drop alpha (preserve as mask) | Easy |
| 512×512 → 512×288 | Center crop + cy adjustment | Easy |
| 6 → 4+ cameras | Select by angular spread OR **modify loader for 6 views** | Easy-Medium |
| Per-frame → per-seq dirs | Restructure script | Easy |
| opencv_cameras.json → Dy_train_meta.json | Format mapping | Medium |
| Camera convention | OpenCV (both use w2c) — verify Z-axis | Medium (silent failure risk) |

## 3. View Count / Resolution / FPS Flexibility (Code Analysis)

### Camera View Count

**Finding**: Panoptic loader에서 `cam_ids = [3, 21, 23, 25]` 하드코딩 (casual_dataset.py:259).

**6-view 지원 가능 여부**: YES — `cam_ids` 리스트와 `cam_iddddds` 딕셔너리를 수정하면 됨.
모델 아키텍처 자체에 view 수 제약 없음 (각 view를 독립적으로 렌더링/비교).
4→6 view 변경 시 영향:
- 학습 데이터 50% 증가 → 학습 시간 비례 증가
- 초기화 point cloud 밀도 증가 → 잠재적 품질 향상
- Memory 사용량 batch_size × view_count 비례 증가

**권장**: PoC는 4-view로 시작, 성공 후 6-view 실험.

### Resolution

**Finding**: 동적으로 이미지에서 읽음. 고정값 아님.
- `factor` 파라미터로 다운샘플링 지원
- 512×512 그대로 사용 가능 (crop 불필요할 수 있음)
- 단, Panoptic 벤치마크는 512×288 기준

**권장**: 512×512 그대로 시도 → Panoptic 비교 시 512×288.

### FPS / Frame Count

**Finding**: `glb_step` (default 3)으로 stride, `start`/`end`로 범위 설정.

| Scenario | glb_step | Frames | Effective FPS | Note |
|----------|----------|--------|--------------|------|
| PoC minimal | 10 | ~360 | ~10fps equiv | 빠른 검증용 |
| PoC standard | 3 | ~1200 | ~33fps equiv | 논문과 유사 |
| Full dataset | 1 | 3600 | 100fps | 최대 품질 |

**권장**: PoC는 glb_step=10 (360 frames)으로 시작.

## 4. CUDA / Environment Analysis

### CUDA Extension Breakdown

| Module | Purpose | CUDA Files | Min CUDA | Architecture | PoC 필수? |
|--------|---------|-----------|----------|-------------|----------|
| **gsplat** | Gaussian rendering | pip package | Any | Auto | **YES** |
| XFormers | DINOv2 attention | 48+ | 11.0+ | Auto | Preprocessing only |
| DROID-SLAM | Visual SLAM | 3 | 11.0 | Auto | Preprocessing only |
| **LieTorch** | Lie group ops | 1 | 11.0 | **SM 6.0-7.5 hardcoded** | Preprocessing only |
| Dust3R CuRoPE | Positional encoding | 1 | 11.0 | Auto-detect | Preprocessing only |
| SEA-RAFT | Optical flow | N/A | **12.2** | N/A | Preprocessing only |

### Key Insight: Training vs Preprocessing 분리

**Training core**에 필요한 CUDA extension:
- gsplat (pip install — **커스텀 컴파일 불필요!**)
- PyTorch 표준 CUDA ops

**Preprocessing**에 필요한 CUDA extensions (11개 submodule 대부분):
- DUSt3R, MoGe, SAM2, DINOv2, TAPNet, DROID-SLAM...
- 이들은 **전처리 결과물만 있으면 학습 시 불필요**

→ **전략**: 전처리를 별도로 실행하거나, 기존 M5t2 RGBA alpha를 mask로 재활용하면
   CUDA compilation 리스크를 대폭 줄일 수 있음.

### Environment Strategy

```
Target GPU: A6000 (GPU 4/5, sm_86, 48GB)
CUDA Toolkit: 11.8 (cu118) — middle ground
PyTorch: 2.0.x+cu118 또는 2.1.x+cu118
gsplat: pip install (pre-built wheels)

Blackwell (GPU 1/3, sm_120, 98GB):
  - gsplat이 sm_120 지원하면 사용 가능 (확인 필요)
  - PyTorch 2.10.0+cu128 기존 env 활용 가능성
```

## 5. GPU / Server Resource Status (2026-03-27)

| GPU | Model | VRAM | Status |
|-----|-------|------|--------|
| 0 | RTX PRO 6000 Blackwell | 98GB | Occupied (97% used) |
| 1 | RTX PRO 6000 Blackwell | 98GB | **Free** |
| 2 | RTX PRO 6000 Blackwell | 98GB | Partial (34GB used) |
| 3 | RTX PRO 6000 Blackwell | 98GB | **Free** |
| 4 | RTX A6000 | 48GB | **Free** |
| 5 | RTX A6000 | 48GB | Mostly free |
| 6 | RTX A6000 | 48GB | Partial |
| 7 | RTX A6000 | 48GB | Occupied |

Disk: /home 395GB free, /node_data 1.4TB free — 충분.

## 6. Risk Matrix

| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| CUDA extension compilation (preprocessing) | 60% | Medium | 전처리 분리, alpha mask 재활용 |
| gsplat sm_120 미지원 (Blackwell) | 30% | Low | A6000 fallback |
| Camera convention 무음 오류 | 40% | High | Reprojection sanity check (필수) |
| Scene scale (mouse vs human) | 20% | Low | Scene scale data-adaptive 확인됨 |
| Memory OOM (3600 frames) | 50% | Medium | 360 frame subset |
| DUSt3R/MoGe depth 품질 (mouse scale) | 40% | Medium | RGBA depth 활용 검토 |

## 7. Execution Plan

### Phase 0: Environment Setup (Day 1)
1. gpu03에 conda env 생성 (monofusion, python 3.10)
2. PyTorch + gsplat 설치
3. gsplat compilation 확인 (A6000 → Blackwell 순차 시도)
4. RAPIDS 제외한 requirements 설치

### Phase 1: Data Conversion (Day 1-2)
1. M5t2 → MonoFusion 포맷 변환 스크립트 작성
2. 4 camera 선택 (angular spread) + 6-view option
3. RGBA alpha → mask 추출
4. Camera format 변환 + reprojection 검증

### Phase 2: Preprocessing (Day 2-3)
1. DUSt3R depth 생성 (or skip if alpha depth available)
2. DINOv2 feature 추출
3. TAPNet tracking
4. 결과물 검증

### Phase 3: Training PoC (Day 3-5)
1. 360 frames × 4 views로 minimal training
2. Loss convergence 확인
3. 시각화 결과 생성
4. 6-view / full resolution 실험

## 8. MoA Deliberation Confidence

3-model consensus (Claude Sonnet + Gemini 2.5 Pro + GPT-4o):
- **Overall: 2.5/5** (CUDA compilation 불확실성)
- **Data conversion: 4/5** (straightforward)
- **Training convergence: 3/5** (scene domain 차이)
- **코드 분석 후 상향 조정: 3/5** (gsplat pip 설치 가능, training core CUDA 리스크 낮음)

---

*MonoFusion PoC Initial Report | 2026-03-27*
