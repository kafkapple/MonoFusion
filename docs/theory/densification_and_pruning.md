# Densification & Pruning — Theory and MonoFusion Implementation

> Verified against `flow3d/trainer.py`, `flow3d/configs.py` (2026-03-30)

---

## 1. Why Densification?

3D Gaussian Splatting 초기화 시 Gaussian 수는 고정. 하지만 장면의 디테일 수준은 영역마다 다름:
- 마우스 귀 끝: 정밀한 형상 → Gaussian 밀도 높아야
- 빈 배경: 적은 Gaussian으로 충분
- 텍스처 많은 영역: 더 많은 Gaussian 필요

**Densification** = 학습 중 gradient 정보를 바탕으로 Gaussian을 **자동 추가/제거**하여 장면 복잡도에 맞게 적응.

---

## 2. 3DGS 표준 vs MonoFusion

### 2.1 Densification (Split / Clone)

```
매 70 step마다 (step 50~2000):

각 Gaussian에 대해:
  grad_avg = 누적 2D gradient / 관측 횟수

  if grad_avg > 0.0002:                    ← gradient 높음 = 부정확
    if scale > 0.01 OR screen_radius > 0.05:
      → SPLIT: Gaussian을 2개로 분할       ← 큰 것을 쪼갬
      - 위치: 원본 중심 ± scale 방향으로 이동
      - scale: 원본의 1/1.6배로 축소
    else:
      → CLONE: 동일한 Gaussian 복제         ← 작은 것을 복사
      - 위치: 원본과 같은 위치
      - 모든 파라미터 복사
```

| | 3DGS 표준 (Kerbl 2023) | MonoFusion |
|--|----------------------|------------|
| Gradient threshold | 0.0002 | 0.0002 (같음) |
| Split 조건 | grad & scale_big | grad & (scale_big \| radius_big) |
| Clone 조건 | grad & ~scale_big | grad & ~scale_big (같음) |
| Scale threshold | scene-dependent | 0.01 (고정) |
| Screen radius threshold | — | 0.05 |
| Stop step | varies | **2000** |

### 2.2 Pruning (Cull)

```
매 70 step마다 (step 50~4000):

제거 조건 (OR):
  1. opacity < 0.1                    ← 거의 투명한 Gaussian 제거
  2. scale > 0.5 × scene_scale       ← 비정상적으로 큰 Gaussian
  3. screen_radius > 0.15            ← 화면의 15% 이상 차지
```

| | 3DGS 표준 | MonoFusion |
|--|----------|------------|
| Opacity threshold | **0.005** | **0.1** (20× 더 공격적) |
| Scale threshold | varies | 0.5 × scene_scale |
| Screen threshold | — | 0.15 |
| Stop step | varies | **4000** |

**MonoFusion이 더 공격적인 이유**: Sparse-view (4 카메라)에서 floater artifact가 심함.
관측되지 않는 영역에 Gaussian이 생겨도 pruning이 제거하지 않으면 렌더링 품질 저하.

### 2.3 Opacity Reset (MonoFusion 고유)

```
매 2100 step (30 × control_every):
  모든 Gaussian의 opacity를 ~0.08로 리셋

목적:
  - 불필요한 Gaussian이 높은 opacity를 유지하는 것을 방지
  - 리셋 후 필요한 Gaussian만 다시 opacity 상승
  - 불필요한 것은 opacity < 0.1 → 다음 pruning에서 제거
```

이것은 3DGS 표준에 없는 **MonoFusion 고유 기법**. Dynamic scene에서 시간에 따라 보이는 영역이 변하므로, 주기적 리셋으로 "죽은" Gaussian을 재활용.

---

## 3. M5t2 PoC의 Densification 타임라인

```
Step    0~50:    warmup (densification 비활성)
Step   50~2000:  SPLIT + CLONE 활성 (Gaussian 수 증가)
                 매 2100 step: opacity reset
Step 2000~4000:  pruning만 (CULL only, 수 감소 가능)
Step 4000~9000:  모든 control 비활성 (Gaussian 수 고정)
```

### 실제 Gaussian 수 변화 (V2 학습)
```
Epoch  0 (step    0): 11,721 (초기화)
Epoch  7 (step  105): 14,363 (+22%)
Epoch 19 (step  285): 24,539 (+109%)
Epoch 23 (step  345): 27,549 (+135%)
...
Epoch ~133 (step 2000): densification 종료
Epoch ~267 (step 4000): pruning 종료
Epoch 600 (step 9000): 최종 (고정)
```

---

## 4. 논문 대비 PoC 차이 종합

### 학습 파이프라인 차이

| Component | 논문 | PoC | 영향 | 해결 |
|-----------|------|-----|------|------|
| DINOv2 | ViT-L, PCA 32d | ViT-S, raw 384d | feat loss 불균형 | w_feat ↓ 또는 PCA |
| Depth | MoGe (metric) | DA-V2 (relative) | 3D 스케일 모호 | w_depth 활성화 |
| Depth loss | w=0.5 | w=0.0 | 기하 정규화 없음 | **w=0.5 활성화** |
| Tracking | BootsTAPIR | RAFT+mask | 드리프트 | CoTracker |
| Frame rate | 원본 fps | stride=10 (3fps) | 20px/frame 이동 | stride=5 |
| 학습 steps | 30K+ | 9K | 수렴 부족 | 더 학습 |
| Camera | DUSt3R | opencv_cameras.json | 호환 (검증됨) | OK |
| Mask | SAM2 | RGBA alpha | GT급 | OK |

### 데이터 특성 차이

| | Panoptic (논문) | M5t2 (PoC) |
|--|----------------|-----------|
| 피사체 | 사람 (느림) | 마우스 (빠름) |
| FG 비율 | 10-30% | **2-3%** (매우 작음) |
| 프레임 수 | 150+ | 60 |
| 카메라 | 31-cam (dense) | 4-cam (sparse) |
| 해상도 | 1080p+ | 512×512 |
| 움직임 | ~2px/frame | **~20px/frame** (stride=10) |

---

## 5. 즉시 적용 권장 설정 변경

```python
# LossesConfig 수정 제안:
w_rgb: 5.0        # 유지
w_feat: 0.3       # 1.5 → 0.3 (384d는 32d 대비 12배 큰 gradient)
w_depth_reg: 0.5  # 0.0 → 0.5 (활성화)
w_track: 2.0      # 유지
w_mask: 7.0       # 유지
```

---

↑ MOC: `docs/README.md` | ↔ Related: `monofusion_architecture.md`, `scene_flow_and_tracking.md`, `training_guide.md`

*Densification & Pruning Theory | MonoFusion M5t2 PoC | 2026-03-30*
