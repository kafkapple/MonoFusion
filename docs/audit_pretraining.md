# Pre-Training Audit Report — MonoFusion M5t2 PoC

> 3-model audit (Claude/Gemini/GPT) × 2 rounds, 2026-03-29

---

## Executive Summary

MonoFusion M5t2 PoC: 4 cameras × 60 frames, mouse dynamic reconstruction.
Training prerequisites are **not met**. 2 critical blockers, 3 major issues.

---

## Findings

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | 🔴 Critical | GPU03 코드 미동기화 + tapir_raft/ broken (expected_dist=0.0) | Fix: rsync + 재생성 |
| 2 | 🔴 Critical | expected_dist=0.0 → 모든 visible 트랙 학습에서 버려짐 | Fix: CONFIDENT_DIST=-2.0 |
| 3 | 🟡 Major | visible_counts threshold collapse (all-zero → quantile=0 → 무효 통과) | Guard 추가 권장 |
| 4 | 🟡 Major | DINOv2 실제 ViT-S/14(384dim), 문서는 ViT-L/14(1024dim) | 문서 수정 |
| 5 | 🟡 Major | cam02 zero FG tracks → empty batch 가능 | per-camera weight |
| 6 | 🟢 Minor | mask `== 1` + grid_sample bilinear → 경계 실패 가능 | 원본 설계 의도 |
| 7 | 🟢 Minor | Depth: DA-V2(relative), not MoGe | w_depth_reg=0.0이라 loss 무관 |
| 8 | 🟢 Minor | symlink 의존성 fragile | PoC 허용 |

---

## Finding #2 Deep Dive: expected_dist=0.0

### 수학적 구조

`parse_tapir_track_info` (flow3d/data/utils.py:53-66):

```python
visibility  = 1 - sigmoid(occ_logit)      # "이 점이 보일 확률"
confidence  = 1 - sigmoid(expected_dist)   # "트랙커 예측 신뢰도"
valid_visible = visibility * confidence > 0.5   # 2중 게이트
```

설계 의도: 확률 × 신뢰도 > 0.5 → 충분히 확신할 때만 학습 신호로 사용.

### 왜 0.0이면 버려지는가

| expected_dist | sigmoid | confidence | visible 트랙(vis=0.88) | occluded 트랙(vis≈0) |
|---------------|---------|------------|----------------------|---------------------|
| 0.0 | 0.5 | **0.5** | 0.88×0.5=**0.44 < 0.5 ❌** | 0×0.5=0 ❌, 1×0.5=**0.5 ≤ 0.5 ❌** |
| -2.0 | 0.12 | **0.88** | 0.88×0.88=**0.77 > 0.5 ✅** | 0×0.88=0 ❌, 1×0.88=**0.88 > 0.5 ✅** |

confidence=0.5는 "완전 중립". 어떤 visibility 값이든 곱이 0.5를 초과할 수 없음 (vis < 1.0).

결과: **모든 RAFT 트랙이 "neither" 상태** → 학습 신호 0 → FG Gaussian이 학습되지 않음.

### Fix 효과 (expected_dist=-2.0)

- **visible 트랙**: valid_visible=True → L_track 학습 신호 활성 ✅
- **occluded 트랙**: valid_invisible=True → "여기 없어야 한다" 추가 신호 ✅
- **risk**: 마스크 경계 노이즈 → mask 2px erosion으로 완화

### 왜 -2.0인가?

VISIBLE_LOGIT도 -2.0. 대칭 설계: occ_logit과 expected_dist가 동일한 sigmoid 기반 공식을 공유하므로, 동일한 값(-2.0)이 "높은 확신"을 의미. TAPNet 원본에서는 tracker가 자체적으로 expected_dist를 출력하지만, RAFT는 confidence 개념이 없으므로 고정값 사용.

---

## Multi-Query RAFT: Seed 구조

### query_frames=[0, 15, 30, 45, 59]

| Query | Seed | Forward 추적 | Backward 추적 | 목적 |
|-------|------|-------------|--------------|------|
| F0 | F0 마스크에서 512점 | F0→F59 | — | 초기 위치 추적 |
| F15 | F15 마스크에서 512점 | F15→F59 | F15→F0 | 드리프트 보상 |
| F30 | F30 마스크에서 512점 | F30→F59 | F30→F0 | 중간 시점 보강 |
| F45 | F45 마스크에서 512점 | F45→F59 | F45→F0 | 후반부 정확도 |
| F59 | F59 마스크에서 512점 | — | F59→F0 | 마지막 위치 추적 |

### 왜 각 카메라(뷰)마다 seed가 다른가

```
cam00 (정면): 마우스 윗면/등 → 정면 실루엣 내 512점 seed
cam01 (측면): 마우스 옆면/다리 → 측면 실루엣 내 512점 seed
cam02 (반대편): 마우스 없음(F30+) → seed 시도하나 대부분 occluded
cam03 (후면): 마우스 뒷면/꼬리 → 후면 실루엣 내 512점 seed
```

각 카메라의 FG 마스크가 다르므로 (다른 각도에서 다른 부분이 보임) seed 위치가 다름.
SE(3) motion bases가 **모든 뷰를 동시에 설명**해야 하므로 3D 일관성 보장.

---

## 실행 계획

| Phase | Step | Command/Action | 선행 조건 |
|-------|------|---------------|----------|
| A | 1. rsync | `rsync -avz scripts/ gpu03:~/dev/MonoFusion/mouse_m5t2/scripts/` | — |
| A | 2. broken 트랙 삭제 | `ssh gpu03 "rm -rf .../tapir_raft/"` | step 1 |
| B | 3. RAFT 재생성 | `generate_raft_tracks.py --query_frames 0 15 30 45 59` | step 2 |
| B | 4. symlink | `ln -sfn tapir_raft tapir` | step 3 |
| C | 5. validate | `validate_training_inputs.py --data_root ...` | step 4 |
| D | 6. train | `mf_001_raft` 학습 시작 | step 5 pass |

---

---

## Temporal Consistency Analysis (MoA 3-model × 2-layer)

### Current Problems

| Problem | Cause | Severity |
|---------|-------|----------|
| Cumulative drift | RAFT = frame-to-frame only, errors chain | 🔴 Critical |
| Low visibility (15-45% at F30) | 55-85% of training signal lost | 🔴 Critical |
| Multi-query = re-seed only | Doesn't fix existing drift | 🟡 Major |
| Mask boundary noise | Edge-uncertain tracks get wrong occ labels | 🟡 Major |

### Improvement Tiers

**Tier 1: CoTracker** (strongest consensus, 2-3 days)
- Long-range tracking (24+ frame window), explicit occlusion
- Expected: 15-45% → 60-80% visibility
- `pip install cotracker`, drop-in replacement

**Tier 2: Bidirectional Flow Consistency** (low cost, immediate)
- Forward(t→t+N) + backward(t+N→t), discard |fwd - bwd| > threshold
- Works with RAFT or CoTracker

**Tier 3: Temporal Smoothness Loss** (training time)
- `L_smooth = ||means(t+1) - 2*means(t) + means(t-1)||²`
- Penalizes jitter in SE(3) basis outputs

**Tier 4: Denser Query Frames** (easy)
- [0,7,15,22,30,37,45,52,59] (9 queries) vs current 5
- Max drift distance halved: 15→7 frames

### Current RAFT Sufficient for PoC?

**Yes** with conditions: expected_dist fixed, cam02 excluded, SE(3) bases compensate.
**No** for production quality → CoTracker required.

---

## 4DGS Completion Timeline (3-model consensus)

| Day | Task | Go/No-Go |
|-----|------|----------|
| 1 | rsync + RAFT regen + validate | All 5 checks pass? |
| 2 | 1cam+20frame 축소 실험 (fast debug) | Loss decreasing? No OOM? |
| 3 | Full 4cam×60frame training (30k iter) | Convergence? |
| 4 | viz_scene_flow.py + render analysis | Quality acceptable? |
| 5 | (if needed) hyperparameter tune + retrain | — |

**Best case: 3 days. Realistic: 5 days. Worst case: 7-11 days (CoTracker switch).**

---

↑ MOC: `docs/README.md` | ↔ Related: `core_architecture.md`, `theory/monofusion_architecture.md`

*Pre-Training Audit | MonoFusion M5t2 PoC | 2026-03-29*
