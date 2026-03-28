# Audit Feedback — MonoFusion PoC (2026-03-27)

## Critical Findings (3-model audit consensus)

### Do NOT substitute core pipeline components without discussion:
- TAPNet → RAFT: **WRONG** (장기 추적 ≠ frame-pair flow)
- DUSt3R+MoGe → Depth-Anything: **WRONG** (metric depth ≠ relative depth)
- Missing data → dummy fallback: **WRONG** (데이터 위조)

### Acceptable:
- RGBA alpha → SAM2 대체: **OK** (동일한 정보)
- DINOv2 upsample: **Marginal** (DinoFeature submodule 확인 필요)

## Corrective Actions

1. 3-env 구조 구성 (JAX/PyTorch/RAFT)
2. 원본 TAPNet/BootsTAPIR 설치
3. DUSt3R+MoGe 원본 파이프라인
4. casual_dataset.py 원본 복원 + 별도 data adapter 작성
5. environment.yml × 3 + SETUP_GUIDE.md 작성

## Pattern (교훈)
- PoC "빠른 진행" ≠ 핵심 컴포넌트 임의 대체
- CUDA 호환성 문제 → 별도 env로 해결, 컴포넌트 대체하지 않음
- 사용자 명시적 요청 위반 → process failure

---

*Audit Feedback | 2026-03-27*
