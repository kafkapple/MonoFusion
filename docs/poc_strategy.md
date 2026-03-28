# MonoFusion PoC Strategy (MoA Deliberation Final)

> **UPDATE 260327**: Initial "synthetic preprocessing" strategy was audited and rejected.
> Current strategy: **Original pipeline with multi-env conda** (see SETUP_GUIDE.md).
> Decisions below superseded where they conflict with feedback_audit.md.

## Decision Summary (2-Loop MoA, 3-Model Consensus) — PARTIALLY SUPERSEDED

| Decision | Value | Confidence | Status |
|----------|-------|------------|--------|
| Preprocessing | ~~Skip → synthetic~~ → **Original pipeline (TAPNet, DUSt3R+MoGe)** | — | **REVISED** |
| Resolution | 512×512 (no crop) | 5/5 |
| GPU | A6000 (GPU 4/5) + cu118 → Blackwell Phase 2 | 5/5 |
| Minimal inputs | 5: RGB + masks + cameras + BG cloud + FG tracks | 5/5 |
| Frames | 60 (glb_step configurable) | 4/5 |
| Views | 4 (expandable to 6) | 5/5 |
| Depth loss | OFF (weight=0) | 5/5 |
| Feature loss | OFF (weight=0) | 5/5 |

## Key Code Analysis Findings

### Camera Views (casual_dataset.py:259)
- 4 hardcoded (`cam_ids=[3,21,23,25]`) — Panoptic specific
- Model architecture has **NO view count constraint**
- 6-view: modify `cam_ids` list + `cam_iddddds` dict

### Resolution (casual_dataset.py:434, iphone_dataset.py:369)
- Dynamic from image file — NOT hardcoded
- 512×512 works as-is

### FPS/Frames (casual_dataset.py:203-212)
- `glb_step` (default 3): frame stride
- `start`/`end`: frame range
- Fully configurable

### Scene Scale (init_utils.py)
- Data-adaptive (97th percentile of point cloud bounds)
- No human-scale assumptions

### Training Core CUDA Requirements
- gsplat: **pip package** (no custom compilation)
- All 11 CUDA-heavy submodules: **preprocessing ONLY**

## Execution Plan

### Phase 1: Environment + Synthetic Validation (Day 1-2)

```bash
# gpu03, A6000
export CUDA_VISIBLE_DEVICES=4
conda create -n monofusion python=3.10 -y
conda activate monofusion
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install gsplat einops scipy imageio open3d timm
```

Verify:
```bash
python -c "import gsplat, torch; print(f'gsplat {gsplat.__version__}, CUDA {torch.version.cuda}')"
```

### Phase 2: Data Conversion (Day 2-3)

1. M5t2 RGBA → RGB + alpha mask extraction
2. Camera format: opencv_cameras.json → MonoFusion Dy_train_meta.json
3. BG point cloud: Depth-Anything-V2 or COLMAP
4. FG tracks: RAFT optical flow + triangulation
5. Reprojection sanity check (mandatory)

### Phase 3: Training PoC (Day 3-5)

1. Wire data into casual_dataset.py (custom loader)
2. Modify cam_ids for M5t2 cameras
3. Set depth_loss=0, feature_loss=0
4. 60 frames × 4 views, 10K iterations
5. Loss convergence check

### Phase 4: Scaling (Week 2+)

1. More frames (120, 300)
2. 6-view experiment
3. Real preprocessing (DINOv2, depth)
4. Blackwell migration

## Memory Estimate (A6000, 48GB)

```
Gaussians: ~50K × 60 params × 4 bytes ≈ 12 MB
Feature maps: 4 × 512 × 512 × 64 × 4 ≈ 256 MB
Rendered images: 4 × 512 × 512 × 4 × 4 ≈ 16 MB
Gradient buffers: ~2× forward ≈ 550 MB
Total peak: ~2-4 GB (A6000 충분)
```

## Escalation to Blackwell (Triggers)

- Confirmed OOM on A6000 with profiler evidence
- Training validated, scaling to 500+ frames
- Multi-scene batch training required

---

*MonoFusion PoC Strategy | MoA 2-Loop Deliberation | 2026-03-27*
