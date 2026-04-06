# MonoFusion M5t2 PoC — Documentation

## Documents

### Project Status & Reference
| File | Description |
|------|-------------|
| [MONOFUSION_STATUS.md](MONOFUSION_STATUS.md) | Project status, experiment history V5-V8, next steps |
| [REFERENCE.md](REFERENCE.md) | Technical reference: pipeline, architecture, data format, training config |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Key lessons: geometry-first, convention traps, BG frozen trap, experiment design |

### Audits & Root Cause Analysis
| File | Description |
|------|-------------|
| [CAMERA_CONVENTION_AUDIT.md](CAMERA_CONVENTION_AUDIT.md) | Camera convention bug (w2c/c2w mismatch) — root cause, evidence, fix |
| [DEPTH_ALIGNMENT_PLAN.md](DEPTH_ALIGNMENT_PLAN.md) | MoGe relative depth → metric scale alignment plan |

### Experiments
| File | Description |
|------|-------------|
| [experiments/mf_v8_isolation_plan.md](experiments/mf_v8_isolation_plan.md) | V8 single-variable isolation: V8a baseline + E1/E2/E3 |
| [experiments/mf_v5e_results.md](experiments/mf_v5e_results.md) | V5e first convergent run results |

### Architecture & Theory
| File | Description |
|------|-------------|
| [core_architecture.md](core_architecture.md) | Full pipeline: SAM2 → DUSt3R → MoGe → DINOv2 → 4DGS |
| [theory/monofusion_architecture.md](theory/monofusion_architecture.md) | SE(3) motion bases deep dive |
| [theory/scene_flow_and_tracking.md](theory/scene_flow_and_tracking.md) | RAFT + TAPIR tracking |
| [theory/densification_and_pruning.md](theory/densification_and_pruning.md) | 3DGS adaptive density control |
| [training_guide.md](training_guide.md) | Pre-training checklist, safeguards, monitoring |

## Quick Links

- **Data**: `gpu03:/node_data/joon/data/monofusion/markerless_v7/` (CURRENT)
- **Best model (V5j)**: `results_v5j/checkpoints/best.ckpt` (loss 2.06, DEPRECATED dataset)
- **Scripts**: `mouse_m5t2/scripts/` | Training: `mouse_m5t2/train_m5t2.py`
- **Conda**: `monofusion` (cu118, A40) — `CC=x86_64-conda-linux-gnu-gcc`
- **Git**: origin=kafkapple/MonoFusion, upstream=Z1hanW/MonoFusion

## Training Quick Reference

```bash
# V8a baseline (seed=42, BG frozen)
CC=x86_64-conda-linux-gnu-gcc CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 \
python -u mouse_m5t2/train_m5t2.py \
    --data_root /node_data/joon/data/monofusion/markerless_v7 \
    --output_dir .../results_v8a \
    --num_fg 5000 --num_bg 10000 --num_motion_bases 10 --num_epochs 300 \
    --seed 42 --bg_lr_config frozen \
    --w_feat 1.5 --w_mask 7.0 --feat_dir_name dinov2_features_pca32_norm \
    --wandb_name v8a_baseline_seed42
```

Key args: `--seed N` (reproducibility), `--bg_lr_config gt|frozen` (BG learning)

---

*MonoFusion M5t2 PoC | 2026-04-05*
