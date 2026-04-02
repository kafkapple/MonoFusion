# MonoFusion M5t2 PoC — Documentation

## Documents

| File | Description |
|------|-------------|
| [MONOFUSION_STATUS.md](MONOFUSION_STATUS.md) | Project status, experiment history, breakthrough analysis, next steps |
| [REFERENCE.md](REFERENCE.md) | Technical reference: pipeline, architecture, data format, training config |
| [CAMERA_CONVENTION_AUDIT.md](CAMERA_CONVENTION_AUDIT.md) | Camera convention audit detail (root cause, evidence, fix) |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Key lessons: geometry-first, convention traps, pipeline interdependency |

## Quick Links

- **Data (V5)**: `gpu03:/node_data/joon/data/monofusion/m5t2_v5/`
- **Best model**: `results_v5j/checkpoints/best.ckpt` (loss 2.06, 300ep)
- **Results (local)**: `~/results/MonoFusion/v5j_viz/`
- **Scripts**: `mouse_m5t2/scripts/`
- **Conda**: `monofusion` (cu118, A6000) — `CC=x86_64-conda-linux-gnu-gcc`

---

*MonoFusion M5t2 PoC | 2026-04-02*
