# MonoFusion M5t2 PoC — Documentation

## Documents

| File | Description | Status |
|------|-------------|--------|
| [poc_initial_report.md](poc_initial_report.md) | Initial analysis: dataset comparison, CUDA, risks | Complete |
| [poc_strategy.md](poc_strategy.md) | MoA deliberation final strategy | Complete |
| [env_setup.md](env_setup.md) | ~~Single-env setup~~ (DEPRECATED → SETUP_GUIDE.md) | Superseded |
| [mouse_m5t2/envs/SETUP_GUIDE.md](../mouse_m5t2/envs/SETUP_GUIDE.md) | Multi-env setup (3 conda envs) | Current |
| [progress_log.md](progress_log.md) | Step-by-step progress with verification | In Progress |
| [experiments/README.md](experiments/README.md) | Experiment log + metrics framework | In Progress |
| [experiments/mf_001_notes.md](experiments/mf_001_notes.md) | First PoC run notes | Pending |
| [data_compatibility.md](data_compatibility.md) | Data format conversion: m5t2 ↔ MonoFusion pipeline | Current |
| [core_architecture.md](core_architecture.md) | Pipeline components, loss decomp, track format, RAFT fix | Current |
| [training_guide.md](training_guide.md) | Pre-training safeguards, monitoring metrics, improvement roadmap | Current |
| [theory/monofusion_architecture.md](theory/monofusion_architecture.md) | SE(3) motion bases: architecture, K selection, scene flow extraction | Current |
| [theory/scene_flow_and_tracking.md](theory/scene_flow_and_tracking.md) | Theory: 4D-GS as scene flow, RAFT vs TAPNet, visualization code | Current |
| [theory/densification_and_pruning.md](theory/densification_and_pruning.md) | Densification/pruning: 3DGS standard vs MonoFusion, thresholds, timeline | Current |
| [audit_pretraining.md](audit_pretraining.md) | Pre-training audit: expected_dist bug, pipeline completeness, execution plan | Current |
| [hardcoding_issues.md](hardcoding_issues.md) | casual_dataset.py hardcoding issues: glb_step, range, paths, query step | Current |
| Research Notes | `/Users/joon/results/MonoFusion/research_notes.md` — Bug log, decisions, prevention | Current |

## Quick Links

- **Paper**: https://arxiv.org/html/2507.23782v1
- **Repo**: https://github.com/Z1hanW/MonoFusion
- **Data (gpu03)**: `/node_data/joon/data/monofusion/m5t2_poc/`
- **Viz (gpu03)**: `/node_data/joon/data/monofusion/m5t2_poc/viz/`
- **Scripts**: `~/dev/MonoFusion/mouse_m5t2/scripts/`
- **Conda envs**: `monofusion_pytorch` (training) / `monofusion_jax` (TAPNet) — see `mouse_m5t2/envs/SETUP_GUIDE.md`

---

*MonoFusion M5t2 PoC | 2026-03-27*
