# MonoFusion V8 — Single-Variable Isolation Experiments

## Motivation

V7c (all params changed simultaneously) cannot attribute improvement to any single variable.
This plan isolates each variable against a reproducible baseline to identify which factors matter.

Deliberation: 5-iteration audit+devil (3 models × 5 rounds). Passed with EXECUTE verdict.
Key debate: Devil argued fundamental mismatch (4 sparse cams, small mouse). Builder countered
with MonoFusion being designed for casual video + V5j loss=2.06 as empirical proof.

## Success Criteria (pre-defined)

| Category | PSNR Delta vs V8a | Interpretation |
|----------|-------------------|----------------|
| Significant | > +1.0 dB | Variable matters, pursue further |
| Marginal | +0.3 ~ +1.0 dB | Possible effect, needs confirmation |
| No effect | < +0.3 dB | Variable not a bottleneck at this scale |

## Experiments

### V8a — Reproducible Baseline (Control)

```bash
CC=x86_64-conda-linux-gnu-gcc CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 \
python -u mouse_m5t2/train_m5t2.py \
    --data_root /node_data/joon/data/monofusion/markerless_v7 \
    --output_dir /node_data/joon/data/monofusion/markerless_v7/results_v8a \
    --num_fg 5000 --num_bg 10000 --num_motion_bases 10 --num_epochs 300 \
    --w_feat 1.5 --w_mask 7.0 --w_depth_reg 0.0 \
    --feat_ramp_start_epoch 5 --feat_ramp_end_epoch 50 \
    --max_gaussians 100000 --stop_densify_pct 0.6 \
    --densify_xys_grad_threshold 0.0002 \
    --feat_dir_name dinov2_features_pca32_norm \
    --seed 42 --bg_lr_config frozen \
    --wandb_name v8a_baseline_seed42
```

- FG=5K, BG=10K, bases=10, BG frozen (LR~1e-9), seed=42
- Matches V7b config but with fixed seed for reproducibility
- ~3h on A40

### E1 — BG LR Isolation (Is BG frozen the bottleneck?)

```bash
# Same as V8a except: --bg_lr_config gt
--bg_lr_config gt \
--wandb_name e1_bg_lr_unfrozen \
--output_dir .../results_e1
```

- Only change: BG LR frozen → paper spec (means=1.6e-4, feats=1e-4)
- Known limitation: BG=10K is 120x below paper's 1.2M
- Tests "direction": does BG learning help AT ALL?
- If >1dB → follow up with E1b (BG=50K unfrozen)

### E2 — FG Count Isolation (Does FG density matter?)

```bash
# Same as V8a except: --num_fg 18000
--num_fg 18000 \
--wandb_name e2_fg_18k \
--output_dir .../results_e2
```

- Only change: FG 5K → 18K (paper spec)
- Known confound: different track sampling at init (acknowledged, not eliminated)
- VRAM: ~8GB (safe on A40)

### E3 — Motion Bases Isolation (Does motion expressiveness matter?)

```bash
# Same as V8a except: --num_motion_bases 28
--num_motion_bases 28 \
--wandb_name e3_bases_28 \
--output_dir .../results_e3
```

- Only change: bases 10 → 28 (paper spec)
- Tests if 10 bases are sufficient for mouse motion
- VRAM: ~8GB (safe on A40)

## Execution Plan

```
V7c (running, ~2h remaining) → check PSNR + visual
  ↓
V8a baseline (3h, GPU4)
  ↓
E1 (3h) → E2 (3h) → E3 (3h)  [sequential on GPU4]
  ↓
Compare PSNR: which variable(s) show >1dB improvement?
  ↓
If E1 significant → E1b (BG=50K unfrozen)
If all fail → data-driven PIVOT decision
```

Total GPU time: ~12h (V8a + E1 + E2 + E3)

## Assumptions (explicitly stated)

1. V8a at FG=5K/BG=10K may be undercapacity — this plan tests "direction" not "magnitude"
2. Paper hyperparameters may not transfer to mouse data
3. E2 has unavoidable init confound (different track sampling)
4. BG=10K is too small for full BG learning test — E1b is contingent follow-up
5. Temporal metrics deferred to post-PoC phase (adding one = scope creep)
6. BG "frozen" uses LR~1e-9, not requires_grad=False (functionally equivalent for PoC)

## Decision Matrix (post-experiment)

| E1 (BG LR) | E2 (FG) | E3 (Bases) | Action |
|:-----------:|:-------:|:----------:|--------|
| Significant | Any | Any | BG LR is key → E1b (BG=50K) |
| No effect | Significant | Any | FG density is key → scale FG further |
| No effect | No effect | Significant | Bases matter → combine with more FG |
| No effect | No effect | No effect | PIVOT: approach may be fundamentally limited |
| Significant | Significant | Any | Both matter → combined E4 (BG+FG) |

## References

- [MONOFUSION_STATUS.md](../MONOFUSION_STATUS.md) — Project status, V5-V8 history
- [LESSONS_LEARNED.md](../LESSONS_LEARNED.md) — §9 (single-variable), §10 (bare except), §12 (success criteria)
- [CAMERA_CONVENTION_AUDIT.md](../CAMERA_CONVENTION_AUDIT.md) — Camera bug that consumed 8 experiments
- MoA+Audit deliberation (260402): BG frozen identified as primary cause
- 5-iteration deliberate --audit --devil (260405): Plan approved with EXECUTE verdict
- V5j (deleted dataset): loss 2.06 — empirical proof pipeline CAN work
