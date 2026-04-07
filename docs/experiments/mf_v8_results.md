# MonoFusion V8 — Single-Variable Isolation Results

> **Date**: 2026-04-07 | **Status**: ✅ Complete (4 experiments, ~11h GPU time on A40)
> **TL;DR**: BG frozen was the bottleneck. Unfreezing BG → **+16.7 dB PSNR** (9.07 → 25.80).

## Summary Table

| Exp | Variable Changed | Best Loss | Final PSNR | Δ vs V8a | Gaussians | Verdict |
|:---:|------------------|----------:|-----------:|:--------:|----------:|---------|
| **V8a** | (baseline, BG frozen) | 12.52 | 9.07 | — | 295,083 | baseline |
| **🏆 E1** | `--bg_lr_config gt` (BG unfrozen) | **4.95** | **25.80** | **+16.73 dB** | 283,610 | ⭐ Massive |
| E2 | `--num_fg 18000` (FG density) | 17.19 | 6.44 | −2.63 dB | 334,499 | Worse |
| E3 | `--num_motion_bases 28` | 21.21 | 7.07 | −2.00 dB | 296,048 | Worse |

**Significance threshold (pre-defined)**: >1.0 dB = significant. **E1 = 16.7× threshold.**

## Experiment Configs (Common)

```bash
--num_epochs 300 --seed 42
--w_feat 1.5 --w_mask 7.0 --w_depth_reg 0.0
--feat_ramp_start_epoch 5 --feat_ramp_end_epoch 50
--max_gaussians 100000 --stop_densify_pct 0.6
--densify_xys_grad_threshold 0.0002
--feat_dir_name dinov2_features_pca32_norm
--data_root /node_data/joon/data/monofusion/markerless_v7
```

| Variable | V8a | E1 | E2 | E3 |
|----------|:---:|:--:|:--:|:--:|
| `--num_fg` | 5000 | 5000 | **18000** | 5000 |
| `--num_bg` | 10000 | 10000 | 10000 | 10000 |
| `--num_motion_bases` | 10 | 10 | 10 | **28** |
| `--bg_lr_config` | frozen | **gt** | frozen | frozen |

**BG LR difference**:
- `frozen` (BGLRConfig): means=1.6e-9, feats=1e-15 (effectively zero learning)
- `gt` (BGLRGTConfig): means=1.6e-4, feats=1e-4 (paper spec, 5 orders of magnitude higher)

## Key Finding: BG Learning is the Critical Lever

### Quantitative

E1 alone explains the FG ghost problem from V5–V7 series:

| Series | BG state | Best Full PSNR |
|--------|----------|---------------:|
| V5b–V5h | BG frozen or num_bg=0 | 9.24 (V5e) – 13.36 (V5j on deprecated dataset) |
| V7a/b/c | BG frozen | 8.91 – 15.55 (V7c, but unstable) |
| **V8 E1** | **BG unfrozen** | **25.80** |

### Qualitative (visual inspection at ep299)

| Exp | Reconstruction quality |
|-----|-----------------------|
| V8a | 노이즈 덩어리 — BG가 학습되지 않아 초기화 상태 그대로, FG도 ghost 수준 |
| **E1** | **선명한 BG 재현 + 마우스 형체 명확** — 거의 GT 수준 |
| E2 | V8a와 유사 — FG 18K 추가가 의미 없음 |
| E3 | V8a와 유사 — 28 motion bases도 의미 없음 |

### Why BG Frozen Was the Bottleneck

When BG Gaussians cannot move/update:
1. **Gradient leakage**: FG gradient signal partially absorbed by static BG Gaussians
2. **Feature mismatch**: BG DINOv2 features stay at initialization, never align with target
3. **Mask compositing failure**: rendered_imgs = FG · mask + BG · (1-mask). Static BG = persistent compositing error
4. **FG over-corrects**: Tries to compensate for broken BG → ghost artifacts on the mouse

E2/E3 didn't help because they added capacity (more FG, more bases) to a pipeline whose **fundamental constraint** was elsewhere.

## Decision Matrix Resolution

From `mf_v8_isolation_plan.md` decision matrix:

| E1 | E2 | E3 | Action |
|:--:|:--:|:--:|--------|
| **Significant** | No effect | No effect | ✅ **BG LR is key → E1b (BG=50K unfrozen)** |

## Reproducibility

- **Seed**: 42 (5-source deterministic: torch+cuda+numpy+random+cudnn)
- **Hardware**: GPU 4 (NVIDIA A40, sm_86, 49 GB)
- **Env**: `monofusion` conda (cu118, gcc 11), `CC=x86_64-conda-linux-gnu-gcc`
- **Wandb**: `kafkapple-joon-kaist/monofusion-m5t2`
  - V8a: `qewn9v38`
  - E1: `[run from 01:00-03:35]`
  - E2: `[run from 03:35-06:29]`
  - E3: `[run from 06:29-09:12]`

## Next Steps: V9 Plan

Based on E1 success, scale up BG learning:

### V9a — E1b: BG=50K unfrozen (validate scaling)
- Same as E1 but `--num_bg 50000`
- Hypothesis: more BG capacity + learning → even higher PSNR
- Risk: gradient instability at higher BG count (V7c had this with BG=985K)

### V9b — Combined: E1 + cleaner code path
- Apply post-V8 fixes from review:
  - P2 (bg_psnr valid_masks) ✅ already committed (59054e6)
  - P3 (camera_convention guard) ✅ already committed
  - Consider: feat_loss masking consistency (compute_losses vs compute_stat_losses)
- Re-run E1 config with updated code

### V9c — Resolution / Quality scaling
- Now that the pipeline works at PSNR 25.80, push for higher quality:
  - DINOv2 ViT-S → ViT-B/L (better features)
  - More motion bases (paper: 28, but only after BG works)
  - Longer training (300 → 500 ep with cosine schedule)

### V9d — Apply to other datasets
- markerless_v7 → markerless_v8 (more frames, more cameras)
- Multi-mouse scenes
- Reproduce on FaceLift M5 dataset for cross-validation

## Confounds & Caveats

1. **V8a is not a perfect V7b reproduction** — seed=42 was new. V8a final loss=12.68 (V7b was 7.71). This means V8a's "baseline" is different from V7b. However, **E1's improvement is so large (+16.7 dB)** that any reasonable baseline still shows a clear win.

2. **feat_loss is unmasked** in `compute_losses` (active path). Both V8a and E1 use the same code, so relative comparison is valid. But absolute PSNR may improve further with masked feat loss.

3. **PSNR is full-image** (not FG-only). V8a's low PSNR (9.07) reflects BG noise dominating the metric. E1's high PSNR (25.80) reflects whole-scene reconstruction.

4. **bg_psnr metric was buggy in V8a** (missing valid_masks at line 1031). Fixed in commit `59054e6` for E1+. V8a's bg_psnr numbers should be discarded.

## Files & Artifacts

- **Train logs**: `/node_data/joon/data/monofusion/markerless_v7/results_{v8a,e1,e2,e3}/train.log`
- **Checkpoints**: `results_{exp}/checkpoints/best.ckpt`, `last.ckpt`, `epoch_0299.ckpt`
- **Previews**: `results_{exp}/previews/epoch_0299_gt_vs_pred.png`
- **Local viz**: `~/dev/MonoFusion/viz/v8_results/{v8a,e1,e2,e3}_ep299.png`
- **Loss curves**: `results_{exp}/loss_curve.json`
- **Wandb runs**: https://wandb.ai/kafkapple-joon-kaist/monofusion-m5t2

## Related Documents

- [V8 isolation plan (pre-experiment)](mf_v8_isolation_plan.md)
- [Project status](../MONOFUSION_STATUS.md)
- [Lessons learned](../LESSONS_LEARNED.md) — §9 (single-variable), §12 (success criteria)
- [V5e first convergent run](mf_v5e_results.md)

## Git Commits (V8 series)

- `75f9451` — Phase 0 bug fixes + V8 isolation plan + docs update
- `59054e6` — Fix bg_psnr metric + camera_convention guard + add V8 runner script

---

*MonoFusion M5t2 PoC | V8 Isolation Results | 2026-04-07*
*Verdict: BG learning unlocks 4D GS for mouse reconstruction. PSNR breakthrough 9 → 26 dB.*
