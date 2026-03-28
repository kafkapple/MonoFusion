# MonoFusion M5t2 Experiments

## Experiment Naming Convention

```
{method}_{run_id}_{key_param}
  mf_001_baseline       # MonoFusion first run
  mf_002_nobg           # FG-only ablation
  mf_003_6view          # 6-camera experiment
  fl_001_baseline       # FaceLift baseline
```

## Evaluation Metrics (FG-Masked Required)

| Metric | Full Image | FG-Masked | Notes |
|--------|-----------|-----------|-------|
| PSNR ↑ | Report | **Primary** | Mouse = 2.5% of image |
| SSIM ↑ | Report | **Primary** | Structural quality |
| LPIPS ↓ | Report | **Primary** | Perceptual quality |
| Mask IoU ↑ | — | **Required** | Geometry accuracy |
| Temporal warp ↓ | — | Report | Motion quality |

## Experiment Log

| ID | Date | Config | Status | PSNR_FG | IoU | Notes |
|----|------|--------|--------|---------|-----|-------|
| mf_001 | 260327 | 60fr×4cam, 5K FG, 10K BG, 10 bases | pending | — | — | First PoC |

## Directory Structure

```
experiments/
├── README.md (this file)
├── {exp_id}/
│   ├── config.yaml
│   ├── notes.md
│   ├── metrics/
│   ├── renders/
│   └── videos/
```

---

*MonoFusion Experiments | 2026-03-27*
