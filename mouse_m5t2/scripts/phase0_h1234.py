# no-split: H1-H4 falsification measurements
"""
Phase 0 Measurements — H1/H2/H3/H4 falsification tests

Run each test, report raw numbers, do NOT interpret.
Pre-specified falsification thresholds written in the summary function.

Inputs:
  --ckpt_final  : results_v10b/checkpoints/best.ckpt (epoch 299)
  --ckpt_mid    : results_v10b/checkpoints/epoch_0250.ckpt (epoch 250)
  --data_root   : markerless_v7 dir (for masks)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def log(msg):
    print(f"[h1234] {msg}", flush=True)


def cont_6d_to_rmat(rot6d):
    a1 = rot6d[..., :3]; a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def compute_transforms(rots, transls, coefs, t):
    transls_t = transls[:, t]          # (K, 3)
    rots_t = rots[:, t]                # (K, 6)
    transls_blend = torch.einsum("pk,ki->pi", coefs, transls_t)  # (G, 3)
    rots_blend = torch.einsum("pk,ki->pi", coefs, rots_t)         # (G, 6)
    rmat = cont_6d_to_rmat(rots_blend)                             # (G, 3, 3)
    return rmat, transls_blend


def warp_means(means, rmat, transl):
    return torch.einsum("pij,pj->pi", rmat, means) + transl


# ----------------------- H1 & H3 -----------------------
def h1_h3_scale_and_means_diff(ckpt_final, ckpt_mid):
    log("=" * 60)
    log("H1 & H3: Scale shrinkage and means displacement — epoch 250 vs 299")
    log("=" * 60)

    sd_f = torch.load(ckpt_final, map_location="cpu", weights_only=False)["model"]
    sd_m = torch.load(ckpt_mid,   map_location="cpu", weights_only=False)["model"]

    means_f = sd_f["fg.params.means"].float()
    means_m = sd_m["fg.params.means"].float()
    scales_f = torch.exp(sd_f["fg.params.scales"].float())
    scales_m = torch.exp(sd_m["fg.params.scales"].float())

    N_f = means_f.shape[0]
    N_m = means_m.shape[0]
    log(f"  FG count: epoch 250 = {N_m}, epoch 299 = {N_f}")
    log(f"  Count diff: +{N_f - N_m} Gaussians added via densification in 49 epochs")

    # Can only compare Gaussians that exist in both. If count differs, take minimum.
    # Warning: without index tracking, we can't 1:1 align after densification.
    # Instead compare aggregate statistics.

    # H1: scale distribution shift
    s_f_mean = scales_f.mean(-1)  # (N_f,)
    s_m_mean = scales_m.mean(-1)  # (N_m,)
    log("")
    log("  H1 — Scale distribution (mean of 3 axes per Gaussian):")
    log(f"    epoch 250: median={s_m_mean.median():.5f}, p95={np.percentile(s_m_mean, 95):.5f}, "
        f"max={s_m_mean.max():.5f}")
    log(f"    epoch 299: median={s_f_mean.median():.5f}, p95={np.percentile(s_f_mean, 95):.5f}, "
        f"max={s_f_mean.max():.5f}")
    delta_median = float(s_f_mean.median() - s_m_mean.median())
    delta_p95 = float(np.percentile(s_f_mean, 95) - np.percentile(s_m_mean, 95))
    log(f"    Δ median: {delta_median:+.5f}   Δ p95: {delta_p95:+.5f}")
    if abs(delta_median) < 1e-4 and abs(delta_p95) < 1e-4:
        log("    → H1 MEASUREMENT: scales essentially UNCHANGED in epoch 250→299")
        h1_verdict = "frozen"
    elif delta_median < 0:
        log("    → H1 MEASUREMENT: scales DECREASED (Gaussians shrinking)")
        h1_verdict = "shrinking"
    else:
        log("    → H1 MEASUREMENT: scales INCREASED")
        h1_verdict = "growing"

    # H3: means displacement (only works if counts match)
    log("")
    log("  H3 — Means displacement:")
    if N_f == N_m:
        delta = (means_f - means_m).norm(dim=-1)  # (N,)
        log(f"    epoch 250→299 per-Gaussian displacement:")
        log(f"      median={delta.median():.5f}, p95={np.percentile(delta, 95):.5f}, "
            f"max={delta.max():.5f}")
        scale_median = s_f_mean.median().item()
        ratio = float(delta.median() / scale_median)
        log(f"    Ratio (displacement median / scale median) = {ratio:.3f}")
        log(f"    (< 0.1 means 'frozen', > 1 means 'moving by more than own size')")
        h3_verdict = {"median_delta": float(delta.median()),
                      "p95_delta": float(np.percentile(delta, 95)),
                      "max_delta": float(delta.max()),
                      "scale_median": scale_median,
                      "ratio_delta_over_scale": ratio}
    else:
        log(f"    N differs ({N_m} vs {N_f}), cannot 1:1 compare. Comparing bounding boxes.")
        bbox_f = means_f.max(0).values - means_f.min(0).values
        bbox_m = means_m.max(0).values - means_m.min(0).values
        log(f"    bbox epoch 250: {bbox_m.tolist()}")
        log(f"    bbox epoch 299: {bbox_f.tolist()}")
        h3_verdict = {"bbox_mid": bbox_m.tolist(), "bbox_final": bbox_f.tolist(),
                      "n_mid": N_m, "n_final": N_f}

    return {"h1": {"verdict": h1_verdict,
                   "scale_median_mid": float(s_m_mean.median()),
                   "scale_median_final": float(s_f_mean.median()),
                   "delta_median": delta_median,
                   "delta_p95": delta_p95},
            "h3": h3_verdict}


# ----------------------- H2 -----------------------
def h2_effective_displacement_across_t(ckpt_final):
    log("=" * 60)
    log("H2: Per-Gaussian effective displacement across all t (warped spread)")
    log("=" * 60)

    sd = torch.load(ckpt_final, map_location="cpu", weights_only=False)["model"]
    means = sd["fg.params.means"].float()
    coefs = F.softmax(sd["fg.params.motion_coefs"].float(), dim=-1)
    rots = sd["motion_bases.params.rots"].float()
    transls = sd["motion_bases.params.transls"].float()
    G = means.shape[0]
    T = rots.shape[1]

    log(f"  Warping {G} Gaussians across {T} frames...")
    # (T, G, 3) warped positions
    warped_all = torch.zeros(T, G, 3)
    for t in range(T):
        rmat, trans = compute_transforms(rots, transls, coefs, t)
        warped_all[t] = warp_means(means, rmat, trans)

    # Per-Gaussian bbox across t
    g_min = warped_all.min(dim=0).values  # (G, 3)
    g_max = warped_all.max(dim=0).values  # (G, 3)
    g_range = (g_max - g_min).norm(dim=-1)  # (G,) total spatial range
    g_std = warped_all.std(dim=0).norm(dim=-1)  # (G,) temporal std

    log(f"  Per-Gaussian spatial range (max dist across t):")
    log(f"    median={g_range.median():.3f}, p25={np.percentile(g_range, 25):.3f}, "
        f"p75={np.percentile(g_range, 75):.3f}, p95={np.percentile(g_range, 95):.3f}, "
        f"max={g_range.max():.3f}")
    log(f"  Per-Gaussian temporal std:")
    log(f"    median={g_std.median():.3f}, p95={np.percentile(g_std, 95):.3f}")

    # Canonical scale for reference
    scales = torch.exp(sd["fg.params.scales"].float()).mean(-1)
    log(f"  For reference: Gaussian scale median = {scales.median():.5f}")
    log(f"  Ratio (displacement range median / scale median) = "
        f"{float(g_range.median() / scales.median()):.1f}")

    # Also compute "blob size at a single frame" = std of all warped positions at t=30
    t30_pos = warped_all[30]  # (G, 3)
    t30_std = t30_pos.std(dim=0).norm()
    log(f"")
    log(f"  At t=30: global std of all Gaussian positions (cloud spread) = {t30_std:.3f}")
    log(f"    (compared to BG scene bbox range ~ 500, this is {100*t30_std/500:.1f}% of scene)")

    return {
        "per_gaussian_range": {
            "median": float(g_range.median()),
            "p25": float(np.percentile(g_range, 25)),
            "p75": float(np.percentile(g_range, 75)),
            "p95": float(np.percentile(g_range, 95)),
            "max": float(g_range.max()),
        },
        "scale_median": float(scales.median()),
        "range_over_scale_median": float(g_range.median() / scales.median()),
        "t30_cloud_std": float(t30_std),
    }


# ----------------------- H4 -----------------------
def h4_cam1_mask_quality(data_root, cam_names, n_frames=80):
    log("=" * 60)
    log("H4: Per-cam mask statistics across all frames")
    log("=" * 60)

    result = {}
    for cam in cam_names:
        mask_dir = Path(data_root) / "masks" / cam
        paths = sorted(mask_dir.glob("*.npz"))[:n_frames]
        if not paths:
            log(f"  {cam}: no masks found at {mask_dir}")
            continue
        fg_fracs = []
        nonzero_frames = 0
        for i, p in enumerate(paths):
            raw = np.load(p)["dyn_mask"]
            mask = (raw > 0).astype(np.float32)
            frac = float(mask.mean())
            fg_fracs.append(frac)
            if frac > 0.001: nonzero_frames += 1
        fg_fracs = np.array(fg_fracs)
        log(f"  {cam}: n_frames={len(paths)}")
        log(f"    FG fraction: mean={fg_fracs.mean()*100:.2f}%, "
            f"median={np.median(fg_fracs)*100:.2f}%, "
            f"min={fg_fracs.min()*100:.2f}%, max={fg_fracs.max()*100:.2f}%")
        log(f"    frames with nonzero mask (>0.1%): {nonzero_frames}/{len(paths)}")
        result[cam] = {
            "n_frames": len(paths),
            "fg_frac_mean": float(fg_fracs.mean()),
            "fg_frac_median": float(np.median(fg_fracs)),
            "fg_frac_min": float(fg_fracs.min()),
            "fg_frac_max": float(fg_fracs.max()),
            "nonzero_frames": int(nonzero_frames),
        }
    return result


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_final", required=True)
    ap.add_argument("--ckpt_mid",   required=True)
    ap.add_argument("--data_root",  required=True)
    ap.add_argument("--output",     required=True)
    args = ap.parse_args()

    all_results = {}
    all_results.update(h1_h3_scale_and_means_diff(args.ckpt_final, args.ckpt_mid))
    all_results["h2"] = h2_effective_displacement_across_t(args.ckpt_final)
    all_results["h4"] = h4_cam1_mask_quality(
        args.data_root,
        ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"],
    )

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("")
    log(f"Wrote {out}")

    # Final falsification summary (pre-specified thresholds)
    log("")
    log("=" * 60)
    log("FALSIFICATION SUMMARY (thresholds fixed before measurement)")
    log("=" * 60)

    h1 = all_results["h1"]
    log(f"  H1 (scale frozen): delta_median={h1['delta_median']:.5f}")
    log(f"    verdict: {h1['verdict']}")

    h2 = all_results["h2"]
    log(f"  H2 (motion spread): per-Gaussian range median = {h2['per_gaussian_range']['median']:.3f}")
    log(f"    Gaussian scale median = {h2['scale_median']:.5f}")
    log(f"    ratio = {h2['range_over_scale_median']:.1f}× (>10 means spreading far beyond own size)")

    h3 = all_results["h3"]
    if "ratio_delta_over_scale" in h3:
        log(f"  H3 (means frozen): displacement/scale ratio = {h3['ratio_delta_over_scale']:.3f}")
        log(f"    (<0.1 = frozen, >1 = moving)")

    h4 = all_results["h4"]
    log(f"  H4 (cam1 mask): FG fractions per cam:")
    for cam, stats in h4.items():
        log(f"    {cam}: mean={stats['fg_frac_mean']*100:.2f}%  "
            f"nonzero_frames={stats['nonzero_frames']}/{stats['n_frames']}")


if __name__ == "__main__":
    main()
