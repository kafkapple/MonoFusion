# no-split: lightweight diagnostic bypassing gsplat JIT
"""
Phase 0 Diagnostics (lightweight version — no gsplat import, no rendering)

D6-lite: Direct torch.load of V10b checkpoint, inspect state_dict
         - FG/BG Gaussian counts (from tensor shapes)
         - FG opacity percentiles
         - FG scale percentiles
         - FG:BG ratio

D8-lite: Load CasualDataset, call get_tracks_3d with num_samples={5K, 18K, 50K}
         - Verify if S1 (num_fg bounded by TAPNet) holds empirically

Bypasses: SceneModel, gsplat (no rendering). D5/D7 run later.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[phase0-lite] {msg}", flush=True)


def d6_checkpoint_stats_direct(ckpt_path):
    """Parse V10b checkpoint state_dict directly — no SceneModel instantiation."""
    log("=" * 60)
    log("D6-lite: V10b checkpoint (direct state_dict inspection)")
    log("=" * 60)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    log(f"  checkpoint top-level keys: {list(ckpt.keys())}")

    # Identify FG and BG parameter groups by key naming
    fg_keys = [k for k in sd.keys() if k.startswith("fg.") or "fg." in k]
    bg_keys = [k for k in sd.keys() if k.startswith("bg.") or "bg." in k]

    log(f"  FG keys: {fg_keys[:10]}")
    log(f"  BG keys: {bg_keys[:10]}")

    result = {}

    def stats_for(prefix, label):
        # Means is the definitive count
        means_key = None
        for k in sd.keys():
            if k == f"{prefix}.params.means" or k.endswith(f"{prefix}.params.means"):
                means_key = k
                break
        if means_key is None:
            for k in sd.keys():
                if f"{prefix}.params" in k and "mean" in k:
                    means_key = k
                    break
        if means_key is None:
            log(f"  {label}: no means tensor found")
            return None
        means = sd[means_key]
        n = means.shape[0]
        log(f"  {label}: num_gaussians = {n:,}  (key={means_key})")

        # Opacity
        for k in sd.keys():
            if f"{prefix}.params" in k and "opac" in k:
                opac_logit = sd[k].float()
                opac = torch.sigmoid(opac_logit).numpy()
                pcts = np.percentile(opac, [5, 25, 50, 75, 95])
                near_zero = int((opac < 0.05).sum())
                saturated = int((opac > 0.95).sum())
                log(f"    opacity percentiles [5,25,50,75,95]: {pcts.tolist()}")
                log(f"    opacity near-zero (<0.05): {near_zero}/{n}")
                log(f"    opacity saturated (>0.95): {saturated}/{n}")
                result[f"{label}_opacity_pcts"] = pcts.tolist()
                result[f"{label}_opacity_near_zero"] = near_zero
                result[f"{label}_opacity_saturated"] = saturated
                break

        # Scales (log space)
        for k in sd.keys():
            if f"{prefix}.params" in k and "scale" in k:
                log_scales = sd[k].float()
                scales = torch.exp(log_scales).numpy()
                scale_mean = scales.mean(-1)
                pcts = np.percentile(scale_mean, [5, 25, 50, 75, 95])
                tiny = int((scale_mean < 0.001).sum())
                large = int((scale_mean > 0.1).sum())
                log(f"    scale mean percentiles [5,25,50,75,95]: {pcts.tolist()}")
                log(f"    scale tiny (<1mm): {tiny}/{n}")
                log(f"    scale large (>10cm): {large}/{n}")
                result[f"{label}_scale_pcts"] = pcts.tolist()
                result[f"{label}_scale_tiny"] = tiny
                result[f"{label}_scale_large"] = large
                break

        result[f"{label}_count"] = n
        return n

    n_fg = stats_for("fg", "FG")
    n_bg = stats_for("bg", "BG")

    if n_fg and n_bg:
        ratio = n_bg / max(n_fg, 1)
        log(f"")
        log(f"  📊 FG:BG ratio = 1 : {ratio:.1f}")
        result["bg_over_fg_ratio"] = ratio
        if ratio > 20:
            log(f"  ❌ STARVATION CONFIRMED: BG dominates Gaussian budget (Gemini hypothesis)")
            result["starvation_verdict"] = "confirmed"
        elif ratio > 5:
            log(f"  🟡 PARTIAL STARVATION: BG significantly outweighs FG")
            result["starvation_verdict"] = "partial"
        else:
            log(f"  ✅ Budget balanced")
            result["starvation_verdict"] = "balanced"

    return result


def d8_get_tracks_3d_shape(data_root):
    """Call get_tracks_3d with different num_samples to verify S1 empirically."""
    log("=" * 60)
    log("D8-lite: get_tracks_3d() return shape at num_samples={5K, 18K, 50K}")
    log("=" * 60)

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))

    # Apply monkey-patch for mouse_m5t2 format
    sys.path.insert(0, str(repo_root / "mouse_m5t2"))
    from train_m5t2 import patch_casual_dataset, create_datasets
    patch_casual_dataset()

    # Use create_datasets to get the proper M5t2 dataset setup
    class Args: pass
    args = Args()
    args.data_dir = str(data_root)
    args.feat_dir_name = "dinov2_features_pca32_norm"
    args.num_fg = 5000   # starting value, irrelevant for get_tracks_3d call
    args.num_bg = 10000

    try:
        datasets = create_datasets(args)
        ds = datasets[0]  # cam0 dataset
        log(f"  dataset num_frames={ds.num_frames}")
    except Exception as e:
        log(f"  ❌ create_datasets failed: {e}")
        return {"error": str(e)}

    results = {}
    for num_samples in [5000, 18000, 50000]:
        try:
            tracks_3d, vis, invis, conf, col, feat = ds.get_tracks_3d(num_samples)
            P = int(tracks_3d.shape[0])
            T = int(tracks_3d.shape[1])
            vis_per_frame = vis.sum(dim=0).numpy().astype(int).tolist()
            log(f"  num_samples={num_samples:>6}: returned P={P:>6}  T={T}")
            log(f"    vis per frame [first 10, last 10]: {vis_per_frame[:10]} ... {vis_per_frame[-10:]}")
            cano_t_guess = int(np.argmax(vis_per_frame))
            log(f"    argmax(visible) = frame {cano_t_guess}  (cano_t candidate)")
            results[str(num_samples)] = {
                "requested": num_samples,
                "returned_P": P,
                "num_frames": T,
                "cano_t_guess": cano_t_guess,
                "vis_at_cano_t": int(vis_per_frame[cano_t_guess]),
                "vis_per_frame_sample": vis_per_frame[::10],
            }
        except Exception as e:
            log(f"  num_samples={num_samples}: ERR {e}")
            results[str(num_samples)] = {"error": str(e)}

    log("")
    log("  🔍 S1 VERDICT:")
    if all(isinstance(results.get(str(n)), dict) and "returned_P" in results[str(n)]
           for n in [5000, 18000, 50000]):
        p5k = results["5000"]["returned_P"]
        p18k = results["18000"]["returned_P"]
        p50k = results["50000"]["returned_P"]
        log(f"  P(5K)={p5k}, P(18K)={p18k}, P(50K)={p50k}")
        if p18k <= 1.2 * p5k + 100:
            log("  ❌ V12b CONFIRMED DEAD — num_fg cap is NOT the bottleneck")
            log("     TAPNet track count is the bottleneck → V12b would be a no-op")
            results["verdict"] = "V12b_dead"
        elif p18k < 10000:
            log("  🟡 V12b PARTIAL — P grows slightly but < 18K")
            results["verdict"] = "V12b_partial"
        else:
            log("  ✅ V12b VIABLE — num_fg cap actually matters")
            results["verdict"] = "V12b_viable"
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log(f"torch version: {torch.__version__}")

    all_results = {}

    log("")
    all_results["d6"] = d6_checkpoint_stats_direct(args.checkpoint)

    log("")
    try:
        all_results["d8"] = d8_get_tracks_3d_shape(Path(args.data_root))
    except Exception as e:
        log(f"D8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_results["d8"] = {"error": str(e)}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
