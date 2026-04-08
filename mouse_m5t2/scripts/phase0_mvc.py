# no-split: MVC — Multi-View Consistency check across 4 cams
"""
MVC (Multi-View Consistency) Test — decisive preprocessing sanity check

Algorithm:
  1. For each source cam: sample N pixels inside GT mask at t=0
  2. Unproject to 3D world using depth at that pixel
  3. Project 3D point to all OTHER cams at t=0
  4. Check if projected pixel lands inside target cam's mask
  5. Report per-pair pass rate (tolerance: 2-3 pixels)

Pass criteria (pre-specified):
  - >95% aligned (Gemini threshold) or >80% (OpenAI threshold) with 2px tolerance → preprocessing sufficient
  - 50-95% aligned → targeted fix for failing cam
  - <50% aligned → markerless_v8 rewrite needed

Usage:
  PYTHONPATH=. python mouse_m5t2/scripts/phase0_mvc.py \
    --data_root /node_data/joon/data/monofusion/markerless_v7 \
    --output /node_data/joon/data/monofusion/markerless_v7/phase0_mvc.json \
    --viz_dir /node_data/joon/data/monofusion/markerless_v7/phase0_mvc_viz \
    --n_samples 500
"""
import argparse
import json
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[mvc] {msg}", flush=True)


def load_mask(data_root, cam_name, t):
    paths = sorted((Path(data_root) / "masks" / cam_name).glob("*.npz"))
    return (np.load(paths[t])["dyn_mask"] > 0).astype(np.float32)


def load_depth(data_root, cam_name, t):
    """Load depth map from markerless_v7 layout: aligned_moge_depth/m5t2/{cam_name}/depth/{t:06d}.npy"""
    candidates = [
        Path(data_root) / "aligned_moge_depth" / "m5t2" / cam_name / "depth",
        Path(data_root) / "aligned_moge_depth" / cam_name / "depth",
        Path(data_root) / "aligned_moge_depth" / cam_name,
    ]
    for d in candidates:
        if d.exists():
            paths = sorted(d.glob("*.npy"))
            if paths and t < len(paths):
                return np.load(paths[t])
    raise FileNotFoundError(f"No depth file found for {cam_name} (checked {candidates})")


def load_Ks_w2cs(data_root, cam_names, n_frames=80):
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    Ks, w2cs = {}, {}
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])
        with open(meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json") as f:
            meta = json.load(f)
        K = np.array(meta["k"], dtype=np.float32).squeeze()
        w2c = np.array(meta["w2c"], dtype=np.float32).squeeze()
        if meta.get("camera_convention") == "c2w":
            w2c = np.array([np.linalg.inv(w) for w in w2c]) if w2c.ndim == 3 else np.linalg.inv(w2c)
        if K.ndim == 2: K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2: w2c = np.tile(w2c[None], (n_frames, 1, 1))
        Ks[cam_name] = K
        w2cs[cam_name] = w2c
    return Ks, w2cs


def unproject(px, py, depth_val, K, w2c):
    """Unproject pixel (px, py) with depth → 3D world point."""
    # Camera ray
    x_cam = (px - K[0, 2]) / K[0, 0] * depth_val
    y_cam = (py - K[1, 2]) / K[1, 1] * depth_val
    z_cam = depth_val
    pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    c2w = np.linalg.inv(w2c)
    pt_world = (c2w @ pt_cam)[:3]
    return pt_world


def project(pt_world, K, w2c, img_wh):
    pt_h = np.concatenate([pt_world, [1.0]])
    pt_cam = (w2c @ pt_h)[:3]
    if pt_cam[2] <= 0:
        return None, None
    pixel = (K @ pt_cam)[:2] / pt_cam[2]
    W, H = img_wh
    in_view = 0 <= pixel[0] < W and 0 <= pixel[1] < H
    return pixel, in_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--viz_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir); viz_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("MVC: Multi-View Consistency test")
    log("=" * 60)
    log(f"  n_samples per cam: {args.n_samples}")
    log(f"  tolerance: {args.tolerance} pixels")
    log(f"  frame: t={args.t}")

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    Ks, w2cs = load_Ks_w2cs(args.data_root, cam_names, 80)

    # Load depth and masks
    depths = {}
    masks = {}
    for cam in cam_names:
        try:
            depths[cam] = load_depth(args.data_root, cam, args.t)
            masks[cam] = load_mask(args.data_root, cam, args.t)
            log(f"  {cam}: depth shape={depths[cam].shape}, mask FG={100*masks[cam].mean():.2f}%")
        except Exception as e:
            log(f"  {cam}: FAILED to load — {e}")
            return

    rng = np.random.RandomState(42)
    W, H = args.img_wh

    all_results = {}
    pair_stats = {}

    for src_cam in cam_names:
        mask = masks[src_cam]
        depth = depths[src_cam]
        # Find mask pixels with valid depth
        fg_ys, fg_xs = np.where(mask > 0.5)
        valid_depth = depth[fg_ys, fg_xs] > 0
        fg_ys = fg_ys[valid_depth]
        fg_xs = fg_xs[valid_depth]
        if len(fg_xs) == 0:
            log(f"  [{src_cam}] NO valid mask pixels with depth")
            continue

        # Sample
        n_sample = min(args.n_samples, len(fg_xs))
        idx = rng.choice(len(fg_xs), n_sample, replace=False)
        sample_px = fg_xs[idx]
        sample_py = fg_ys[idx]
        sample_depth = depth[sample_py, sample_px]

        log("")
        log(f"  [{src_cam}] sampled {n_sample} mask pixels, depth range=[{sample_depth.min():.2f}, {sample_depth.max():.2f}]")

        K_src = Ks[src_cam][args.t]
        w2c_src = w2cs[src_cam][args.t]

        # Unproject all samples
        world_pts = []
        for i in range(n_sample):
            pt = unproject(float(sample_px[i]), float(sample_py[i]),
                           float(sample_depth[i]), K_src, w2c_src)
            world_pts.append(pt)
        world_pts = np.array(world_pts)
        log(f"    3D world points bbox: min={world_pts.min(0).tolist()}, max={world_pts.max(0).tolist()}")

        # Project to all 4 cams (including self as sanity)
        per_target = {}
        for tgt_cam in cam_names:
            K_tgt = Ks[tgt_cam][args.t]
            w2c_tgt = w2cs[tgt_cam][args.t]
            tgt_mask = masks[tgt_cam]

            hits_in_mask = 0
            hits_in_view = 0
            hits_near_mask = 0  # within tolerance
            pixel_distances = []  # distance to nearest mask pixel
            for w_pt in world_pts:
                pixel, in_view = project(w_pt, K_tgt, w2c_tgt, args.img_wh)
                if not in_view:
                    continue
                hits_in_view += 1
                px_int = int(np.clip(pixel[0], 0, W - 1))
                py_int = int(np.clip(pixel[1], 0, H - 1))
                if tgt_mask[py_int, px_int] > 0.5:
                    hits_in_mask += 1
                    hits_near_mask += 1
                    pixel_distances.append(0.0)
                else:
                    # Distance to nearest mask pixel (slow but N=500, tolerable)
                    # Check local region within tolerance
                    r = int(np.ceil(args.tolerance))
                    lo_y = max(0, py_int - r); hi_y = min(H, py_int + r + 1)
                    lo_x = max(0, px_int - r); hi_x = min(W, px_int + r + 1)
                    local = tgt_mask[lo_y:hi_y, lo_x:hi_x]
                    if local.sum() > 0:
                        ys, xs = np.where(local > 0.5)
                        dists = np.sqrt((ys - (py_int - lo_y))**2 + (xs - (px_int - lo_x))**2)
                        min_d = float(dists.min())
                        pixel_distances.append(min_d)
                        if min_d <= args.tolerance:
                            hits_near_mask += 1
                    else:
                        pixel_distances.append(np.inf)

            rate_exact = hits_in_mask / max(hits_in_view, 1)
            rate_tol = hits_near_mask / max(hits_in_view, 1)
            pixel_distances_finite = [d for d in pixel_distances if np.isfinite(d)]
            median_dist = float(np.median(pixel_distances_finite)) if pixel_distances_finite else float("inf")
            log(f"    → {tgt_cam}: in_view={hits_in_view}/{n_sample}, "
                f"exact={100*rate_exact:.1f}%, tol({args.tolerance}px)={100*rate_tol:.1f}%, "
                f"median_dist={median_dist:.2f}px")
            per_target[tgt_cam] = {
                "in_view": hits_in_view,
                "hits_exact": hits_in_mask,
                "hits_within_tolerance": hits_near_mask,
                "rate_exact": rate_exact,
                "rate_within_tolerance": rate_tol,
                "median_distance_px": median_dist,
            }
            pair_stats[f"{src_cam}→{tgt_cam}"] = rate_tol

        all_results[src_cam] = per_target

    # Overall pass rate (off-diagonal pairs, excluding self)
    log("")
    log("=" * 60)
    log("MVC Summary (off-diagonal pairs, src → tgt where src ≠ tgt)")
    log("=" * 60)
    off_diag_rates = []
    for pair, rate in pair_stats.items():
        src, tgt = pair.split("→")
        if src != tgt:
            off_diag_rates.append(rate)
            log(f"  {pair}: {100*rate:.1f}% within {args.tolerance}px")

    mean_rate = np.mean(off_diag_rates) if off_diag_rates else 0.0
    min_rate = np.min(off_diag_rates) if off_diag_rates else 0.0
    log("")
    log(f"  Mean off-diag pair pass rate: {100*mean_rate:.1f}%")
    log(f"  Min  off-diag pair pass rate: {100*min_rate:.1f}%")

    # Verdict
    log("")
    log("=" * 60)
    log("VERDICT (pre-specified thresholds)")
    log("=" * 60)
    if mean_rate > 0.95:
        verdict = "PASS: preprocessing MVG-consistent (>95% threshold)"
    elif mean_rate > 0.80:
        verdict = "CAUTION: 80-95% aligned (OpenAI threshold met, Gemini threshold not)"
    elif mean_rate > 0.50:
        verdict = "FAIL-partial: 50-80% — targeted cam fix needed"
    else:
        verdict = "FAIL-global: <50% — markerless_v8 needed"
    log(f"  {verdict}")

    result = {
        "per_src": all_results,
        "pair_stats": pair_stats,
        "off_diag_mean_rate": float(mean_rate),
        "off_diag_min_rate": float(min_rate),
        "tolerance_px": args.tolerance,
        "n_samples": args.n_samples,
        "verdict": verdict,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log("")
    log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
