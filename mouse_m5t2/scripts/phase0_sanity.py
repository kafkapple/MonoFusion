# no-split: Sanity 1-4 — cheap checks before preprocessing re-validation
"""
Sanity 1-4 — distinguishes projection-code-bug vs FG-degenerate-collapse vs preprocessing-bug

Sanity 1: Random points in BG bbox → project 4 cams → expect hit rate ≈ mask FG%
  Fail → projection code path bug (all D10/D11b invalid)

Sanity 2: BG Gaussian means → project 4 cams → expect uniform low hit across cams
  Asymmetric → cam-specific coordinate frame issue

Sanity 3: FG "in-view" Gaussian 2D pixel histogram for cam0/1/3 (the 0% cams)
  Clustered in one region → FG degenerate collapse (Devil's hypothesis)

Sanity 4: Dump per-cam intrinsic/extrinsic values from markerless_v7 metadata
  Sanity-check that cam2 depth outlier matches camera geometry
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[sanity] {msg}", flush=True)


def load_Ks_w2cs(data_root, cam_names, n_frames):
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    Ks_list, w2cs_list, meta_list = [], [], []
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])
        with open(meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json") as f:
            meta = json.load(f)
        K = np.array(meta["k"], dtype=np.float32).squeeze()
        w2c = np.array(meta["w2c"], dtype=np.float32).squeeze()
        meta_list.append({
            "cam": cam_name,
            "convention": meta.get("camera_convention"),
            "K_at_t0": K[0].tolist() if K.ndim == 3 else K.tolist(),
            "w2c_at_t0": (w2c[0] if w2c.ndim == 3 else w2c).tolist(),
        })
        if K.ndim == 2: K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2: w2c = np.tile(w2c[None], (n_frames, 1, 1))
        Ks_list.append(K)
        w2cs_list.append(w2c)
    return (
        np.concatenate(Ks_list, 0),
        np.concatenate(w2cs_list, 0),
        meta_list,
    )


def project(points, K, w2c, img_wh):
    W, H = img_wh
    N = points.shape[0]
    pts_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    proj = (K @ pts_cam.T).T
    pixels = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    in_view = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < W) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < H) &
        (depth > 0)
    )
    return pixels, depth, in_view


def load_mask(data_root, cam_name, frame_idx):
    paths = sorted((Path(data_root) / "masks" / cam_name).glob("*.npz"))
    return (np.load(paths[frame_idx])["dyn_mask"] > 0).astype(np.float32)


def hit_rate(pixels, in_view, mask, img_wh):
    W, H = img_wh
    px = pixels[in_view].astype(int)
    if px.shape[0] == 0:
        return 0.0, 0, 0
    px[:, 0] = np.clip(px[:, 0], 0, W - 1)
    px[:, 1] = np.clip(px[:, 1], 0, H - 1)
    hits = int((mask[px[:, 1], px[:, 0]] > 0.5).sum())
    return hits / px.shape[0], hits, px.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--viz_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir); viz_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    fg_means = sd["fg.params.means"].float().numpy()
    bg_means = sd["bg.params.means"].float().numpy()

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    Ks, w2cs, meta_list = load_Ks_w2cs(args.data_root, cam_names, 80)

    results = {}
    rng = np.random.RandomState(42)

    # ======================= Sanity 1 =======================
    log("=" * 60)
    log("Sanity 1: Random points in BG bbox → projection code check")
    log("=" * 60)
    bg_min = bg_means.min(0); bg_max = bg_means.max(0)
    n_rand = 10000
    random_points = rng.uniform(bg_min, bg_max, size=(n_rand, 3)).astype(np.float32)
    log(f"  random bbox: min={bg_min.tolist()}, max={bg_max.tolist()}, n={n_rand}")
    s1 = []
    for cam_idx, cam_name in enumerate(cam_names):
        K = Ks[cam_idx * 80 + 0]; w2c = w2cs[cam_idx * 80 + 0]
        pixels, depth, in_view = project(random_points, K, w2c, args.img_wh)
        mask = load_mask(args.data_root, cam_name, 0)
        rate, hits, nv = hit_rate(pixels, in_view, mask, args.img_wh)
        mask_frac = float(mask.mean())
        log(f"  {cam_name}: in_view={int(in_view.sum())}/{n_rand}, hits={hits}/{nv} ({100*rate:.2f}%), mask_frac={100*mask_frac:.2f}%")
        expected_ratio = rate / mask_frac if mask_frac > 0 else float("nan")
        log(f"    ratio hit/mask_frac = {expected_ratio:.3f} (expect ~1.0 if projection code is sound)")
        s1.append({"cam": cam_name, "hit_rate": rate, "mask_frac": mask_frac,
                   "ratio": expected_ratio, "in_view": int(in_view.sum()), "hits": hits})
    results["sanity1_random"] = s1
    s1_pass = all(0.3 < r["ratio"] < 3.0 for r in s1 if not np.isnan(r["ratio"]))
    log(f"  VERDICT: Sanity 1 {'PASS (projection code sound)' if s1_pass else 'FAIL (code path bug)'}")
    results["sanity1_verdict"] = "pass" if s1_pass else "fail"

    # ======================= Sanity 2 =======================
    log("")
    log("=" * 60)
    log("Sanity 2: BG Gaussian means → project 4 cams, expect uniform low hit")
    log("=" * 60)
    # Subsample BG for speed
    bg_sub = bg_means[::20]  # ~14K BG points
    log(f"  BG subsample: {bg_sub.shape[0]} points")
    s2 = []
    for cam_idx, cam_name in enumerate(cam_names):
        K = Ks[cam_idx * 80 + 0]; w2c = w2cs[cam_idx * 80 + 0]
        pixels, depth, in_view = project(bg_sub, K, w2c, args.img_wh)
        mask = load_mask(args.data_root, cam_name, 0)
        rate, hits, nv = hit_rate(pixels, in_view, mask, args.img_wh)
        mask_frac = float(mask.mean())
        log(f"  {cam_name}: in_view={int(in_view.sum())}/{len(bg_sub)}, hits={hits}/{nv} ({100*rate:.2f}%), mask_frac={100*mask_frac:.2f}%")
        s2.append({"cam": cam_name, "bg_hit_rate": rate, "mask_frac": mask_frac,
                   "bg_in_view": int(in_view.sum()), "bg_hits": hits})
    results["sanity2_bg"] = s2
    bg_rates = [r["bg_hit_rate"] for r in s2]
    uniform = max(bg_rates) - min(bg_rates) < 0.1
    log(f"  Max-min BG hit rate across cams: {100*(max(bg_rates) - min(bg_rates)):.2f}%")
    log(f"  VERDICT: Sanity 2 {'UNIFORM (no cam-specific frame bug)' if uniform else 'ASYMMETRIC (cam bug suspected)'}")
    results["sanity2_verdict"] = "uniform" if uniform else "asymmetric"

    # ======================= Sanity 3 =======================
    log("")
    log("=" * 60)
    log("Sanity 3: FG Gaussian pixel histogram per cam at t=0")
    log("=" * 60)
    s3 = []
    for cam_idx, cam_name in enumerate(cam_names):
        K = Ks[cam_idx * 80 + 0]; w2c = w2cs[cam_idx * 80 + 0]
        pixels, depth, in_view = project(fg_means, K, w2c, args.img_wh)
        in_view_px = pixels[in_view]
        if in_view_px.shape[0] == 0:
            log(f"  {cam_name}: no in-view Gaussians")
            continue
        px_mean = in_view_px.mean(axis=0)
        px_std = in_view_px.std(axis=0)
        px_min = in_view_px.min(axis=0)
        px_max = in_view_px.max(axis=0)
        W, H = args.img_wh
        # Also find cluster centers via simple modal analysis
        log(f"  {cam_name}: n_in_view={in_view_px.shape[0]}")
        log(f"    pixel mean={px_mean.tolist()}, std={px_std.tolist()}")
        log(f"    pixel bbox=[{px_min.tolist()}, {px_max.tolist()}]")
        log(f"    spread as % of image: x={100*px_std[0]/W:.1f}%, y={100*px_std[1]/H:.1f}%")
        s3.append({
            "cam": cam_name, "n_in_view": int(in_view_px.shape[0]),
            "pixel_mean": px_mean.tolist(), "pixel_std": px_std.tolist(),
            "pixel_min": px_min.tolist(), "pixel_max": px_max.tolist(),
            "spread_pct_x": float(100 * px_std[0] / W),
            "spread_pct_y": float(100 * px_std[1] / H),
        })

        # Visualize histogram
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100, facecolor="black")
            ax.set_facecolor("black")
            mask = load_mask(args.data_root, cam_name, 0)
            ax.imshow(mask, cmap="Greens", alpha=0.5, origin="upper")
            ax.hist2d(in_view_px[:, 0], in_view_px[:, 1], bins=50, cmap="Reds", alpha=0.7)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_title(f"Sanity 3: {cam_name} — FG pixel density + mask", color="white")
            ax.axis("off")
            fig.savefig(viz_dir / f"sanity3_{cam_name}.png", facecolor="black", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log(f"    viz err: {e}")
    results["sanity3_fg_hist"] = s3

    # ======================= Sanity 4 =======================
    log("")
    log("=" * 60)
    log("Sanity 4: Per-cam intrinsic/extrinsic dump")
    log("=" * 60)
    for m in meta_list:
        log(f"  {m['cam']}:")
        log(f"    convention: {m['convention']}")
        K = np.array(m["K_at_t0"])
        w2c = np.array(m["w2c_at_t0"])
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cam_center_world = -w2c[:3, :3].T @ w2c[:3, 3]
        cam_z_world = w2c[:3, :3].T @ np.array([0, 0, 1])
        dist_to_origin = np.linalg.norm(cam_center_world)
        log(f"    K: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        log(f"    camera center (world): {cam_center_world.tolist()}")
        log(f"    camera forward (z axis): {cam_z_world.tolist()}")
        log(f"    |cam_center| = {dist_to_origin:.2f}")
    results["sanity4_cams"] = meta_list

    # Save
    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
