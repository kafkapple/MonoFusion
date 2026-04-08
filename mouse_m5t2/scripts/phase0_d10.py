# no-split: D10 — FG Gaussian spatial distribution analysis
"""
D10 — FG Gaussian spatial distribution in canonical space

Answers: "Where are the 14,613 FG Gaussians actually located?"

Measurements (no interpretation, no hypothesis):
  M1: fg.params.means 3D bbox (min/max/range per axis)
  M2: Gaussian centers vs GT mouse mask overlap (project to cam view)
  M3: Scale anisotropy ratio (max_scale / min_scale per Gaussian)

Pre-specified falsification targets written before running.

Usage:
  PYTHONPATH=. python mouse_m5t2/scripts/phase0_d10.py \
    --checkpoint /node_data/joon/data/monofusion/markerless_v7/results_v10b/checkpoints/best.ckpt \
    --data_root /node_data/joon/data/monofusion/markerless_v7 \
    --output /node_data/joon/data/monofusion/markerless_v7/phase0_d10.json \
    --viz_dir /node_data/joon/data/monofusion/markerless_v7/phase0_d10_viz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[d10] {msg}", flush=True)


def load_gt_mask(data_root, cam_name, frame_idx):
    """Load mask .npz, return binary HxW float32 [0, 1]."""
    mask_dir = Path(data_root) / "masks" / cam_name
    paths = sorted(mask_dir.glob("*.npz"))
    if not paths:
        # fallback: try different layout
        mask_dir = Path(data_root) / cam_name / "masks"
        paths = sorted(mask_dir.glob("*.npz"))
    raw = np.load(paths[frame_idx])["dyn_mask"]
    return (raw > 0).astype(np.float32)


def load_Ks_w2cs(data_root, cam_names, n_frames=80):
    """Load cameras from Dy_train_meta_cam0N.json files. Returns [N_cam*T, 3, 3], [N_cam*T, 4, 4]."""
    import json
    Ks_list = []
    w2cs_list = []
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])  # "m5t2_cam00" → 0
        meta_path = meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json"
        with open(meta_path) as f:
            meta = json.load(f)
        K = np.array(meta["k"], dtype=np.float32).squeeze()    # [80, 3, 3] or [3, 3]
        w2c = np.array(meta["w2c"], dtype=np.float32).squeeze() # [80, 4, 4] or [4, 4]
        conv = meta.get("camera_convention", "w2c")
        if conv == "c2w":
            if w2c.ndim == 2:
                w2c = np.linalg.inv(w2c)
            else:
                w2c = np.array([np.linalg.inv(w) for w in w2c])
        if K.ndim == 2:
            K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2:
            w2c = np.tile(w2c[None], (n_frames, 1, 1))
        print(f"    [load_Ks_w2cs] {cam_name}: K{K.shape}, w2c{w2c.shape}, conv={conv}")
        Ks_list.append(K)
        w2cs_list.append(w2c)
    Ks = np.concatenate(Ks_list, axis=0)
    w2cs = np.concatenate(w2cs_list, axis=0)
    return Ks, w2cs


def project_points(points_world, K, w2c):
    """Project N world-space 3D points to 2D pixel coords. Returns [N, 2] + [N] depth."""
    # points: [N, 3] world
    N = points_world.shape[0]
    points_h = np.concatenate([points_world, np.ones((N, 1))], axis=1)  # [N, 4]
    points_cam = (w2c @ points_h.T).T[:, :3]  # [N, 3]
    depth = points_cam[:, 2]
    pixels = (K @ points_cam.T).T  # [N, 3]
    pixels_2d = pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-6)
    return pixels_2d, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--viz_dir", required=True)
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("D10: FG Gaussian spatial distribution")
    log("=" * 60)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    fg_means = sd["fg.params.means"].float().numpy()       # [N, 3]
    fg_scales_log = sd["fg.params.scales"].float().numpy()  # [N, 3]
    fg_scales = np.exp(fg_scales_log)                       # [N, 3]
    fg_opacities = torch.sigmoid(sd["fg.params.opacities"].float()).numpy()  # [N]

    # Also load BG for context
    bg_means = sd["bg.params.means"].float().numpy()  # [M, 3]
    bg_scales = np.exp(sd["bg.params.scales"].float().numpy())

    N = fg_means.shape[0]
    log(f"  FG Gaussians loaded: {N:,}")
    log(f"  BG Gaussians loaded: {bg_means.shape[0]:,}")

    # --- M1: FG means 3D bounding box ---
    log("")
    log("M1: FG means 3D bounding box (world coords)")
    fg_min = fg_means.min(axis=0)
    fg_max = fg_means.max(axis=0)
    fg_range = fg_max - fg_min
    fg_centroid = fg_means.mean(axis=0)
    fg_std = fg_means.std(axis=0)
    log(f"  min       = {fg_min.tolist()}")
    log(f"  max       = {fg_max.tolist()}")
    log(f"  range     = {fg_range.tolist()}")
    log(f"  centroid  = {fg_centroid.tolist()}")
    log(f"  std       = {fg_std.tolist()}")

    bg_min = bg_means.min(axis=0)
    bg_max = bg_means.max(axis=0)
    bg_range = bg_max - bg_min
    log(f"  (for scale context) BG range = {bg_range.tolist()}")
    log(f"  FG volume / BG volume ratio  = {np.prod(fg_range) / np.prod(bg_range):.4f}")

    M1 = {
        "fg_min_xyz": fg_min.tolist(),
        "fg_max_xyz": fg_max.tolist(),
        "fg_range_xyz": fg_range.tolist(),
        "fg_centroid_xyz": fg_centroid.tolist(),
        "fg_std_xyz": fg_std.tolist(),
        "bg_range_xyz": bg_range.tolist(),
        "fg_over_bg_volume_ratio": float(np.prod(fg_range) / np.prod(bg_range)),
    }

    # --- M3: Scale anisotropy ---
    log("")
    log("M3: FG Gaussian scale anisotropy (per-Gaussian max/min scale ratio)")
    scale_max_per_g = fg_scales.max(axis=1)  # [N]
    scale_min_per_g = fg_scales.min(axis=1)
    aniso = scale_max_per_g / np.maximum(scale_min_per_g, 1e-9)
    aniso_pcts = np.percentile(aniso, [5, 25, 50, 75, 95, 99])
    log(f"  anisotropy percentiles [5,25,50,75,95,99]: {aniso_pcts.tolist()}")
    log(f"  > 5× elongated: {int((aniso > 5).sum())}/{N}")
    log(f"  > 10× elongated: {int((aniso > 10).sum())}/{N}")
    # Also report absolute scales
    scale_mean_per_g = fg_scales.mean(axis=1)
    log(f"  scale mean (m) percentiles [5,50,95]: "
        f"{np.percentile(scale_mean_per_g, [5, 50, 95]).tolist()}")

    M3 = {
        "anisotropy_pcts_5_25_50_75_95_99": aniso_pcts.tolist(),
        "elongated_over_5x": int((aniso > 5).sum()),
        "elongated_over_10x": int((aniso > 10).sum()),
        "scale_mean_pcts_5_50_95": np.percentile(scale_mean_per_g, [5, 50, 95]).tolist(),
    }

    # --- M2: Gaussian centers vs GT mouse mask overlap (per cam, frame 0) ---
    log("")
    log("M2: FG Gaussian projection into cam views at frame 0 — mask overlap")
    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]

    # Load cameras from checkpoint-adjacent paths (try multiple common locations)
    Ks = None
    w2cs = None
    try:
        Ks, w2cs = load_Ks_w2cs(args.data_root, cam_names, n_frames=80)
        log(f"  Loaded cam intrinsics/extrinsics from files: Ks={Ks.shape}, w2cs={w2cs.shape}")
    except Exception as e:
        log(f"  [WARN] Could not load K/w2c from files: {e}")
        log(f"  Trying to get from checkpoint buffers...")

    if Ks is None:
        # Try to load via dataset
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from train_m5t2 import patch_casual_dataset
            patch_casual_dataset()
            from flow3d.data.casual_dataset import CasualDataset, CustomDataConfig
            Ks_list, w2cs_list = [], []
            for cam in cam_names:
                cfg = CustomDataConfig(
                    seq_name=cam, root_dir=args.data_root,
                    video_name=cam, depth_type="moge",
                )
                ds = CasualDataset(data_cfg=cfg, training=False, num_targets_per_frame=1)
                Ks_list.append(ds.Ks.numpy())
                w2cs_list.append(ds.w2cs.numpy())
            Ks = np.concatenate(Ks_list, axis=0)
            w2cs = np.concatenate(w2cs_list, axis=0)
            log(f"  Loaded via CasualDataset: Ks={Ks.shape}, w2cs={w2cs.shape}")
        except Exception as e:
            log(f"  [ERR] Could not load cameras: {e}")
            import traceback
            traceback.print_exc()
            Ks = None

    M2 = {}
    if Ks is not None:
        W, H = args.img_wh
        per_cam = []
        for cam_idx, cam_name in enumerate(cam_names):
            t = 0  # frame 0
            idx = cam_idx * 80 + t
            K = Ks[idx]
            w2c = w2cs[idx]

            # Project FG means to cam view
            pixels_2d, depth = project_points(fg_means, K, w2c)
            in_view = (
                (pixels_2d[:, 0] >= 0) & (pixels_2d[:, 0] < W) &
                (pixels_2d[:, 1] >= 0) & (pixels_2d[:, 1] < H) &
                (depth > 0)
            )
            n_in_view = int(in_view.sum())

            # Load GT mask at t=0
            try:
                mask = load_gt_mask(args.data_root, cam_name, t)
                # For each in-view Gaussian, check if its pixel is inside the mask
                px = pixels_2d[in_view].astype(int)
                px[:, 0] = np.clip(px[:, 0], 0, W - 1)
                px[:, 1] = np.clip(px[:, 1], 0, H - 1)
                mask_vals = mask[px[:, 1], px[:, 0]]
                n_in_mask = int((mask_vals > 0.5).sum())
                mask_fg_frac = float(mask.mean())
                log(f"  {cam_name} @ t=0:")
                log(f"    FG Gaussian in-view: {n_in_view}/{N}")
                log(f"    FG Gaussian inside GT mask: {n_in_mask}/{n_in_view} "
                    f"({100 * n_in_mask / max(n_in_view, 1):.1f}%)")
                log(f"    GT mask FG fraction: {mask_fg_frac * 100:.2f}%")
                per_cam.append({
                    "cam": cam_name,
                    "n_in_view": n_in_view,
                    "n_in_mask": n_in_mask,
                    "mask_hit_rate": float(n_in_mask / max(n_in_view, 1)),
                    "gt_mask_fg_frac": mask_fg_frac,
                })

                # Save scatter overlay viz
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100, facecolor="black")
                    ax.set_facecolor("black")
                    ax.imshow(mask, cmap="Greens", alpha=0.5, origin="upper")
                    in_view_px = pixels_2d[in_view]
                    ax.scatter(in_view_px[:, 0], in_view_px[:, 1], s=0.5, c="red", alpha=0.6)
                    ax.set_xlim(0, W); ax.set_ylim(H, 0)
                    ax.set_title(f"{cam_name} t=0 — FG Gaussians (red) vs GT mask (green)",
                                 color="white")
                    ax.axis("off")
                    fig.savefig(viz_dir / f"m2_{cam_name}_t00.png",
                                facecolor="black", bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    log(f"  [WARN] viz failed: {e}")

            except Exception as e:
                log(f"  [WARN] mask load failed for {cam_name}: {e}")
                per_cam.append({"cam": cam_name, "error": str(e)})

        M2["per_cam"] = per_cam
        if per_cam:
            valid = [c for c in per_cam if "mask_hit_rate" in c]
            if valid:
                mean_hit = np.mean([c["mask_hit_rate"] for c in valid])
                log("")
                log(f"  📊 Mean FG-in-mask hit rate across cams: {mean_hit * 100:.1f}%")
                M2["mean_hit_rate"] = float(mean_hit)

    # --- 3D scatter viz: top/side/front orthographic ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100, facecolor="black")
        for ax in axes: ax.set_facecolor("black")
        # Top: XZ
        axes[0].scatter(fg_means[:, 0], fg_means[:, 2], s=0.3, c="red", alpha=0.5, label="FG")
        axes[0].scatter(bg_means[::50, 0], bg_means[::50, 2], s=0.1, c="gray", alpha=0.2, label="BG (subsampled)")
        axes[0].set_xlabel("X", color="white"); axes[0].set_ylabel("Z", color="white")
        axes[0].set_title("Top view (XZ)", color="white")
        axes[0].legend(loc="upper right", facecolor="black", labelcolor="white")
        # Side: YZ
        axes[1].scatter(fg_means[:, 1], fg_means[:, 2], s=0.3, c="red", alpha=0.5)
        axes[1].scatter(bg_means[::50, 1], bg_means[::50, 2], s=0.1, c="gray", alpha=0.2)
        axes[1].set_xlabel("Y", color="white"); axes[1].set_ylabel("Z", color="white")
        axes[1].set_title("Side view (YZ)", color="white")
        # Front: XY
        axes[2].scatter(fg_means[:, 0], fg_means[:, 1], s=0.3, c="red", alpha=0.5)
        axes[2].scatter(bg_means[::50, 0], bg_means[::50, 1], s=0.1, c="gray", alpha=0.2)
        axes[2].set_xlabel("X", color="white"); axes[2].set_ylabel("Y", color="white")
        axes[2].set_title("Front view (XY)", color="white")
        for ax in axes:
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("white")
        fig.suptitle(f"D10: FG Gaussian spatial distribution ({N:,} FG vs BG context)",
                     color="white")
        fig.savefig(viz_dir / "m1_3d_scatter_orthographic.png",
                    facecolor="black", bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved 3D scatter viz to {viz_dir}/m1_3d_scatter_orthographic.png")
    except Exception as e:
        log(f"  [WARN] 3D scatter viz failed: {e}")

    # --- Save results ---
    results = {
        "M1_fg_bbox": M1,
        "M2_mask_overlap": M2,
        "M3_scale_anisotropy": M3,
        "counts": {"fg": N, "bg": int(bg_means.shape[0])},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("")
    log(f"Wrote {out}")
    log(f"Viz dir: {viz_dir}")


if __name__ == "__main__":
    main()
