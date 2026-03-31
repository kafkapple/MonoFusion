"""
Option A: Mask-based post-filtering of existing TAPNet tracks.

Problem: TAPNet marks mouse FG tracks as occluded; only background tracks
remain visible (11-19% visible ratio, centroid error 100-170px).

Solution: For each frame, mark tracks as occluded if they fall outside
the FG mask. This forces MonoFusion to only use FG-located tracks.

Two output modes:
  --mode save_filtered: write new filtered .npy files (preserves format)
  --mode viz_only:      visualization comparison only (default)

Usage:
    python filter_tracks_by_mask.py \\
        --data_root /node_data/joon/data/monofusion/m5t2_poc \\
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/tapir_filtered \\
        --mode save_filtered
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from viz_utils import get_cam_dirs, get_frame_names


def load_mask(mask_root: Path, cam_name: str, frame_name: str) -> np.ndarray | None:
    """Load binary FG mask from npz file. Returns H×W bool array or None."""
    mask_path = mask_root / cam_name / f"{frame_name}.npz"
    if not mask_path.exists():
        return None
    data = np.load(mask_path)
    raw = data["dyn_mask"]
    # dyn_mask is float32 in [-1, 1]: +1 = foreground, -1 = background.
    # Must threshold at >0, not .astype(bool) (which treats -1.0 as True).
    mask = raw > 0
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    return mask


def filter_tracks_single_frame(
    track: np.ndarray,
    mask: np.ndarray,
    dilation_px: int = 5,
) -> np.ndarray:
    """Apply mask filter to a single frame's track array.

    Args:
        track: [N, 4] array of (x, y, occlusion_logit, expected_dist)
        mask: [H, W] bool array (True = foreground)
        dilation_px: dilate mask by this many pixels to handle border noise

    Returns:
        [N, 4] array with occlusion_logit set to +10.0 for out-of-mask points.
    """
    if dilation_px > 0:
        struct = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=bool)
        mask = binary_dilation(mask, structure=struct)

    H, W = mask.shape
    xs = track[:, 0].clip(0, W - 1).astype(int)
    ys = track[:, 1].clip(0, H - 1).astype(int)
    in_mask = mask[ys, xs]

    filtered = track.copy()
    # Set occlusion logit to +10 (sigmoid ≈ 1.0 → very occluded) for BG points
    filtered[~in_mask, 2] = 10.0
    return filtered


def compute_stats(tracks_raw: list, tracks_filtered: list, thresh: float = 0.0) -> dict:
    """Compare visible point counts before/after filtering."""
    stats = {"before": [], "after": []}
    for raw, filt in zip(tracks_raw, tracks_filtered):
        if raw is None:
            continue
        vis_before = (raw[:, 2] <= thresh).sum()
        vis_after = (filt[:, 2] <= thresh).sum()
        stats["before"].append(vis_before)
        stats["after"].append(vis_after)
    return stats


def process_camera(
    cam_dir: Path,
    track_dir: Path,
    mask_root: Path,
    output_dir: Path | None,
    dilation_px: int,
    query_frame_idx: int = 0,
) -> dict:
    """Process one camera: load tracks, apply mask filter, optionally save.

    Returns stats dict for visualization.
    """
    frames = get_frame_names(cam_dir)
    if not frames:
        return {}

    query_name = frames[query_frame_idx]
    cam_name = cam_dir.name

    tracks_raw = []
    tracks_filtered = []

    for fname in frames:
        tp = track_dir / f"{query_name}_{fname}.npy"
        if not tp.exists():
            tracks_raw.append(None)
            tracks_filtered.append(None)
            continue

        raw = np.load(tp)
        tracks_raw.append(raw)

        mask = load_mask(mask_root, cam_name, fname)
        if mask is None:
            tracks_filtered.append(raw.copy())
        else:
            tracks_filtered.append(filter_tracks_single_frame(raw, mask, dilation_px))

    if output_dir is not None:
        cam_out = output_dir / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)
        for fname, filt in zip(frames, tracks_filtered):
            if filt is not None:
                np.save(cam_out / f"{query_name}_{fname}.npy", filt)
        print(f"  Saved filtered tracks: {cam_out}")

    stats = compute_stats(tracks_raw, tracks_filtered)
    stats["cam_name"] = cam_name
    stats["frames"] = frames
    stats["tracks_raw"] = tracks_raw
    stats["tracks_filtered"] = tracks_filtered
    return stats


def viz_comparison(
    cam_dir: Path,
    stats: dict,
    viz_dir: Path,
    n_show: int = 200,
    occlusion_thresh: float = 0.0,
):
    """Generate before/after 3-frame comparison for one camera."""
    frames = stats["frames"]
    tracks_raw = stats["tracks_raw"]
    tracks_filtered = stats["tracks_filtered"]
    cam_name = stats["cam_name"]

    T = len(frames)
    frame_indices = [0, T // 4, T // 2, T - 1]
    n_cols = len(frame_indices)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    fig.suptitle(f"Track Filter Comparison — {cam_name}\n"
                 f"Top row: BEFORE (original TAPNet)  |  Bottom row: AFTER (mask-filtered)",
                 fontsize=13, y=0.98)

    for col, fi in enumerate(frame_indices):
        fname = frames[fi]
        img = np.array(Image.open(cam_dir / f"{fname}.png"))

        raw = tracks_raw[fi]
        filt = tracks_filtered[fi]

        for row, (track, label) in enumerate([(raw, "BEFORE"), (filt, "AFTER")]):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")

            if track is None:
                ax.set_title(f"Frame {fi}\n[no track]")
                continue

            N = track.shape[0]
            # Show all points (subsample if too many)
            if N > n_show:
                idx = np.random.choice(N, n_show, replace=False)
            else:
                idx = np.arange(N)

            vis_mask = track[idx, 2] <= occlusion_thresh
            n_vis = vis_mask.sum()

            ax.scatter(track[idx][vis_mask, 0], track[idx][vis_mask, 1],
                       c="lime", s=6, alpha=0.8, linewidths=0)
            ax.scatter(track[idx][~vis_mask, 0], track[idx][~vis_mask, 1],
                       c="red", s=2, alpha=0.2, linewidths=0)

            vis_ratio = n_vis / len(idx) * 100
            ax.set_title(f"F{fi} {label}\n{n_vis}/{len(idx)} vis ({vis_ratio:.0f}%)",
                         fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = viz_dir / f"08_track_filter_{cam_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Viz saved: {out_path}")


def viz_visibility_curve(all_stats: list, viz_dir: Path):
    """Plot visible track count over time for all cameras, before vs after."""
    fig, axes = plt.subplots(1, len(all_stats), figsize=(5 * len(all_stats), 4),
                             sharey=False)
    if len(all_stats) == 1:
        axes = [axes]

    for ax, stats in zip(axes, all_stats):
        before = stats["before"]
        after = stats["after"]
        ax.plot(before, label="Before (TAPNet)", color="red", alpha=0.8)
        ax.plot(after, label="After (mask filter)", color="green", alpha=0.8)
        ax.set_title(stats["cam_name"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Visible track count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Visible Track Count Over Time — Before vs After Mask Filter", fontsize=13)
    fig.tight_layout()
    out_path = viz_dir / "08_visibility_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Visibility curve: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Mask-based TAPNet track filtering (Option A)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to MonoFusion data root (contains tapir/, masks/, images/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save filtered tracks. None = viz_only mode")
    parser.add_argument("--viz_dir", type=str, default=None,
                        help="Where to save comparison viz. Default: data_root/viz")
    parser.add_argument("--mode", choices=["save_filtered", "viz_only"],
                        default="viz_only",
                        help="save_filtered: write new npy files; viz_only: only visualize")
    parser.add_argument("--dilation_px", type=int, default=5,
                        help="Mask dilation in pixels (handles border noise, default=5)")
    parser.add_argument("--n_show", type=int, default=200,
                        help="Max tracks to show in visualization")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Camera names to process (default: all)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    viz_dir = Path(args.viz_dir) if args.viz_dir else data_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    output_dir = None
    if args.mode == "save_filtered" and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    track_root = data_root / "tapir"
    mask_root = data_root / "masks"
    cam_dirs = get_cam_dirs(data_root)

    if args.cameras:
        cam_dirs = [d for d in cam_dirs if d.name in args.cameras]

    print(f"Processing {len(cam_dirs)} cameras, dilation={args.dilation_px}px")
    all_stats = []

    for cam_dir in cam_dirs:
        track_dir = track_root / cam_dir.name
        if not track_dir.exists():
            print(f"  Skipping {cam_dir.name}: no track dir")
            continue

        print(f"  Camera: {cam_dir.name}")
        stats = process_camera(
            cam_dir, track_dir, mask_root, output_dir, args.dilation_px
        )
        if not stats:
            continue

        all_stats.append(stats)
        viz_comparison(cam_dir, stats, viz_dir, n_show=args.n_show)

        before_avg = np.mean(stats["before"]) if stats["before"] else 0
        after_avg = np.mean(stats["after"]) if stats["after"] else 0
        if stats["tracks_raw"] and stats["tracks_raw"][0] is not None:
            N_total = stats["tracks_raw"][0].shape[0]
        else:
            N_total = 0
        print(f"    Visible avg: {before_avg:.0f}/{N_total} → {after_avg:.0f}/{N_total} "
              f"({before_avg/N_total*100:.1f}% → {after_avg/N_total*100:.1f}%)")

    if all_stats:
        viz_visibility_curve(all_stats, viz_dir)

    print(f"\nDone. Visualizations in: {viz_dir}")
    if output_dir:
        print(f"Filtered tracks in: {output_dir}")


if __name__ == "__main__":
    main()
