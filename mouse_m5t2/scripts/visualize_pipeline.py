"""
Multi-view visualization and camera consistency check for M5t2 MonoFusion data.

Generates 7 diagnostic plots: multi-view RGB, mask overlay, depth, DINOv2 PCA,
tracks, camera poses 3D, reprojection consistency.

Usage:
    python visualize_pipeline.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --frame_idx 0 10 30 59
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from viz_utils import (
    load_meta, get_cam_dirs, get_frame_names,
    viz_multiview_rgb, viz_mask_overlay, viz_depth_overlay, viz_dinov2_pca,
)


def viz_tracks(data_root: Path, output_dir: Path, n_show: int = 50,
               occlusion_thresh: float = 0.0):
    """Show tracked points across frames for each camera.

    Track .npy format: [N, 4] = (x, y, occlusion_logit, expected_dist).
    Points with occlusion_logit > occlusion_thresh are considered occluded.
    """
    cam_dirs = get_cam_dirs(data_root)
    track_root = data_root / "tapir"

    for cam_dir in cam_dirs:
        frames = get_frame_names(cam_dir)
        track_dir = track_root / cam_dir.name
        if not track_dir.exists():
            continue
        query_name = frames[0]
        T = len(frames)

        # Load full track data (coords + occlusion + expected_dist)
        all_raw = []
        for fname in frames:
            tp = track_dir / f"{query_name}_{fname}.npy"
            all_raw.append(np.load(tp) if tp.exists() else None)
        if not any(r is not None for r in all_raw):
            continue

        # Filter by visibility: occlusion_logit <= threshold = visible
        N = all_raw[0].shape[0] if all_raw[0] is not None else 0
        # Select points visible across most frames
        vis_count = np.zeros(N)
        for r in all_raw:
            if r is not None and r.shape[0] == N:
                vis_count += (r[:, 2] <= occlusion_thresh).astype(float)
        # Pick top-n most visible points
        top_idx = np.argsort(-vis_count)[:n_show]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img0 = np.array(Image.open(cam_dir / f"{frames[0]}.png"))
        axes[0].imshow(img0)
        if all_raw[0] is not None:
            pts = all_raw[0][top_idx]
            visible = pts[:, 2] <= occlusion_thresh
            axes[0].scatter(pts[visible, 0], pts[visible, 1], c="lime", s=8, alpha=0.8)
        axes[0].set_title(f"Frame 0 - Query ({visible.sum()}/{n_show} vis)")
        axes[0].axis("off")

        mid = T // 2
        img_mid = np.array(Image.open(cam_dir / f"{frames[mid]}.png"))
        axes[1].imshow(img_mid)
        if all_raw[mid] is not None and all_raw[mid].shape[0] == N:
            pts = all_raw[mid][top_idx]
            visible = pts[:, 2] <= occlusion_thresh
            axes[1].scatter(pts[visible, 0], pts[visible, 1], c="cyan", s=8, alpha=0.8)
            axes[1].scatter(pts[~visible, 0], pts[~visible, 1], c="red", s=3, alpha=0.3)
        axes[1].set_title(f"Frame {mid} - Tracked (red=occluded)")
        axes[1].axis("off")

        img_last = np.array(Image.open(cam_dir / f"{frames[-1]}.png"))
        axes[2].imshow(img_last)
        for pi_idx in range(len(top_idx)):
            pi = top_idx[pi_idx]
            traj_x, traj_y = [], []
            for t in range(T):
                if all_raw[t] is not None and pi < all_raw[t].shape[0]:
                    if all_raw[t][pi, 2] <= occlusion_thresh:
                        traj_x.append(all_raw[t][pi, 0])
                        traj_y.append(all_raw[t][pi, 1])
                    else:
                        # Break trajectory at occlusion
                        if traj_x:
                            axes[2].plot(traj_x, traj_y, linewidth=0.7, alpha=0.6)
                        traj_x, traj_y = [], []
            if traj_x:
                axes[2].plot(traj_x, traj_y, linewidth=0.7, alpha=0.6)
        axes[2].set_title(f"Frame {T-1} - Trajectories (visible only)")
        axes[2].axis("off")

        fig.suptitle(f"Tracks - {cam_dir.name} (N={N}, shown={n_show})", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"05_tracks_{cam_dir.name}.png", dpi=150)
        plt.close(fig)
    print(f"  [5] Tracks: {len(cam_dirs)} cameras saved")


def viz_camera_poses(data_root: Path, output_dir: Path):
    """3D visualization of camera positions and orientations."""
    meta = load_meta(data_root)
    n_cams = len(meta["hw"])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, n_cams))

    all_pos = []
    for ci in range(n_cams):
        w2c = np.array(meta["w2c"][0][ci])
        c2w = np.linalg.inv(w2c)
        pos = c2w[:3, 3]
        forward = c2w[:3, 2] * 0.3
        all_pos.append(pos)
        ax.scatter(*pos, c=[colors[ci]], s=100, marker="o", label=f"cam{ci}")
        ax.quiver(*pos, *forward, color=colors[ci], arrow_length_ratio=0.2, linewidth=2)

    center = np.mean(all_pos, axis=0)
    ax.scatter(*center, c="red", s=200, marker="x", label="Scene center")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(f"Camera Poses ({n_cams} cameras)")
    fig.tight_layout()
    fig.savefig(output_dir / "06_camera_poses_3d.png", dpi=150)
    plt.close(fig)
    print(f"  [6] Camera poses: saved")


def viz_reprojection_check(data_root: Path, output_dir: Path):
    """Project scene center into each camera to verify consistency."""
    meta = load_meta(data_root)
    cam_dirs = get_cam_dirs(data_root)
    n_cams = len(meta["hw"])

    positions = [np.linalg.inv(np.array(meta["w2c"][0][ci]))[:3, 3] for ci in range(n_cams)]
    scene_center = np.mean(positions, axis=0)

    fig, axes = plt.subplots(1, n_cams, figsize=(4 * n_cams, 4))
    if n_cams == 1:
        axes = [axes]

    ok_count = 0
    for ci in range(min(n_cams, len(cam_dirs))):
        frames = get_frame_names(cam_dirs[ci])
        img = np.array(Image.open(cam_dirs[ci] / f"{frames[0]}.png"))
        axes[ci].imshow(img)

        K = np.array(meta["k"][0][ci])
        w2c = np.array(meta["w2c"][0][ci])
        pt_cam = w2c[:3, :3] @ scene_center + w2c[:3, 3]

        if pt_cam[2] > 0:
            px = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
            py = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
            h, w = meta["hw"][ci]
            in_frame = 0 <= px <= w and 0 <= py <= h
            color = "lime" if in_frame else "red"
            axes[ci].plot(px, py, "+", markersize=20, markeredgewidth=3, color=color)
            axes[ci].set_title(f"cam{ci}: ({px:.0f},{py:.0f})\n{'OK' if in_frame else 'OUT'}")
            if in_frame:
                ok_count += 1
        else:
            axes[ci].set_title(f"cam{ci}: BEHIND (z={pt_cam[2]:.2f})")
        axes[ci].axis("off")

    fig.suptitle(f"Reprojection Check (center={scene_center.round(3)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "07_reprojection_check.png", dpi=150)
    plt.close(fig)
    print(f"  [7] Reprojection: {ok_count}/{n_cams} cameras OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, nargs="+", default=[0, 10, 30, 59])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MonoFusion M5t2 Pipeline Visualization")
    print("=" * 60)

    viz_multiview_rgb(data_root, output_dir, args.frame_idx)
    viz_mask_overlay(data_root, output_dir, args.frame_idx)
    viz_depth_overlay(data_root, output_dir, args.frame_idx)
    viz_dinov2_pca(data_root, output_dir, args.frame_idx)
    viz_tracks(data_root, output_dir)
    viz_camera_poses(data_root, output_dir)
    viz_reprojection_check(data_root, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
