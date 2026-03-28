"""Render synchronized multi-view videos with all preprocessing overlays.

Generates mp4 videos:
- 4-cam RGB side-by-side
- 4-cam depth overlay
- 4-cam point trajectories (from TAPNet, if available)

Usage:
    python render_video.py --data_root /node_data/joon/data/monofusion/m5t2_poc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tqdm import tqdm

from viz_utils import get_cam_dirs, get_frame_names


def render_rgb_video(data_root: Path, output_path: Path, fps: int = 10):
    """4-cam synchronized RGB video."""
    cam_dirs = get_cam_dirs(data_root)
    frames = get_frame_names(cam_dirs[0])
    n_cams = len(cam_dirs)
    T = len(frames)

    # Get image size from first frame
    sample = np.array(Image.open(cam_dirs[0] / f"{frames[0]}.png"))
    H, W = sample.shape[:2]

    import imageio
    writer = imageio.get_writer(str(output_path), fps=fps, quality=8)

    for fi in tqdm(range(T), desc="RGB video"):
        canvas = np.zeros((H, W * n_cams, 3), dtype=np.uint8)
        for ci, cam_dir in enumerate(cam_dirs):
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png"))
            canvas[:, ci * W:(ci + 1) * W] = img[:, :, :3]
        writer.append_data(canvas)
    writer.close()
    print(f"  RGB video: {output_path}")


def render_depth_video(data_root: Path, output_path: Path, fps: int = 10):
    """4-cam depth overlay video."""
    cam_dirs = get_cam_dirs(data_root)
    frames = get_frame_names(cam_dirs[0])
    n_cams = len(cam_dirs)
    T = len(frames)

    sample = np.array(Image.open(cam_dirs[0] / f"{frames[0]}.png"))
    H, W = sample.shape[:2]

    # Collect global depth range from a few samples
    depth_vals = []
    for cam_dir in cam_dirs:
        for fi_sample in [0, T // 2, T - 1]:
            dp = _find_depth(data_root, cam_dir.name, frames[fi_sample])
            if dp is not None:
                d = np.load(dp)
                depth_vals.extend(d[d > 0].tolist()[:1000])
    if depth_vals:
        vmin, vmax = np.percentile(depth_vals, [5, 95])
    else:
        vmin, vmax = 0, 1

    import imageio
    writer = imageio.get_writer(str(output_path), fps=fps, quality=8)
    cmap = cm.get_cmap("viridis")

    for fi in tqdm(range(T), desc="Depth video"):
        canvas = np.zeros((H * 2, W * n_cams, 3), dtype=np.uint8)
        for ci, cam_dir in enumerate(cam_dirs):
            # Top row: RGB
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png"))
            canvas[:H, ci * W:(ci + 1) * W] = img[:, :, :3]

            # Bottom row: depth colormap
            dp = _find_depth(data_root, cam_dir.name, frames[fi])
            if dp is not None:
                depth = np.load(dp)
                depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
                depth_rgb = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                canvas[H:, ci * W:(ci + 1) * W] = depth_rgb
        writer.append_data(canvas)
    writer.close()
    print(f"  Depth video: {output_path}")


def render_tracks_video(data_root: Path, output_path: Path, fps: int = 10,
                        n_tracks: int = 30, trail_len: int = 10,
                        occlusion_thresh: float = 0.0):
    """4-cam point trajectory video with temporal trails."""
    cam_dirs = get_cam_dirs(data_root)
    frames = get_frame_names(cam_dirs[0])
    track_root = data_root / "tapir"
    n_cams = len(cam_dirs)
    T = len(frames)

    sample = np.array(Image.open(cam_dirs[0] / f"{frames[0]}.png"))
    H, W = sample.shape[:2]

    # Pre-load tracks for each camera (query from frame 0)
    cam_tracks = {}  # cam_name -> (T, N, 4) array
    for cam_dir in cam_dirs:
        td = track_root / cam_dir.name
        if not td.exists():
            continue
        query_name = frames[0]
        raw_list = []
        for fname in frames:
            tp = td / f"{query_name}_{fname}.npy"
            if tp.exists():
                raw_list.append(np.load(tp))
            else:
                raw_list.append(None)
        if not any(r is not None for r in raw_list):
            continue

        N = raw_list[0].shape[0] if raw_list[0] is not None else 0
        # Select most-visible tracks
        vis_count = np.zeros(N)
        for r in raw_list:
            if r is not None and r.shape[0] == N:
                vis_count += (r[:, 2] <= occlusion_thresh).astype(float)
        top_idx = np.argsort(-vis_count)[:n_tracks]
        cam_tracks[cam_dir.name] = (raw_list, top_idx, N)

    if not cam_tracks:
        print("  No track data available, skipping track video")
        return

    # Color palette for tracks
    colors = plt.cm.hsv(np.linspace(0, 0.9, n_tracks))[:, :3]

    import imageio
    writer = imageio.get_writer(str(output_path), fps=fps, quality=8)

    for fi in tqdm(range(T), desc="Tracks video"):
        canvas = np.zeros((H, W * n_cams, 3), dtype=np.uint8)
        for ci, cam_dir in enumerate(cam_dirs):
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png"))
            overlay = img[:, :, :3].copy()

            if cam_dir.name in cam_tracks:
                raw_list, top_idx, N = cam_tracks[cam_dir.name]

                for ti, pi in enumerate(top_idx):
                    # Draw trail
                    start = max(0, fi - trail_len)
                    trail_pts = []
                    for t in range(start, fi + 1):
                        if raw_list[t] is not None and pi < raw_list[t].shape[0]:
                            if raw_list[t][pi, 2] <= occlusion_thresh:
                                x = int(np.clip(raw_list[t][pi, 0], 0, W - 1))
                                y = int(np.clip(raw_list[t][pi, 1], 0, H - 1))
                                trail_pts.append((x, y))

                    # Draw trail line
                    color = (colors[ti] * 255).astype(np.uint8)
                    for j in range(1, len(trail_pts)):
                        _draw_line(overlay, trail_pts[j - 1], trail_pts[j], color, 1)

                    # Draw current point
                    if trail_pts:
                        cx, cy = trail_pts[-1]
                        r = 3
                        y1, y2 = max(0, cy - r), min(H, cy + r + 1)
                        x1, x2 = max(0, cx - r), min(W, cx + r + 1)
                        overlay[y1:y2, x1:x2] = color

            canvas[:, ci * W:(ci + 1) * W] = overlay
        writer.append_data(canvas)
    writer.close()
    print(f"  Tracks video: {output_path}")


def _draw_line(img, pt1, pt2, color, thickness=1):
    """Simple Bresenham line drawing without cv2."""
    x0, y0 = pt1
    x1, y1 = pt2
    H, W = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < H and 0 <= x0 < W:
            r = thickness
            y_lo, y_hi = max(0, y0 - r), min(H, y0 + r + 1)
            x_lo, x_hi = max(0, x0 - r), min(W, x0 + r + 1)
            img[y_lo:y_hi, x_lo:x_hi] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _find_depth(data_root: Path, cam_name: str, frame_name: str) -> Path | None:
    """Find depth file trying multiple naming conventions."""
    candidates = [
        data_root / "aligned_moge_depth" / "m5t2" / cam_name / "depth" / f"{frame_name}.npy",
        data_root / "aligned_moge_depth" / "m5t2" / cam_name.replace("_cam", "_undist_cam") / "depth" / f"{frame_name}.npy",
    ]
    # Try 5-digit variant
    try:
        frame_num = int(frame_name)
        candidates.append(data_root / "aligned_moge_depth" / "m5t2" / cam_name.replace("_cam", "_undist_cam") / "depth" / f"{frame_num:05d}.npy")
    except ValueError:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Render MonoFusion pipeline videos")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("MonoFusion Video Rendering")
    print("=" * 50)

    render_rgb_video(data_root, output_dir / "vid_01_rgb_4cam.mp4", fps=args.fps)
    render_depth_video(data_root, output_dir / "vid_02_depth_4cam.mp4", fps=args.fps)
    render_tracks_video(data_root, output_dir / "vid_03_tracks_4cam.mp4", fps=args.fps)

    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
