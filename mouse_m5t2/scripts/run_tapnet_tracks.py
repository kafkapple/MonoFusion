"""
Generate 2D tracks using CoTracker for M5t2 MonoFusion data.

CoTracker is easier to install than TAPNet (pure PyTorch, no JAX).
Output format matches MonoFusion TAPIR track format: [N, 4] = [x, y, occ, dist].

Usage:
    CUDA_VISIBLE_DEVICES=4 python run_tapnet_tracks.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --n_points 512
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_video_frames(cam_dir: Path) -> torch.Tensor:
    """Load all frames as tensor (T, 3, H, W) float32 [0,1]."""
    frames = sorted(cam_dir.glob("*.png"))
    if not frames:
        frames = sorted(cam_dir.glob("*.jpg"))
    imgs = []
    for fp in frames:
        img = np.array(Image.open(str(fp)).convert("RGB"))
        imgs.append(torch.from_numpy(img).float() / 255.0)
    # (T, H, W, 3) -> (T, 3, H, W)
    video = torch.stack(imgs).permute(0, 3, 1, 2)
    return video, [fp.stem for fp in frames]


def sample_fg_query_points(mask_dir: Path, frame_name: str, n_points: int) -> np.ndarray:
    """Sample query points from foreground mask at given frame."""
    mask_path = mask_dir / f"{frame_name}.npz"
    if mask_path.exists():
        with np.load(mask_path) as data:
            if "dyn_mask" in data:
                mask = data["dyn_mask"]
            else:
                mask = data[data.files[0]]
    else:
        # Try png
        mask_path = mask_dir / f"{frame_name}.png"
        mask = np.array(Image.open(str(mask_path)))
        mask = (mask > 128).astype(np.float32)

    # FG pixels where mask == 1
    fg_pixels = np.argwhere(mask > 0.5)  # (N, 2) as [row, col] = [y, x]
    if len(fg_pixels) == 0:
        print(f"  WARNING: no foreground pixels in {frame_name}")
        return np.zeros((n_points, 2))

    if len(fg_pixels) < n_points:
        # Repeat if not enough points
        idx = np.random.choice(len(fg_pixels), n_points, replace=True)
    else:
        idx = np.random.choice(len(fg_pixels), n_points, replace=False)

    points_yx = fg_pixels[idx]
    # Return as [x, y] (MonoFusion convention)
    return points_yx[:, ::-1].copy().astype(np.float32)  # [N, 2] as [x, y]


def generate_tracks_with_cotracker(
    video: torch.Tensor,
    query_points_xy: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use CoTracker to track points through video.
    Returns tracks [N, T, 2] and visibility [N, T].
    """
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(device).eval()

    T_total, C, H, W = video.shape
    N = len(query_points_xy)

    # CoTracker expects video: (1, T, 3, H, W)
    video_input = video.unsqueeze(0).to(device)

    # Queries format for CoTracker: (1, N, 3) = [t, x, y]
    queries = torch.zeros(1, N, 3, device=device)
    queries[0, :, 0] = 0  # query at frame 0
    queries[0, :, 1] = torch.from_numpy(query_points_xy[:, 0]).float()  # x
    queries[0, :, 2] = torch.from_numpy(query_points_xy[:, 1]).float()  # y

    with torch.no_grad():
        # CoTracker3 online mode processes in windows
        is_first_step = True
        for i in range(0, T_total, model.step):
            if is_first_step:
                pred_tracks, pred_visibility = model(
                    video_chunk=video_input[:, :model.step],
                    is_first_step=True,
                    queries=queries,
                )
                is_first_step = False
            else:
                end = min(i + model.step, T_total)
                pred_tracks, pred_visibility = model(
                    video_chunk=video_input[:, i:end],
                    is_first_step=False,
                )

    # pred_tracks: (1, T, N, 2) in pixel coords
    # pred_visibility: (1, T, N) boolean
    tracks = pred_tracks[0].permute(1, 0, 2).cpu().numpy()  # (N, T, 2)
    vis = pred_visibility[0].permute(1, 0).cpu().numpy().astype(np.float32)  # (N, T)

    return tracks, vis


def generate_tracks_with_flow(
    video: torch.Tensor,
    query_points_xy: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback: simple optical flow based tracking using RAFT.
    Less accurate than CoTracker but works without extra deps.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    T_total, C, H, W = video.shape
    N = len(query_points_xy)

    tracks = np.zeros((N, T_total, 2), dtype=np.float32)
    vis = np.ones((N, T_total), dtype=np.float32)

    # Initialize with query points
    tracks[:, 0, :] = query_points_xy

    current_pts = query_points_xy.copy()

    for t in tqdm(range(1, T_total), desc="  flow tracking"):
        frame1 = video[t - 1].unsqueeze(0).to(device)
        frame2 = video[t].unsqueeze(0).to(device)

        # Normalize for RAFT
        frame1 = frame1 * 255.0
        frame2 = frame2 * 255.0

        with torch.no_grad():
            flow = model(frame1, frame2)[-1]  # (1, 2, H, W)

        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Sample flow at current point locations
        for i in range(N):
            x, y = int(round(current_pts[i, 0])), int(round(current_pts[i, 1]))
            if 0 <= x < W and 0 <= y < H:
                dx, dy = flow_np[y, x]
                current_pts[i, 0] += dx
                current_pts[i, 1] += dy
                # Check bounds
                if not (0 <= current_pts[i, 0] < W and 0 <= current_pts[i, 1] < H):
                    vis[i, t] = 0.0
            else:
                vis[i, t] = 0.0

        tracks[:, t, :] = current_pts.copy()

        torch.cuda.empty_cache()

    return tracks, vis


def save_tracks_monofusion_format(
    tracks: np.ndarray,
    vis: np.ndarray,
    frame_names: list[str],
    output_dir: Path,
    query_frame_idx: int = 0,
):
    """
    Save tracks in MonoFusion TAPIR format.
    MonoFusion expects: {query_frame}_{target_frame}.npy with shape [N, 4]
    where 4 = [x, y, occlusion_logit, expected_dist_logit]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    N, T, _ = tracks.shape
    query_name = frame_names[query_frame_idx]

    for t in range(T):
        target_name = frame_names[t]
        # Convert visibility to TAPIR format:
        # occlusion_logit: negative = visible, positive = occluded
        # expected_dist_logit: negative = confident
        occ_logit = np.where(vis[:, t] > 0.5, -5.0, 5.0)  # visible -> -5 (sigmoid->~0)
        dist_logit = np.where(vis[:, t] > 0.5, -5.0, 5.0)

        track_data = np.zeros((N, 4), dtype=np.float32)
        track_data[:, 0] = tracks[:, t, 0]  # x
        track_data[:, 1] = tracks[:, t, 1]  # y
        track_data[:, 2] = occ_logit
        track_data[:, 3] = dist_logit

        out_path = output_dir / f"{query_name}_{target_name}.npy"
        np.save(str(out_path), track_data)

    print(f"  Saved {T} track files to {output_dir}")


def main(data_root: str, n_points: int, use_flow_fallback: bool):
    data_root = Path(data_root)
    device = "cuda"

    img_root = data_root / "images"
    mask_root = data_root / "masks"
    cam_dirs = sorted([d for d in img_root.iterdir() if d.is_dir() and list(d.glob("*.png"))])

    for cam_dir in cam_dirs:
        seq_name = cam_dir.name
        mask_dir = mask_root / seq_name
        track_dir = data_root / "tapir" / seq_name

        print(f"\n=== {seq_name} ===")

        # Load video
        video, frame_names = load_video_frames(cam_dir)
        T, C, H, W = video.shape
        print(f"  Video: {T} frames, {H}x{W}")

        # Sample query points from FG mask at frame 0
        query_pts = sample_fg_query_points(mask_dir, frame_names[0], n_points)
        print(f"  Query points: {query_pts.shape}")

        # Track points
        if use_flow_fallback:
            tracks, vis = generate_tracks_with_flow(video, query_pts, device)
        else:
            try:
                tracks, vis = generate_tracks_with_cotracker(video, query_pts, device)
            except Exception as e:
                print(f"  CoTracker failed ({e}), falling back to RAFT flow")
                tracks, vis = generate_tracks_with_flow(video, query_pts, device)

        print(f"  Tracks: {tracks.shape}, visibility: {vis.mean():.2%} visible")

        # Save in MonoFusion format
        save_tracks_monofusion_format(tracks, vis, frame_names, track_dir)

        torch.cuda.empty_cache()

    print("\nTrack generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--n_points", type=int, default=512)
    parser.add_argument("--use_flow_fallback", action="store_true")
    args = parser.parse_args()
    main(args.data_root, args.n_points, args.use_flow_fallback)
