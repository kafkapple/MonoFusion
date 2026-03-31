"""
Trajectory on Video: Project 3D Gaussian trajectories onto actual video frames.

Shows whether learned SE(3) motion bases match the real mouse movement.
Each Gaussian's 3D path is projected to 2D via camera intrinsics/extrinsics,
then drawn as colored trails on the original RGB frames.

Usage:
    cd /node_data/joon/data/monofusion/m5t2_poc
    PYTHONPATH=~/dev/MonoFusion:~/dev/MonoFusion/preproc/Dust3R \
    CUDA_VISIBLE_DEVICES=2 python ~/dev/MonoFusion/mouse_m5t2/scripts/viz_trajectory_on_video.py \
        --checkpoint results_50ep/checkpoints/epoch_0049.ckpt \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/viz \
        --device cuda
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


def load_model(ckpt_path, device):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SceneModel.init_from_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_gt_images(data_root, cam_name):
    from PIL import Image
    img_dir = data_root / "images" / cam_name
    return [np.array(Image.open(p).convert("RGB")) for p in sorted(img_dir.glob("*.png"))]


@torch.no_grad()
def extract_all_trajectories(model, device, n_gaussians=200):
    """Extract 2D projected trajectories for top-N moving Gaussians per camera."""
    T = model.num_frames
    Ks = model.Ks.cpu().numpy()   # (T_or_C, 3, 3) — per frame or per cam
    w2cs = model.w2cs.cpu().numpy()  # (T_or_C, 4, 4)

    # Get 3D positions at all frames
    all_means = []
    for t in range(T):
        ts = torch.tensor([t], device=device)
        means, _ = model.compute_poses_fg(ts)
        all_means.append(means[:, 0].cpu().numpy())  # (G, 3)
    all_means = np.stack(all_means)  # (T, G, 3)
    G = all_means.shape[1]

    # Select top-N by total path length
    path_lengths = np.sum(np.linalg.norm(np.diff(all_means, axis=0), axis=-1), axis=0)
    top_idx = np.argsort(path_lengths)[-n_gaussians:]

    return all_means, top_idx, Ks, w2cs


def project_to_2d(pts_3d, K, w2c):
    """Project 3D points to 2D using camera params. Returns (N, 2) pixel coords."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (R @ pts_3d.T + t[:, None]).T  # (N, 3)
    z = pts_cam[:, 2]
    valid = z > 0.01
    u = K[0, 0] * pts_cam[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / (z + 1e-8) + K[1, 2]
    return np.stack([u, v], axis=1), valid


def draw_trajectories_on_frame(
    frame: np.ndarray,
    all_means: np.ndarray,  # (T, G, 3)
    top_idx: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    current_t: int,
    trail_length: int = 10,
):
    """Draw trajectory trails on a single frame.

    Each trail shows the past `trail_length` frames of each Gaussian's 2D position.
    Color: time-coded (blue=oldest, red=current).
    """
    img = frame.copy()
    T = all_means.shape[0]
    t_start = max(0, current_t - trail_length)

    cmap = cm.hot  # blue→red

    for gi in top_idx:
        pts_trail = []
        for t in range(t_start, current_t + 1):
            uv, valid = project_to_2d(all_means[t, gi:gi+1], K, w2c)
            if valid[0]:
                u, v = int(uv[0, 0]), int(uv[0, 1])
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    pts_trail.append((u, v, t))

        # Draw trail as connected line segments
        for i in range(1, len(pts_trail)):
            alpha = (pts_trail[i][2] - t_start) / max(1, trail_length)
            color = tuple(int(c * 255) for c in cmap(alpha)[:3])
            cv2.line(img, pts_trail[i-1][:2], pts_trail[i][:2], color, 1, cv2.LINE_AA)

        # Draw current position as dot
        if pts_trail:
            cv2.circle(img, pts_trail[-1][:2], 2, (0, 255, 0), -1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Trajectory on Video")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--cam_name", type=str, default="m5t2_cam00")
    parser.add_argument("--n_gaussians", type=int, default=150)
    parser.add_argument("--trail_length", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    gt_imgs = load_gt_images(args.data_root, args.cam_name)
    T = len(gt_imgs)

    print(f"\nExtracting trajectories ({args.n_gaussians} Gaussians)...")
    all_means, top_idx, Ks, w2cs = extract_all_trajectories(
        model, args.device, args.n_gaussians
    )

    # Fused layout: cam c, frame t → index = c * n_frames + t
    # Use frame 0 for camera params (static camera)
    cam_base_idx = args.cam_idx * T
    K = Ks[cam_base_idx]
    w2c = w2cs[cam_base_idx]

    # Generate video frames
    print(f"Drawing trajectories on {args.cam_name} ({T} frames)...")
    import imageio.v3 as iio
    frames = []
    for t in range(T):
        img = draw_trajectories_on_frame(
            gt_imgs[t], all_means, top_idx, K, w2c, t, args.trail_length
        )
        frames.append(img)
        if t % 15 == 0:
            print(f"  frame {t}/{T}")

    # Save video
    out_video = args.output_dir / f"R6_trajectory_on_video_{args.cam_name}.mp4"
    iio.imwrite(str(out_video), frames, fps=15)
    print(f"  Saved: {out_video}")

    # Save key stills
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, t in enumerate([0, 15, 30, 45]):
        axes[i].imshow(frames[t])
        axes[i].set_title(f"F{t}", fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f"Learned Trajectories on {args.cam_name} (top {args.n_gaussians} Gaussians)", fontsize=12)
    plt.tight_layout()
    out_still = args.output_dir / f"R6_trajectory_stills_{args.cam_name}.png"
    plt.savefig(out_still, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_still}")


if __name__ == "__main__":
    main()
