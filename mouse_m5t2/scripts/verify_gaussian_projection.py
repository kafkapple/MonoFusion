"""Verify FG Gaussian projection matches GT mouse position.

Loads a trained checkpoint, projects FG Gaussian centroids onto GT images,
and measures pixel offset from GT mask centroid.

Success criteria: projection error < 20 pixels (was 117-2384px with convention bug).
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image

MONOFUSION_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(MONOFUSION_ROOT))


def compute_mask_centroid(mask_path):
    """Get centroid of FG mask (value=1 region)."""
    data = np.load(mask_path)
    mask = data['dyn_mask']
    fg = (mask == 1.0)
    if fg.sum() == 0:
        return None
    ys, xs = np.where(fg)
    return np.array([xs.mean(), ys.mean()])


def project_gaussians(means_3d, w2c, K):
    """Project 3D Gaussian means to 2D pixel coords.

    means_3d: (N, 3) world coordinates
    w2c: (4, 4) world-to-camera
    K: (3, 3) intrinsics
    Returns: (N, 2) pixel coords
    """
    N = means_3d.shape[0]
    ones = torch.ones(N, 1, device=means_3d.device)
    pts_h = torch.cat([means_3d, ones], dim=1)  # (N, 4)
    pts_cam = (w2c @ pts_h.T).T  # (N, 4)
    pts_cam = pts_cam[:, :3]  # (N, 3)
    z = pts_cam[:, 2:3].clamp(min=1e-5)
    pts_2d = (K @ pts_cam.T).T  # (N, 3)
    px = pts_2d[:, :2] / pts_2d[:, 2:3].clamp(min=1e-5)
    return px, pts_cam[:, 2]  # (N, 2), (N,) depths


def main(ckpt_path, data_root):
    ckpt_path = Path(ckpt_path)
    data_root = Path(data_root)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract FG Gaussian means and camera params
    model = ckpt['model']
    fg_means = model['fg.params.means']  # (N, 3)
    w2cs = model['w2cs']  # (C*T, 4, 4)
    Ks = model['Ks']      # (C*T, 3, 3)

    print(f"FG Gaussians: {fg_means.shape[0]}")
    print(f"w2cs: {w2cs.shape}, Ks: {Ks.shape}")

    # For M5t2: w2cs layout is [cam0_f0..f79, cam1_f0..f79, ...]
    n_frames = 80
    n_cams = w2cs.shape[0] // n_frames
    print(f"Detected: {n_cams} cameras, {n_frames} frames")

    # Test frame 0 for each camera
    results = []
    for cam_idx in range(n_cams):
        frame_idx = 0
        global_idx = cam_idx * n_frames + frame_idx

        w2c = w2cs[global_idx]  # (4, 4)
        K = Ks[global_idx]      # (3, 3)

        # Project all FG Gaussians
        px_coords, depths = project_gaussians(fg_means, w2c, K)

        # Filter: only Gaussians in front of camera with reasonable depth
        valid = (depths > 0.1) & (depths < 50.0)
        valid_px = px_coords[valid]

        if len(valid_px) == 0:
            print(f"  cam{cam_idx}: NO valid projections!")
            continue

        # Gaussian centroid (mean of all projected FG Gaussians)
        gauss_centroid = valid_px.mean(dim=0).numpy()

        # In-frame Gaussians
        in_frame = (valid_px[:, 0] >= 0) & (valid_px[:, 0] < 512) & \
                   (valid_px[:, 1] >= 0) & (valid_px[:, 1] < 512)
        pct_in_frame = in_frame.float().mean().item() * 100

        # GT mask centroid
        mask_path = data_root / "masks" / f"m5t2_cam{cam_idx:02d}" / f"{frame_idx:06d}.npz"
        gt_centroid = compute_mask_centroid(str(mask_path))

        if gt_centroid is not None:
            error = np.linalg.norm(gauss_centroid - gt_centroid)
            results.append(error)
            print(f"  cam{cam_idx}: Gauss=({gauss_centroid[0]:.1f}, {gauss_centroid[1]:.1f})"
                  f"  GT=({gt_centroid[0]:.1f}, {gt_centroid[1]:.1f})"
                  f"  Error={error:.1f}px  InFrame={pct_in_frame:.0f}%")
        else:
            print(f"  cam{cam_idx}: Gauss=({gauss_centroid[0]:.1f}, {gauss_centroid[1]:.1f})"
                  f"  GT=NO MASK  InFrame={pct_in_frame:.0f}%")

    if results:
        avg_err = np.mean(results)
        max_err = np.max(results)
        print(f"\n=== VERDICT ===")
        print(f"Average projection error: {avg_err:.1f} px")
        print(f"Max projection error: {max_err:.1f} px")
        print(f"Threshold: < 20 px")
        if max_err < 20:
            print("PASS: Camera convention fix verified")
        elif max_err < 50:
            print("MARGINAL: Improved but may need tuning")
        else:
            print("FAIL: Projection error still too large")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "/node_data/joon/data/monofusion/m5t2_v5/results_v5j/checkpoints/best.ckpt"
    data = sys.argv[2] if len(sys.argv) > 2 else \
        "/node_data/joon/data/monofusion/m5t2_v5"
    main(ckpt, data)
