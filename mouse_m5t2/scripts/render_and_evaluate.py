"""
Post-training batch rendering, evaluation, and 4D Gaussian visualization.

Generates:
  1. GT vs Rendered comparison (4 cameras, selected frames)
  2. Rendered video (all 60 frames, per camera)
  3. 4D Gaussian point cloud (colored by motion magnitude)
  4. Scene flow arrows on rendered images
  5. PSNR/SSIM metrics

Usage (on gpu03):
    cd /node_data/joon/data/monofusion/m5t2_poc
    PYTHONPATH=~/dev/MonoFusion:~/dev/MonoFusion/preproc/Dust3R \
    python ~/dev/MonoFusion/mouse_m5t2/scripts/render_and_evaluate.py \
        --checkpoint results_50ep/checkpoints/epoch_0049.ckpt \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/viz \
        --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_model(ckpt_path: Path, device: str):
    """Load SceneModel from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]
    model = SceneModel.init_from_state_dict(state_dict)
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", -1)
    print(f"  Loaded model: {model.num_fg_gaussians} FG, epoch={epoch}")
    return model, epoch


def load_gt_images(data_root: Path, cam_name: str) -> list[np.ndarray]:
    """Load ground truth RGB images for a camera."""
    from PIL import Image
    img_dir = data_root / "images" / cam_name
    imgs = []
    for p in sorted(img_dir.glob("*.png")):
        imgs.append(np.array(Image.open(p).convert("RGB")))
    return imgs


@torch.no_grad()
def render_frame(model, t: int, cam_idx: int, device: str, img_wh=(512, 512), n_frames=60):
    """Render a single frame from a specific camera.

    model.w2cs layout: [cam0_f0, ..., cam0_f59, cam1_f0, ..., cam3_f59] = [240, 4, 4]
    Index for camera c, frame t: idx = c * n_frames + t
    """
    idx = cam_idx * n_frames + t
    w2c = model.w2cs[idx:idx+1].to(device)
    K = model.Ks[idx:idx+1].to(device)
    W, H = img_wh
    out = model.render(
        t=t, w2cs=w2c, Ks=K, img_wh=img_wh,
        return_color=True, return_feat=False,
        return_depth=True, return_mask=True,
    )
    rgb = out["img"][0].clamp(0, 1).cpu().numpy()  # (H, W, 3)
    depth = out.get("depth", torch.zeros(1, H, W, 1))[0, :, :, 0].cpu().numpy()
    mask = out.get("mask", torch.zeros(1, H, W, 1))[0, :, :, 0].cpu().numpy()
    return rgb, depth, mask


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute PSNR and SSIM between GT and predicted images."""
    gt_f = gt.astype(np.float64) / 255.0
    pred_f = pred.astype(np.float64)
    if pred_f.max() > 1.0:
        pred_f = pred_f / 255.0

    mse = np.mean((gt_f - pred_f) ** 2)
    psnr = -10 * np.log10(mse + 1e-10)

    # Simple SSIM (per channel, then average)
    from scipy.ndimage import uniform_filter
    ssim_vals = []
    for c in range(3):
        mu_x = uniform_filter(gt_f[:,:,c], size=11)
        mu_y = uniform_filter(pred_f[:,:,c], size=11)
        sig_x2 = uniform_filter(gt_f[:,:,c]**2, size=11) - mu_x**2
        sig_y2 = uniform_filter(pred_f[:,:,c]**2, size=11) - mu_y**2
        sig_xy = uniform_filter(gt_f[:,:,c]*pred_f[:,:,c], size=11) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y+C1)*(2*sig_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sig_x2+sig_y2+C2))
        ssim_vals.append(ssim_map.mean())
    ssim = np.mean(ssim_vals)
    return {"psnr": psnr, "ssim": ssim}


# ── R1: GT vs Rendered comparison ──

def viz_gt_vs_rendered(model, data_root, output_dir, device, cam_names, frames=[0, 15, 30, 45]):
    """Side-by-side GT vs rendered comparison."""
    print("\n[R1] GT vs Rendered comparison...")
    n_cams = len(cam_names)
    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, n_cams * 2, figsize=(4*n_cams*2, 4*n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    all_metrics = []
    for fi, t in enumerate(frames):
        for ci, cam_name in enumerate(cam_names):
            gt_imgs = load_gt_images(data_root, cam_name)
            gt = gt_imgs[t] if t < len(gt_imgs) else gt_imgs[-1]

            rgb, _, _ = render_frame(model, t, ci, device)
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            m = compute_metrics(gt, rgb)
            all_metrics.append({"cam": cam_name, "frame": t, **m})

            axes[fi, ci*2].imshow(gt)
            axes[fi, ci*2].set_title(f"GT {cam_name} F{t}", fontsize=8)
            axes[fi, ci*2].axis("off")

            axes[fi, ci*2+1].imshow(rgb)
            axes[fi, ci*2+1].set_title(f"Pred PSNR={m['psnr']:.1f}", fontsize=8)
            axes[fi, ci*2+1].axis("off")

    fig.suptitle("GT vs Rendered", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "R1_gt_vs_rendered.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Print metrics table
    print("\n  Metrics:")
    print(f"  {'Camera':<12} {'Frame':<6} {'PSNR':>6} {'SSIM':>6}")
    for m in all_metrics:
        print(f"  {m['cam']:<12} F{m['frame']:<5} {m['psnr']:>6.2f} {m['ssim']:>6.4f}")
    avg_psnr = np.mean([m["psnr"] for m in all_metrics])
    avg_ssim = np.mean([m["ssim"] for m in all_metrics])
    print(f"  {'Average':<12} {'':6} {avg_psnr:>6.2f} {avg_ssim:>6.4f}")
    return all_metrics


# ── R4: 4D Gaussian point cloud ──

def viz_4d_gaussians(model, output_dir, device, frames=[0, 15, 30, 45, 59]):
    """Visualize 4D Gaussians colored by motion magnitude."""
    print("\n[R4] 4D Gaussian point cloud (motion-colored)...")
    from flow3d.transforms import cont_6d_to_rmat

    n_frames = len(frames)
    fig = plt.figure(figsize=(5*n_frames, 5))
    fig.suptitle("4D Gaussians: Position Over Time (color=motion magnitude)", fontsize=12)

    # Extract means at each frame
    all_means = []
    with torch.no_grad():
        for t in frames:
            ts = torch.tensor([t], device=device)
            means, _ = model.compute_poses_fg(ts)
            all_means.append(means[:, 0].cpu().numpy())  # (G, 3)

    # Compute motion magnitude (total displacement from F0)
    means_f0 = all_means[0]
    for i, (t, means_t) in enumerate(zip(frames, all_means)):
        ax = fig.add_subplot(1, n_frames, i+1, projection="3d")
        mag = np.linalg.norm(means_t - means_f0, axis=1)

        # Subsample for clarity
        max_pts = 3000
        if len(means_t) > max_pts:
            idx = np.random.choice(len(means_t), max_pts, replace=False)
        else:
            idx = np.arange(len(means_t))

        sc = ax.scatter(means_t[idx, 0], means_t[idx, 1], means_t[idx, 2],
                        c=mag[idx], cmap="plasma", s=1, alpha=0.6)
        ax.set_title(f"F{t}", fontsize=10)
        ax.view_init(elev=30, azim=45)
        ax.tick_params(labelsize=5)

    plt.colorbar(sc, ax=fig.axes, label="|displacement from F0|", shrink=0.6)
    plt.tight_layout()
    out_path = output_dir / "R4_4d_gaussians.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── R5: Scene flow on rendered images ──

def viz_sceneflow_on_image(model, data_root, output_dir, device, cam_idx=0, t=15):
    """Overlay 3D scene flow arrows on rendered image."""
    print(f"\n[R5] Scene flow on rendered image (cam{cam_idx}, F{t})...")
    rgb, depth, _ = render_frame(model, t, cam_idx, device)

    # Get 3D scene flow
    with torch.no_grad():
        ts0 = torch.tensor([t], device=device)
        ts1 = torch.tensor([t+1], device=device)
        means_t, _ = model.compute_poses_fg(ts0)
        means_t1, _ = model.compute_poses_fg(ts1)
        flow_3d = (means_t1 - means_t)[:, 0].cpu().numpy()  # (G, 3)
        pos_3d = means_t[:, 0].cpu().numpy()  # (G, 3)

    # Project to 2D (fused layout: cam c, frame t → idx = c*60+t)
    cam_base = cam_idx * 60
    K = model.Ks[cam_base].cpu().numpy()
    w2c = model.w2cs[cam_base].cpu().numpy()
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Transform to camera space
    pts_cam = (w2c[:3, :3] @ pos_3d.T + w2c[:3, 3:4]).T  # (G, 3)
    z = pts_cam[:, 2]
    valid = z > 0.1
    u = fx * pts_cam[valid, 0] / z[valid] + cx
    v = fy * pts_cam[valid, 1] / z[valid] + cy

    # Flow in camera space
    flow_cam = (w2c[:3, :3] @ flow_3d[valid].T).T
    du = fx * flow_cam[:, 0] / z[valid]
    dv = fy * flow_cam[:, 1] / z[valid]

    # In-frame filter
    W, H = 512, 512
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, du, dv = u[in_frame], v[in_frame], du[in_frame], dv[in_frame]
    flow_mag = np.sqrt(du**2 + dv**2)

    # Subsample top-K by magnitude
    top_k = min(200, len(u))
    top_idx = np.argsort(flow_mag)[-top_k:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Scene Flow on Rendered Image — cam{cam_idx} F{t}→F{t+1}", fontsize=12)

    axes[0].imshow(rgb)
    axes[0].quiver(u[top_idx], v[top_idx], du[top_idx], -dv[top_idx],
                   flow_mag[top_idx], cmap="hot", scale=50, width=0.003, alpha=0.8)
    axes[0].set_title(f"3D Scene Flow → 2D ({top_k} arrows)")
    axes[0].axis("off")

    # GT image for comparison
    cam_name = f"m5t2_cam{cam_idx:02d}"
    gt_imgs = load_gt_images(data_root, cam_name)
    axes[1].imshow(gt_imgs[t])
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"R5_sceneflow_on_image_cam{cam_idx}_f{t}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-training rendering + evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading model from {args.checkpoint}")
    model, epoch = load_model(args.checkpoint, args.device)

    # R1: GT vs Rendered
    metrics = viz_gt_vs_rendered(model, args.data_root, args.output_dir,
                                args.device, args.cameras, frames=[0, 15, 30, 45])

    # R4: 4D Gaussian point cloud
    viz_4d_gaussians(model, args.output_dir, args.device)

    # R5: Scene flow on image
    viz_sceneflow_on_image(model, args.data_root, args.output_dir, args.device, cam_idx=0, t=15)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
