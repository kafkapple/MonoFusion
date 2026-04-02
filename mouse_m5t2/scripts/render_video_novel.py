"""
R2: Rendered video (GT vs Predicted, 60 frames, per camera)
R3: Novel view synthesis (camera interpolation between existing views)

Usage:
    cd /node_data/joon/data/monofusion/m5t2_poc
    PYTHONPATH=~/dev/MonoFusion:~/dev/MonoFusion/preproc/Dust3R \
    CUDA_VISIBLE_DEVICES=2 python ~/dev/MonoFusion/mouse_m5t2/scripts/render_video_novel.py \
        --checkpoint results_50ep/checkpoints/epoch_0049.ckpt \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/viz \
        --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_model(ckpt_path: Path, device: str):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SceneModel.init_from_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, ckpt.get("epoch", -1)


def load_gt_images(data_root: Path, cam_name: str) -> list[np.ndarray]:
    from PIL import Image
    img_dir = data_root / "images" / cam_name
    return [np.array(Image.open(p).convert("RGB")) for p in sorted(img_dir.glob("*.png"))]


@torch.no_grad()
def render_frame(model, t: int, w2c, K, device, img_wh=(512, 512)):
    """Render a single frame with given camera."""
    w2c_t = w2c.unsqueeze(0).to(device) if w2c.dim() == 2 else w2c.to(device)
    K_t = K.unsqueeze(0).to(device) if K.dim() == 2 else K.to(device)
    out = model.render(
        t=t, w2cs=w2c_t, Ks=K_t, img_wh=img_wh,
        return_color=True, return_feat=False,
        return_depth=True, return_mask=False,
    )
    rgb = out["img"][0].clamp(0, 1).cpu().numpy()
    depth = out.get("depth", torch.zeros(1, 512, 512, 1))[0, :, :, 0].cpu().numpy()
    return rgb, depth


# ── R2: Rendered Video ──

def render_video(model, data_root, output_dir, device, cam_idx=0, cam_name="m5t2_cam00"):
    """Render all 60 frames, create side-by-side GT vs Pred video."""
    import imageio.v3 as iio
    print(f"\n[R2] Rendering video for {cam_name}...")

    gt_imgs = load_gt_images(data_root, cam_name)
    T = len(gt_imgs)
    w2cs = model.w2cs.to(device)
    Ks = model.Ks.to(device)

    n_frames = T
    frames = []
    for t in range(T):
        # Fused layout: cam c, frame t → index = c * n_frames + t
        idx = cam_idx * n_frames + t
        rgb, _ = render_frame(model, t, w2cs[idx], Ks[idx], device)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        gt = gt_imgs[t]

        # Side-by-side
        H, W = gt.shape[:2]
        combined = np.zeros((H, W * 2 + 10, 3), dtype=np.uint8)
        combined[:, :W] = gt
        combined[:, W+10:] = rgb_uint8
        frames.append(combined)
        if t % 15 == 0:
            print(f"    frame {t}/{T}")

    out_path = output_dir / f"R2_video_{cam_name}.mp4"
    import imageio
    writer = imageio.get_writer(str(out_path), format='FFMPEG', fps=15, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved: {out_path}")

    # Also save a still comparison at F0, F30
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, t in enumerate([0, 30]):
        idx = cam_idx * n_frames + t
        rgb, _ = render_frame(model, t, w2cs[idx], Ks[idx], device)
        axes[i, 0].imshow(gt_imgs[t])
        axes[i, 0].set_title(f"GT F{t}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rgb)
        axes[i, 1].set_title(f"Rendered F{t}")
        axes[i, 1].axis("off")
    fig.suptitle(f"GT vs Rendered — {cam_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"R2_still_{cam_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── R3: Novel View Synthesis ──

def slerp_rotation(R0, R1, alpha):
    """Spherical linear interpolation between two rotation matrices."""
    from scipy.spatial.transform import Rotation
    r0 = Rotation.from_matrix(R0)
    r1 = Rotation.from_matrix(R1)
    # Use SLERP via quaternion
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    return slerp(alpha).as_matrix()


def interpolate_cameras(w2c0, w2c1, n_steps=10):
    """Interpolate between two camera poses (w2c matrices)."""
    c2w0 = np.linalg.inv(w2c0)
    c2w1 = np.linalg.inv(w2c1)
    R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
    R1, t1 = c2w1[:3, :3], c2w1[:3, 3]

    cameras = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        R = slerp_rotation(R0, R1, alpha)
        t = (1 - alpha) * t0 + alpha * t1
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        cameras.append(np.linalg.inv(c2w).astype(np.float32))
    return cameras


def render_novel_views(model, output_dir, device, t_frame=15, n_interp=8):
    """Render novel views by interpolating between existing cameras."""
    import imageio.v3 as iio
    print(f"\n[R3] Novel view synthesis (F{t_frame}, {n_interp} interp steps)...")

    w2cs_np = model.w2cs.cpu().numpy()
    Ks_np = model.Ks.cpu().numpy()
    n_total = len(w2cs_np)
    n_cams = 4
    n_frames = n_total // n_cams  # fused layout: cam_c * n_frames + t

    # Interpolation path: cam0→cam1→cam2→cam3→cam0 (closed loop)
    # Use camera poses at t_frame for each camera
    all_novel_w2cs = []
    K_ref = Ks_np[0]  # use cam0 intrinsics for all novel views

    for i in range(n_cams):
        j = (i + 1) % n_cams
        idx_i = i * n_frames + t_frame
        idx_j = j * n_frames + t_frame
        interp = interpolate_cameras(w2cs_np[idx_i], w2cs_np[idx_j], n_steps=n_interp)
        all_novel_w2cs.extend(interp[:-1])  # avoid duplicate at boundary

    print(f"    {len(all_novel_w2cs)} novel viewpoints")

    # Render
    frames = []
    for idx, w2c in enumerate(all_novel_w2cs):
        w2c_t = torch.from_numpy(w2c).to(device)
        K_t = torch.from_numpy(K_ref).to(device)
        rgb, _ = render_frame(model, t_frame, w2c_t, K_t, device)
        frames.append((rgb * 255).astype(np.uint8))

    # Save video
    out_path = output_dir / f"R3_novel_views_f{t_frame}.mp4"
    import imageio
    writer = imageio.get_writer(str(out_path), format='FFMPEG', fps=8, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved: {out_path}")

    # Save grid (4×4 novel views)
    n_show = min(16, len(frames))
    fig, axes = plt.subplots(2, n_show // 2, figsize=(3 * n_show // 2, 6))
    for i in range(n_show):
        ax = axes[i // (n_show // 2), i % (n_show // 2)]
        ax.imshow(frames[i * len(frames) // n_show])
        ax.set_title(f"V{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Novel Views at F{t_frame}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"R3_novel_views_grid_f{t_frame}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: R3_novel_views_grid_f{t_frame}.png")


def main():
    parser = argparse.ArgumentParser(description="R2+R3: Rendered video + Novel views")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--cam_name", type=str, default="m5t2_cam00")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, epoch = load_model(args.checkpoint, args.device)
    print(f"  Epoch: {epoch}")

    # R2: Rendered video
    render_video(model, args.data_root, args.output_dir, args.device,
                 cam_idx=args.cam_idx, cam_name=args.cam_name)

    # R3: Novel views
    render_novel_views(model, args.output_dir, args.device, t_frame=15)
    render_novel_views(model, args.output_dir, args.device, t_frame=0)

    print(f"\nDone. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
