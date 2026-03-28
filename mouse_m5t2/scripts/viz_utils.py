"""
Visualization utilities for M5t2 MonoFusion pipeline diagnostics.
Each function generates one diagnostic plot.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA


def load_meta(data_root: Path) -> dict:
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def get_cam_dirs(data_root: Path):
    img_root = data_root / "images"
    return sorted([d for d in img_root.iterdir()
                   if d.is_dir() and not d.is_symlink()
                   and "_undist_" not in d.name
                   and list(d.glob("*.png"))])


def get_frame_names(cam_dir: Path):
    # Only 6-digit names (originals), skip 5-digit symlinks
    return sorted([p.stem for p in cam_dir.glob("??????.png")])


def viz_multiview_rgb(data_root: Path, output_dir: Path, frame_indices: list):
    """Grid of all cameras at selected frames."""
    cam_dirs = get_cam_dirs(data_root)
    n_cams = len(cam_dirs)
    for fi in frame_indices:
        fig, axes = plt.subplots(1, n_cams, figsize=(4 * n_cams, 4))
        if n_cams == 1:
            axes = [axes]
        for ci, cam_dir in enumerate(cam_dirs):
            frames = get_frame_names(cam_dir)
            if fi >= len(frames):
                continue
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png"))
            axes[ci].imshow(img)
            axes[ci].set_title(f"{cam_dir.name}\nframe {fi}")
            axes[ci].axis("off")
        fig.suptitle(f"Multi-view RGB - Frame {fi}", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"01_multiview_rgb_f{fi:04d}.png", dpi=150)
        plt.close(fig)
    print(f"  [1] Multi-view RGB: {len(frame_indices)} frames saved")


def viz_mask_overlay(data_root: Path, output_dir: Path, frame_indices: list):
    """RGB with mask overlay (FG=green, border=yellow)."""
    cam_dirs = get_cam_dirs(data_root)
    mask_root = data_root / "masks"
    for fi in frame_indices:
        fig, axes = plt.subplots(1, len(cam_dirs), figsize=(4 * len(cam_dirs), 4))
        if len(cam_dirs) == 1:
            axes = [axes]
        for ci, cam_dir in enumerate(cam_dirs):
            frames = get_frame_names(cam_dir)
            if fi >= len(frames):
                continue
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png")).astype(float) / 255
            mask_path = mask_root / cam_dir.name / f"{frames[fi]}.npz"
            if mask_path.exists():
                with np.load(mask_path) as data:
                    mask = data.get("dyn_mask", data[data.files[0]])
            else:
                mask = np.zeros(img.shape[:2])
            overlay = img.copy()
            overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
            overlay[mask == 0] = overlay[mask == 0] * 0.5 + np.array([1, 1, 0]) * 0.5
            fg_pct = (mask > 0.5).sum() / mask.size * 100
            axes[ci].imshow(overlay)
            axes[ci].set_title(f"{cam_dir.name}\nFG: {fg_pct:.1f}%")
            axes[ci].axis("off")
        fig.suptitle(f"Mask Overlay - Frame {fi}", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"02_mask_overlay_f{fi:04d}.png", dpi=150)
        plt.close(fig)
    print(f"  [2] Mask overlay: {len(frame_indices)} frames saved")


def viz_depth_overlay(data_root: Path, output_dir: Path, frame_indices: list):
    """Depth maps colored with viridis."""
    cam_dirs = get_cam_dirs(data_root)
    for fi in frame_indices:
        fig, axes = plt.subplots(2, len(cam_dirs), figsize=(4 * len(cam_dirs), 8))
        if len(cam_dirs) == 1:
            axes = axes.reshape(-1, 1)
        for ci, cam_dir in enumerate(cam_dirs):
            frames = get_frame_names(cam_dir)
            if fi >= len(frames):
                continue
            img = np.array(Image.open(cam_dir / f"{frames[fi]}.png"))
            axes[0, ci].imshow(img)
            axes[0, ci].set_title(f"{cam_dir.name} RGB")
            axes[0, ci].axis("off")
            # Try both naming conventions (cam vs undist_cam)
            depth_path = (data_root / "aligned_moge_depth" / "m5t2" /
                         cam_dir.name / "depth" / f"{frames[fi]}.npy")
            if not depth_path.exists():
                undist_name = cam_dir.name.replace("_cam", "_undist_cam")
                depth_path = (data_root / "aligned_moge_depth" / "m5t2" /
                             undist_name / "depth" / f"{frames[fi]}.npy")
            if not depth_path.exists():
                # Try 5-digit name
                frame_num = int(frames[fi])
                depth_path = (data_root / "aligned_moge_depth" / "m5t2" /
                             cam_dir.name / "depth" / f"{frame_num:05d}.npy")
            if not depth_path.exists():
                undist_name = cam_dir.name.replace("_cam", "_undist_cam")
                depth_path = (data_root / "aligned_moge_depth" / "m5t2" /
                             undist_name / "depth" / f"{frame_num:05d}.npy")
            if depth_path.exists():
                depth = np.load(depth_path)
                valid = depth[depth > 0]
                vmin, vmax = (np.percentile(valid, [5, 95]) if len(valid) > 0 else (0, 1))
                im = axes[1, ci].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[1, ci].set_title(f"Depth [{vmin:.2f}, {vmax:.2f}]")
                plt.colorbar(im, ax=axes[1, ci], fraction=0.046)
            else:
                axes[1, ci].text(0.5, 0.5, "No depth", ha="center", va="center",
                                transform=axes[1, ci].transAxes)
            axes[1, ci].axis("off")
        fig.suptitle(f"Depth Maps - Frame {fi}", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"03_depth_f{fi:04d}.png", dpi=150)
        plt.close(fig)
    print(f"  [3] Depth maps: {len(frame_indices)} frames saved")


def viz_dinov2_pca(data_root: Path, output_dir: Path, frame_indices: list):
    """PCA of DINOv2 features as RGB."""
    cam_dirs = get_cam_dirs(data_root)
    feat_root = data_root / "dinov2_features"
    for fi in frame_indices[:1]:
        all_feats, feat_shapes = [], []
        for cam_dir in cam_dirs:
            frames = get_frame_names(cam_dir)
            if fi >= len(frames):
                continue
            feat_path = feat_root / cam_dir.name / f"{frames[fi]}.npy"
            if not feat_path.exists():
                frame_num = int(frames[fi])
                feat_path = feat_root / cam_dir.name / f"{frame_num:05d}.npy"
            if feat_path.exists():
                feat = np.load(feat_path).astype(np.float32)
                H, W, D = feat.shape
                feat_shapes.append((H, W, D))
                all_feats.append(feat.reshape(-1, D))
        if not all_feats:
            continue
        combined = np.concatenate(all_feats, axis=0)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(combined)
        pca_min, pca_max = pca_result.min(0), pca_result.max(0)
        pca_norm = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
        fig, axes = plt.subplots(1, len(cam_dirs), figsize=(4 * len(cam_dirs), 4))
        if len(cam_dirs) == 1:
            axes = [axes]
        offset = 0
        for ci, (cam_dir, shape) in enumerate(zip(cam_dirs, feat_shapes)):
            H, W, D = shape
            pca_img = pca_norm[offset:offset + H * W].reshape(H, W, 3)
            offset += H * W
            axes[ci].imshow(pca_img)
            axes[ci].set_title(f"{cam_dir.name}\n{H}x{W}x{D}")
            axes[ci].axis("off")
        fig.suptitle(f"DINOv2 Feature PCA - Frame {fi}", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"04_dinov2_pca_f{fi:04d}.png", dpi=150)
        plt.close(fig)
    print(f"  [4] DINOv2 PCA: saved")
