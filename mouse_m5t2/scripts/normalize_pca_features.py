"""
L2-normalize PCA features for MonoFusion training stability.

Reads (H, W, D) .npy files, applies per-pixel L2 normalization along
the feature dimension, and saves to a new directory.

Usage:
    python normalize_pca_features.py \
        --data_root /node_data/joon/data/monofusion/m5t2_v5 \
        --pca_dim 32
"""
import argparse
import numpy as np
from pathlib import Path


def main(data_root: str, pca_dim: int = 32):
    data_root = Path(data_root)
    src_dir = data_root / f"dinov2_features_pca{pca_dim}"
    out_dir = data_root / f"dinov2_features_pca{pca_dim}_norm"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    cam_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    print(f"Normalizing {len(cam_dirs)} camera dirs from {src_dir}")

    norm_stats = []
    total_files = 0

    for cam_dir in cam_dirs:
        cam_out = out_dir / cam_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(cam_dir.glob("*.npy"))
        for npy_file in npy_files:
            feat = np.load(str(npy_file)).astype(np.float32)  # (H, W, D)
            norms = np.linalg.norm(feat, axis=-1, keepdims=True)  # (H, W, 1)
            norms = np.clip(norms, 1e-8, None)
            normalized = feat / norms
            np.save(str(cam_out / npy_file.name), normalized.astype(np.float32))
            norm_stats.append(norms.mean())
            total_files += 1

        print(f"  {cam_dir.name}: {len(npy_files)} files")

    # Copy PCA model for reference
    pca_model = src_dir / "pca_model.pkl"
    if pca_model.exists():
        import shutil
        shutil.copy2(str(pca_model), str(out_dir / "pca_model.pkl"))

    norm_stats = np.array(norm_stats)
    print(f"\nDone! {total_files} files normalized.")
    print(f"  Norm stats — mean: {norm_stats.mean():.4f}, std: {norm_stats.std():.4f}, "
          f"min: {norm_stats.min():.4f}, max: {norm_stats.max():.4f}")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pca_dim", type=int, default=32)
    args = parser.parse_args()
    main(args.data_root, args.pca_dim)
