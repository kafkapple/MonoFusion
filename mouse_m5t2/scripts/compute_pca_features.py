"""
Reduce DINOv2 384d features to PCA 32d for MonoFusion training.

Reads existing .npy files (H, W, 384) in float16, fits PCA on a subsample,
transforms all features to (H, W, 32) in float32, saves to a new directory.

Usage:
    python compute_pca_features.py \
        --data_root /node_data/joon/data/monofusion/m5t2_v5 \
        --pca_dim 32 \
        --sample_per_cam 20
"""
import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import pickle


def main(data_root: str, pca_dim: int = 32, sample_per_cam: int = 20):
    data_root = Path(data_root)
    feat_root = data_root / "dinov2_features"
    out_root = data_root / f"dinov2_features_pca{pca_dim}"

    cam_dirs = sorted([d for d in feat_root.iterdir() if d.is_dir()])
    print(f"Found {len(cam_dirs)} camera dirs in {feat_root}")

    # Step 1: Collect samples for PCA fitting
    print(f"\nStep 1: Sampling features for PCA fitting ({sample_per_cam}/cam)...")
    all_samples = []
    for cam_dir in cam_dirs:
        npy_files = sorted(cam_dir.glob("*.npy"))
        # Subsample frames evenly
        indices = np.linspace(0, len(npy_files) - 1, min(sample_per_cam, len(npy_files)))
        indices = np.unique(indices.astype(int))

        for idx in indices:
            feat = np.load(str(npy_files[idx])).astype(np.float32)  # (H, W, 384)
            H, W, D = feat.shape
            # Subsample spatially (every 4th pixel) to keep memory reasonable
            flat = feat[::4, ::4].reshape(-1, D)  # (~H/4 * W/4, 384)
            all_samples.append(flat)

    all_samples = np.concatenate(all_samples, axis=0)  # (N, 384)
    print(f"  PCA input: {all_samples.shape} ({all_samples.nbytes / 1e6:.1f} MB)")

    # Step 2: Fit PCA
    print(f"\nStep 2: Fitting PCA ({all_samples.shape[1]}d → {pca_dim}d)...")
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(all_samples)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.4f} ({explained*100:.1f}%)")

    # Save PCA model
    pca_path = out_root / "pca_model.pkl"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  PCA model saved: {pca_path}")

    # Step 3: Transform all features
    print(f"\nStep 3: Transforming all features...")
    total_files = 0
    for cam_dir in cam_dirs:
        cam_out = out_root / cam_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(cam_dir.glob("*.npy"))
        for npy_file in npy_files:
            feat = np.load(str(npy_file)).astype(np.float32)  # (H, W, 384)
            H, W, D = feat.shape
            flat = feat.reshape(-1, D)  # (H*W, 384)
            reduced = pca.transform(flat)  # (H*W, pca_dim)
            reduced = reduced.reshape(H, W, pca_dim)  # (H, W, pca_dim)
            np.save(str(cam_out / npy_file.name), reduced.astype(np.float32))
            total_files += 1

        print(f"  {cam_dir.name}: {len(npy_files)} files → {cam_out}")

    # Summary
    sample_file = next(out_root.rglob("*.npy"))
    sample = np.load(str(sample_file))
    mem_per_frame = sample.nbytes
    print(f"\nDone! {total_files} files transformed.")
    print(f"  Output shape: {sample.shape} ({sample.dtype})")
    print(f"  Per-frame size: {mem_per_frame / 1024:.1f} KB "
          f"(was {mem_per_frame * D // pca_dim / 1024:.1f} KB)")
    print(f"  Output dir: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--sample_per_cam", type=int, default=20,
                        help="Frames to sample per camera for PCA fitting")
    args = parser.parse_args()
    main(args.data_root, args.pca_dim, args.sample_per_cam)
