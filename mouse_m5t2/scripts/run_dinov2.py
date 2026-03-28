"""
Extract DINOv2 features for M5t2 MonoFusion data.

Usage:
    CUDA_VISIBLE_DEVICES=4 python run_dinov2.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --batch_size 16
"""
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def main(data_root: str, batch_size: int = 16):
    data_root = Path(data_root)
    device = "cuda"

    # Load DINOv2 via torch.hub
    print("Loading DINOv2 model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()

    # DINOv2 patch size = 14, input should be divisible by 14
    # 512 / 14 = 36.57 -> use 504 (36 patches) or 518 (37 patches)
    INPUT_SIZE = 518  # 37 * 14
    PATCH_GRID = INPUT_SIZE // 14  # 37

    transform = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find all camera sequence dirs
    img_root = data_root / "images"
    cam_dirs = sorted([d for d in img_root.iterdir() if d.is_dir()])

    for cam_dir in cam_dirs:
        seq_name = cam_dir.name
        feat_dir = data_root / "dinov2_features" / seq_name
        feat_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(cam_dir.glob("*.png"))
        if not frames:
            frames = sorted(cam_dir.glob("*.jpg"))

        print(f"\n{seq_name}: {len(frames)} frames -> {feat_dir}")

        for i in range(0, len(frames), batch_size):
            batch_paths = frames[i : i + batch_size]
            imgs = torch.stack([
                transform(Image.open(str(p)).convert("RGB"))
                for p in batch_paths
            ]).to(device)

            with torch.no_grad():
                # Get patch tokens from last layer
                feats = model.get_intermediate_layers(
                    imgs, n=1, return_class_token=False
                )[0]
                # feats: (B, num_patches, feat_dim) e.g. (B, 1369, 384) for vits14
                B, N, D = feats.shape
                feats_spatial = feats.reshape(B, PATCH_GRID, PATCH_GRID, D)
                # Save as (H, W, D) per MonoFusion expectation

            for fp, feat in zip(batch_paths, feats_spatial):
                out_path = feat_dir / fp.name.replace(".png", ".npy").replace(".jpg", ".npy")
                np.save(str(out_path), feat.cpu().numpy().astype(np.float16))

            if i % (batch_size * 4) == 0:
                print(f"  {i}/{len(frames)} done, feat shape: ({PATCH_GRID},{PATCH_GRID},{D})")

            torch.cuda.empty_cache()

    print("\nDINOv2 feature extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args.data_root, args.batch_size)
