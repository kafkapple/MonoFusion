"""
Generate MoGe monocular depth maps for M5t2 MonoFusion data.

Uses MoGe (Monocular Geometry Estimation) from the MonoFusion fork.
Falls back to Depth-Anything-V2 if MoGe is unavailable.

Usage:
    CUDA_VISIBLE_DEVICES=4 python run_moge_depth.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --monofusion_root /home/joon/dev/MonoFusion \
        --batch_size 8
"""
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def run_depth_anything(data_root: Path, batch_size: int):
    """Fallback: use Depth-Anything-V2 from transformers pipeline."""
    from transformers import pipeline

    print("Using Depth-Anything-V2 (fallback)")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                     device="cuda")

    img_root = data_root / "images"
    cam_dirs = sorted([d for d in img_root.iterdir() if d.is_dir()])

    for cam_dir in cam_dirs:
        seq_name = cam_dir.name
        # MonoFusion expects: aligned_moge_depth/{video_name}/{seq_name}/depth/
        depth_dir = data_root / "aligned_moge_depth" / "m5t2" / seq_name / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(cam_dir.glob("*.png"))
        print(f"\n{seq_name}: {len(frames)} frames -> {depth_dir}")

        for fp in tqdm(frames, desc=seq_name):
            out_path = depth_dir / fp.name.replace(".png", ".npy")
            if out_path.exists():
                continue

            img = Image.open(str(fp)).convert("RGB")
            result = pipe(img)
            depth = np.array(result["depth"]).astype(np.float32)

            # Normalize to approximate metric depth (rough scale)
            # Depth-Anything outputs relative depth, need scaling
            depth = depth / (depth.max() + 1e-8) * 5.0  # rough 0-5m range

            np.save(str(out_path), depth)

        torch.cuda.empty_cache()

    print("\nDepth estimation complete (Depth-Anything-V2)!")


def run_moge(data_root: Path, monofusion_root: Path, batch_size: int):
    """Use MonoFusion's MoGe fork for depth estimation."""
    moge_dir = monofusion_root / "preproc" / "MoGE"
    sys.path.insert(0, str(moge_dir))

    try:
        from moge.model import MoGeModel
        print("Using MoGe (MonoFusion fork)")
    except ImportError:
        print("MoGe not available, trying moge.py directly...")
        try:
            # MonoFusion fork may have different structure
            import importlib.util
            spec = importlib.util.spec_from_file_location("moge", str(moge_dir / "moge.py"))
            moge_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(moge_mod)
            print("Loaded moge.py directly")
        except Exception as e:
            print(f"MoGe import failed: {e}")
            print("Falling back to Depth-Anything-V2...")
            run_depth_anything(data_root, batch_size)
            return

    # If MoGe loaded, run it
    device = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

    img_root = data_root / "images"
    cam_dirs = sorted([d for d in img_root.iterdir() if d.is_dir()])

    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for cam_dir in cam_dirs:
        seq_name = cam_dir.name
        depth_dir = data_root / "aligned_moge_depth" / "m5t2" / seq_name / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(cam_dir.glob("*.png"))
        print(f"\n{seq_name}: {len(frames)} frames -> {depth_dir}")

        for i in range(0, len(frames), batch_size):
            batch_paths = frames[i : i + batch_size]
            imgs = torch.stack([
                transform(Image.open(str(p)).convert("RGB"))
                for p in batch_paths
            ]).to(device)

            with torch.no_grad():
                output = model.infer(imgs)
                depths = output["depth"]  # (B, H, W)

            for fp, depth in zip(batch_paths, depths):
                out_path = depth_dir / fp.name.replace(".png", ".npy")
                np.save(str(out_path), depth.cpu().numpy())

            if i % (batch_size * 4) == 0:
                print(f"  {i}/{len(frames)} done")
            torch.cuda.empty_cache()

    print("\nMoGe depth estimation complete!")


def main(data_root: str, monofusion_root: str, batch_size: int, use_fallback: bool):
    data_root = Path(data_root)
    monofusion_root = Path(monofusion_root)

    if use_fallback:
        run_depth_anything(data_root, batch_size)
    else:
        try:
            run_moge(data_root, monofusion_root, batch_size)
        except Exception as e:
            print(f"MoGe failed: {e}")
            print("Falling back to Depth-Anything-V2...")
            run_depth_anything(data_root, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--monofusion_root", type=str, default="/home/joon/dev/MonoFusion")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_fallback", action="store_true", help="Force Depth-Anything-V2")
    args = parser.parse_args()
    main(args.data_root, args.monofusion_root, args.batch_size, args.use_fallback)
