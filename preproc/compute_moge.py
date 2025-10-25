
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
MOGE_REPO = REPO_ROOT / "MoGE"
if MOGE_REPO.exists():
    repo_path = str(MOGE_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
else:
    raise ImportError(
        "Unable to locate local MoGE repository. Expected directory at "
        f"{MOGE_REPO}. Please clone/build MoGE or install it as a package."
    )

from moge.model import MoGeModel

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _collect_image_paths(img_dir: Path) -> list[Path]:
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths, key=_natural_key)


def _resize_map(array: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0).cpu().numpy()


def _ensure_dir(path: Path, enable: bool) -> Path | None:
    if not enable:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(
    img_dir: str,
    out_dir: str,
    device: str = "cuda",
    model_name: str = "Ruicheng/moge-vitl",
    target_height: int | None = None,
    target_width: int | None = None,
    store_mask: bool = True,
    store_pointcloud: bool = True,
    store_intrinsics: bool = True,
    overwrite: bool = False,
    max_frames: int | None = None,
    frame_step: int = 1,
) -> None:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    img_dir_path = Path(img_dir)
    out_dir_path = Path(out_dir)
    depth_dir = out_dir_path / "depth"
    mask_dir = out_dir_path / "mask"
    point_dir = out_dir_path / "pointcloud"
    intr_dir = out_dir_path / "intrinsics"
    depth_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dir(mask_dir, store_mask)
    _ensure_dir(point_dir, store_pointcloud)
    _ensure_dir(intr_dir, store_intrinsics)

    image_paths = _collect_image_paths(img_dir_path)
    if not image_paths:
        raise FileNotFoundError(f"No images with extensions {SUPPORTED_EXTS} found in {img_dir}")

    selected_paths = image_paths
    if max_frames is not None:
        selected_paths = selected_paths[:max_frames]
    if frame_step > 1:
        selected_paths = selected_paths[::frame_step]
    if not selected_paths:
        raise ValueError("No images selected after applying max_frames and frame_step filters")

    print(f"[MoGE] Loading model {model_name} on {device}")
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    metadata = {
        "original_frame_count": len(image_paths),
        "selected_frame_count": len(selected_paths),
        "frame_step": frame_step,
        "max_frames": max_frames,
        "model_name": model_name,
        "target_height": target_height,
        "target_width": target_width,
        "device": device,
        "store_mask": store_mask,
        "store_pointcloud": store_pointcloud,
        "store_intrinsics": store_intrinsics,
        "overwrite": overwrite,
        "frames": [],
    }

    for path in tqdm(selected_paths, desc="[MoGE] Processing"):
        frame_name = path.stem
        depth_path = depth_dir / f"{frame_name}.npy"
        if depth_path.exists() and not overwrite:
            continue
        metadata["frames"].append(frame_name)

        image = Image.open(path).convert("RGB")
        input_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)

        with torch.no_grad():
            output = model.infer(input_tensor)

        depth = output["depth"].detach().cpu().numpy().astype(np.float32)
        if target_height is not None and target_width is not None:
            depth = _resize_map(depth, target_height, target_width)
        np.save(depth_path, depth)

        if store_mask and "mask" in output:
            mask = output["mask"].detach().cpu().numpy().astype(np.float32)
            if target_height is not None and target_width is not None:
                mask = _resize_map(mask, target_height, target_width)
            np.save(mask_dir / f"{frame_name}.npy", mask)

        if store_pointcloud and "points" in output:
            points = output["points"].detach().cpu().numpy().astype(np.float32)
            np.savez(point_dir / f"{frame_name}.npz", xyz=points)

        if store_intrinsics and "intrinsics" in output:
            intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float32)
            np.save(intr_dir / f"{frame_name}.npy", intrinsics)

    with (out_dir_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[MoGE] Outputs written to {out_dir_path} (total {len(metadata['frames'])} frames)")


if __name__ == "__main__":
    tyro.cli(run)
