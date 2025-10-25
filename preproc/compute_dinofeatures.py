
from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")
_GROUP_PATTERN = re.compile(r"^(?P<base>.+?)_undist_cam\d+$")


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _collect_image_paths(img_dir: Path) -> list[Path]:
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths, key=_natural_key)


def _apply_frame_selection(paths: list[Path], max_frames: int | None, frame_step: int) -> list[Path]:
    selected = paths
    if max_frames is not None:
        selected = selected[:max_frames]
    if frame_step > 1:
        selected = selected[::frame_step]
    return selected


def _group_base_from_name(name: str) -> str | None:
    match = _GROUP_PATTERN.match(name)
    if match is None:
        return None
    return match.group("base")


def _resolve_group_members(img_dir: Path) -> tuple[str | None, list[Path]]:
    base = _group_base_from_name(img_dir.name)
    if base is None:
        return None, [img_dir]
    members = [
        candidate
        for candidate in img_dir.parent.iterdir()
        if candidate.is_dir() and _group_base_from_name(candidate.name) == base
    ]
    members.sort()
    if img_dir not in members:
        members.append(img_dir)
        members.sort()
    return base, members


def _mask_root_for_dir(mask_root: Path | None, mask_parent: Path | None, dir_name: str) -> Path | None:
    if mask_root is not None and mask_root.name == dir_name:
        return mask_root
    if mask_parent is None:
        return None
    candidate = mask_parent / dir_name
    if candidate.exists():
        return candidate
    return None


@dataclass(frozen=True)
class _PCASample:
    sequence_dir: Path
    image_path: Path
    mask_root: Path | None


def _prepare_pca_samples(
    *,
    group_dirs: list[Path],
    selected_paths_map: dict[Path, list[Path]],
    mask_root: Path | None,
    mask_parent: Path | None,
) -> list[_PCASample]:
    samples: list[_PCASample] = []
    for dir_path in group_dirs:
        frames = selected_paths_map.get(dir_path, [])
        if not frames:
            continue
        mask_for_dir = _mask_root_for_dir(mask_root, mask_parent, dir_path.name)
        for frame in frames:
            samples.append(_PCASample(dir_path, frame, mask_for_dir))
    return samples


def _acquire_file_lock(lock_path: Path, *, timeout: float = 3600.0, poll_interval: float = 5.0) -> None:
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if timeout >= 0 and time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            time.sleep(poll_interval)


def _release_file_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _fit_shared_pca(
    *,
    samples: list[_PCASample],
    extractor: "DinoFeatureExtractor",
    feature_shape: tuple[int, int],
    output_dim: int,
    sample_per_image: int,
    rng: np.random.Generator,
    mask_prefix: str,
    mask_key: str,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=output_dim)
    fitted_samples = 0
    if not samples:
        raise ValueError("No frames available for PCA fitting")

    desc = "[DINO] PCA pass"
    for sample in tqdm(samples, desc=desc):
        feats = extractor.extract(sample.image_path, output_size=feature_shape)
        mask = _load_mask(sample.mask_root, sample.image_path.stem, feature_shape, mask_prefix, mask_key)
        flat = feats.reshape(-1, feats.shape[-1])
        if mask is not None:
            mask_flat = mask.reshape(-1)
            flat = flat[mask_flat]
        if flat.size == 0:
            continue
        if sample_per_image > 0 and flat.shape[0] > sample_per_image:
            idx = rng.choice(flat.shape[0], sample_per_image, replace=False)
            flat = flat[idx]
        pca.partial_fit(flat)
        fitted_samples += flat.shape[0]

    if fitted_samples == 0:
        raise ValueError("No foreground pixels available for PCA with the provided masks")
    return pca


def _normalize_to_uint8(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalize per-channel data to [0, 255] for visualization.

    If a mask is provided, statistics are computed from the masked foreground and
    unmasked pixels are rendered as zero.
    """

    if mask is not None:
        mask = mask.astype(bool)
        if mask.shape != array.shape[:2]:
            raise ValueError("Mask shape must match spatial dimensions of input array")
        flat = array.reshape(-1, array.shape[-1])
        mask_flat = mask.reshape(-1)
        masked_flat = flat[mask_flat]
        if masked_flat.size == 0:
            return np.zeros_like(array, dtype=np.uint8)
        min_val = masked_flat.min(axis=0, keepdims=True)
        max_val = masked_flat.max(axis=0, keepdims=True)
        denom = np.maximum(max_val - min_val, 1e-6)
        normalized_flat = np.zeros_like(flat)
        normalized_flat[mask_flat] = np.clip((masked_flat - min_val) / denom, 0.0, 1.0)
        normalized = normalized_flat.reshape(array.shape)
    else:
        min_val = array.min(axis=(0, 1), keepdims=True)
        max_val = array.max(axis=(0, 1), keepdims=True)
        denom = np.maximum(max_val - min_val, 1e-6)
        normalized = np.clip((array - min_val) / denom, 0.0, 1.0)

    return (normalized * 255.0).astype(np.uint8)


def _load_mask(
    mask_root: Path | None,
    frame_name: str,
    expected_shape: tuple[int, int],
    mask_prefix: str,
    mask_key: str,
) -> np.ndarray | None:
    if mask_root is None:
        return None

    mask_path = mask_root / f"{mask_prefix}{frame_name}.npz"
    if not mask_path.exists():
        return None

    with np.load(mask_path) as data:
        if mask_key not in data:
            raise KeyError(f"Mask file {mask_path} missing key '{mask_key}'")
        mask = data[mask_key]

    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask at {mask_path} must be 2D after squeeze, got shape {mask.shape}")

    if mask.shape != expected_shape:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((expected_shape[1], expected_shape[0]), Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.uint8) / 255.0

    return mask.astype(bool)


def _generate_crop_boxes(
    im_size: tuple[int, int],
    n_layers: int,
    overlap_ratio: float,
    num_crops_l0: int,
) -> list[tuple[int, int, int, int]]:
    """Return multi-scale crop boxes covering an image."""

    im_h, im_w = im_size
    short_side = min(im_h, im_w)
    half_w = im_w / 2.0
    half_h = im_h / 2.0
    half_short = short_side / 2.0

    boxes: list[tuple[int, int, int, int]] = [
        (
            int(half_w - half_short),
            int(half_h - half_short),
            int(half_w + half_short),
            int(half_h + half_short),
        )
    ]

    def crop_length(orig_len: int, n_crops: int, overlap: int) -> int:
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    def reverse_overlap(orig_len: int, n_crops: int, crop: int) -> int:
        if n_crops <= 1:
            return 0
        return int((crop * n_crops - orig_len) / (n_crops - 1))

    for layer in range(n_layers):
        crops_w = num_crops_l0 ** (layer + 1) + 1 ** layer
        crops_h = num_crops_l0 ** (layer + 1)

        overlap_w = int(overlap_ratio * im_w * (2.0 / crops_w))
        overlap_h = int(overlap_ratio * im_h * (2.0 / crops_h))

        crop_w = crop_length(im_w, crops_w, overlap_w)
        crop_h = crop_length(im_h, crops_h, overlap_h)
        crop_size = max(crop_w, crop_h)

        if im_w > im_h:
            overlap_h = reverse_overlap(im_h, crops_h, crop_size)
        else:
            overlap_w = reverse_overlap(im_w, crops_w, crop_size)

        step_x = max(crop_size - overlap_w, 1)
        step_y = max(crop_size - overlap_h, 1)

        for x_idx, y_idx in product(range(crops_w), range(crops_h)):
            x0 = int(step_x * x_idx)
            y0 = int(step_y * y_idx)
            x1 = min(x0 + crop_size, im_w)
            y1 = min(y0 + crop_size, im_h)
            boxes.append((x0, y0, x1, y1))

    return boxes


def _get_preprocess_shape(old_h: int, old_w: int, long_side: int) -> tuple[int, int]:
    scale = long_side / float(max(old_h, old_w))
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    return new_h, new_w


def _postprocess_crop(
    feats: torch.Tensor,
    resized_hw: tuple[int, int],
    scaled_hw: tuple[int, int],
    model_input: int,
) -> torch.Tensor:
    feats = F.interpolate(feats, size=(model_input, model_input), mode="bilinear", align_corners=False)
    feats = feats[:, :, : resized_hw[0], : resized_hw[1]]
    feats = F.interpolate(feats, size=scaled_hw, mode="bilinear", align_corners=False)
    return feats


def _predict_tokens(
    crop: np.ndarray,
    transform: transforms.Compose,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    image = Image.fromarray(crop)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(tensor)["x_norm_patchtokens"]
    return feats


def _generate_im_feats(
    image: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    output_size: tuple[int, int],
    *,
    num_crops_l0: int,
    crop_n_layers: int,
    crop_overlap_ratio: float,
    model_input_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Aggregate multi-crop DINO features to the requested spatial resolution."""

    if image.ndim != 3:
        raise ValueError("Expected image with shape (H, W, C)")

    orig_h, orig_w = image.shape[:2]
    if output_size is None:
        output_size = (orig_h, orig_w)

    scale_h = output_size[0] / float(orig_h)
    scale_w = output_size[1] / float(orig_w)

    accum: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    for x0, y0, x1, y1 in _generate_crop_boxes(
        (orig_h, orig_w), crop_n_layers, crop_overlap_ratio, num_crops_l0
    ):
        if x1 <= x0 or y1 <= y0:
            continue

        crop = image[y0:y1, x0:x1, :]

        resized_h, resized_w = _get_preprocess_shape(crop.shape[0], crop.shape[1], model_input_size)
        tokens = _predict_tokens(crop, transform, model, device)

        grid_size = int(math.sqrt(tokens.shape[1]))
        crop_feat = tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[2]).permute(0, 3, 1, 2)

        scaled_h = max(int(round(crop.shape[0] * scale_h)), 1)
        scaled_w = max(int(round(crop.shape[1] * scale_w)), 1)
        crop_feat = _postprocess_crop(crop_feat, (resized_h, resized_w), (scaled_h, scaled_w), model_input_size)
        crop_feat = crop_feat.squeeze(0)

        if accum is None:
            feat_dim = crop_feat.shape[0]
            accum = torch.zeros(feat_dim, output_size[0], output_size[1], device=device)
            weights = torch.zeros(1, output_size[0], output_size[1], device=device)

        dest_y0 = max(int(round(scale_h * y0)), 0)
        dest_x0 = max(int(round(scale_w * x0)), 0)
        dest_y1 = min(dest_y0 + scaled_h, output_size[0])
        dest_x1 = min(dest_x0 + scaled_w, output_size[1])

        slice_h = dest_y1 - dest_y0
        slice_w = dest_x1 - dest_x0
        if slice_h <= 0 or slice_w <= 0:
            continue

        accum[:, dest_y0:dest_y1, dest_x0:dest_x1] += crop_feat[:, :slice_h, :slice_w]
        weights[:, dest_y0:dest_y1, dest_x0:dest_x1] += 1

    if accum is None or weights is None:
        raise ValueError("Failed to generate any DINO features from provided image")

    accum = accum / torch.clamp(weights, min=1.0)
    return accum.permute(1, 2, 0).contiguous()


@dataclass
class DinoFeatureExtractor:
    model_name: str = "dinov2_vits14_reg"
    model_input_size: int = 896
    device: str = "cuda"
    num_crops_l0: int = 4
    crop_n_layers: int = 1
    crop_overlap_ratio: float = 512.0 / 1500.0

    def __post_init__(self) -> None:
        torch.set_grad_enabled(False)
        self.torch_device = torch.device(self.device)
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name).to(self.torch_device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.model_input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.model_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def extract(self, path: Path, output_size: tuple[int, int]) -> np.ndarray:
        image = np.asarray(Image.open(path).convert("RGB"))
        feats = _generate_im_feats(
            image,
            self.model,
            self.transform,
            output_size,
            num_crops_l0=self.num_crops_l0,
            crop_n_layers=self.crop_n_layers,
            crop_overlap_ratio=self.crop_overlap_ratio,
            model_input_size=self.model_input_size,
            device=self.torch_device,
        )
        return feats.cpu().numpy().astype(np.float32)


def run(
    img_dir: str,
    out_dir: str,
    device: str = "cuda",
    model_name: str = "dinov2_vits14_reg",
    model_input_size: int = 896,
    num_crops_l0: int = 4,
    crop_n_layers: int = 1,
    crop_overlap_ratio: float = 512.0 / 1500.0,
    output_height: int = 288,
    output_width: int = 512,
    output_dim: int = 32,
    sample_per_image: int = 8192,
    random_seed: int = 0,
    overwrite: bool = False,
    max_frames: int | None = 300,
    frame_step: int = 3,
    save_viz: bool = True,
    viz_dirname: str = "pca_viz",
    save_pca_model: bool = True,
    mask_dir: str | None = None,
    mask_prefix: str = "dyn_mask_",
    mask_key: str = "dyn_mask",
) -> None:
    img_dir_path = Path(img_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    mask_root_path: Path | None = None
    if mask_dir is not None:
        mask_root_path = Path(mask_dir)
        if not mask_root_path.exists():
            raise FileNotFoundError(f"Mask directory '{mask_root_path}' does not exist")
        mask_root_path = mask_root_path.resolve()

    image_paths = _collect_image_paths(img_dir_path)
    if not image_paths:
        raise FileNotFoundError(f"No images with extensions {SUPPORTED_EXTS} found in {img_dir}")

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    group_base, group_dirs = _resolve_group_members(img_dir_path)

    selected_paths_map: dict[Path, list[Path]] = {}
    for dir_path in group_dirs:
        if dir_path == img_dir_path:
            frames = image_paths
        else:
            frames = _collect_image_paths(dir_path)
        selected = _apply_frame_selection(frames, max_frames, frame_step)
        selected_paths_map[dir_path] = selected

    selected_paths = selected_paths_map.get(img_dir_path, [])
    if not selected_paths:
        raise ValueError("No images selected after applying max_frames and frame_step filters")

    sample_needed = output_dim > 0 and output_dim < 384
    feature_shape = (output_height, output_width)

    extractor = DinoFeatureExtractor(
        model_name=model_name,
        model_input_size=model_input_size,
        device=device,
        num_crops_l0=num_crops_l0,
        crop_n_layers=crop_n_layers,
        crop_overlap_ratio=crop_overlap_ratio,
    )

    rng = np.random.default_rng(random_seed)
    pca: IncrementalPCA | None = None
    if sample_needed:
        mask_parent = mask_root_path.parent if mask_root_path is not None else None
        samples = _prepare_pca_samples(
            group_dirs=group_dirs,
            selected_paths_map=selected_paths_map,
            mask_root=mask_root_path,
            mask_parent=mask_parent,
        )

        if group_base is not None:
            pca_cache_path = out_dir_path.parent / f"{group_base}_pca_model.pkl"
        else:
            pca_cache_path = out_dir_path / "pca_model.pkl"
        pca_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if pca_cache_path.exists():
            with open(pca_cache_path, "rb") as f:
                pca = pickle.load(f)
        else:
            lock_path = pca_cache_path.with_suffix(pca_cache_path.suffix + ".lock")
            _acquire_file_lock(lock_path)
            try:
                if pca_cache_path.exists():
                    with open(pca_cache_path, "rb") as f:
                        pca = pickle.load(f)
                else:
                    group_label = group_base or img_dir_path.name
                    unique_dirs = sorted({sample.sequence_dir.name for sample in samples})
                    print(
                        f"[DINO] Gathering PCA statistics from {len(samples)} frames "
                        f"across {len(unique_dirs)} paths for group '{group_label}'"
                    )
                    pca = _fit_shared_pca(
                        samples=samples,
                        extractor=extractor,
                        feature_shape=feature_shape,
                        output_dim=output_dim,
                        sample_per_image=sample_per_image,
                        rng=rng,
                        mask_prefix=mask_prefix,
                        mask_key=mask_key,
                    )
                    with open(pca_cache_path, "wb") as f:
                        pickle.dump(pca, f)
            finally:
                _release_file_lock(lock_path)

        if pca is None:
            with open(pca_cache_path, "rb") as f:
                pca = pickle.load(f)

    viz_dir: Path | None = None
    if save_viz:
        viz_dir = out_dir_path / viz_dirname
        viz_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "original_frame_count": len(image_paths),
        "selected_frame_count": len(selected_paths),
        "frame_step": frame_step,
        "max_frames": max_frames,
        "output_dim": output_dim,
        "feature_height": output_height,
        "feature_width": output_width,
        "model_name": model_name,
        "model_input_size": model_input_size,
        "num_crops_l0": num_crops_l0,
        "crop_n_layers": crop_n_layers,
        "crop_overlap_ratio": crop_overlap_ratio,
        "sample_per_image": sample_per_image,
        "random_seed": random_seed,
        "frames": [p.stem for p in selected_paths],
    }
    if save_viz and viz_dir is not None:
        metadata["viz_dir"] = viz_dirname
        metadata["viz_channels"] = 3
    else:
        metadata["viz_channels"] = 0
    if mask_root_path is not None:
        metadata["mask_dir"] = str(mask_root_path)
        metadata["mask_prefix"] = mask_prefix
        metadata["mask_key"] = mask_key

    print("[DINO] Saving feature maps")
    for path in tqdm(selected_paths, desc="[DINO] Feature pass"):
        frame_name = path.stem
        out_file = out_dir_path / f"{frame_name}.npy"
        if out_file.exists() and not overwrite:
            continue

        feats = extractor.extract(path, output_size=feature_shape)
        mask = _load_mask(mask_root_path, frame_name, feature_shape, mask_prefix, mask_key)
        if pca is not None:
            original_shape = feats.shape[:-1]
            flat = feats.reshape(-1, feats.shape[-1])
            if mask is not None:
                mask_flat = mask.reshape(-1)
                reduced = np.zeros((flat.shape[0], pca.n_components), dtype=np.float32)
                if mask_flat.any():
                    reduced_masked = pca.transform(flat[mask_flat])
                    reduced[mask_flat] = reduced_masked
                feats = reduced.reshape(*original_shape, -1)
            else:
                feats = pca.transform(flat).reshape(*original_shape, -1)
        elif mask is not None:
            feats = feats * mask[..., None]

        feats = feats.astype(np.float32)
        np.save(out_file, feats)

        if viz_dir is not None and feats.shape[-1] >= 3:
            viz = _normalize_to_uint8(feats[..., :3], mask=mask)
            Image.fromarray(viz).save(viz_dir / f"{frame_name}.png")

    if save_pca_model and pca is not None:
        with open(out_dir_path / "pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)

    with open(out_dir_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DINO] Features written to {out_dir_path} (total {len(selected_paths)} frames)")


if __name__ == "__main__":
    tyro.cli(run)
