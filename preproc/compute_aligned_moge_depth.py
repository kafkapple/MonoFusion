from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_ROOT_CANDIDATES = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT.parent / "data",
    _SCRIPT_DIR / "data",
]
for candidate in _DATA_ROOT_CANDIDATES:
    if candidate.exists():
        _DEFAULT_DATA_ROOT = candidate
        break
else:
    _DEFAULT_DATA_ROOT = _SCRIPT_DIR

_DEFAULT_RAW_ROOT = (_PROJECT_ROOT / "_raw_data" if (_PROJECT_ROOT / "_raw_data").exists()
                     else _PROJECT_ROOT.parent / "raw_data")


def _candidate_mask_roots(mask_root: Path, image_root: Path) -> list[Path]:
        roots: list[Path] = []
        for candidate in (
            mask_root,
            mask_root.parent / "sam_v2_dyn_mask",
            mask_root.parent / "masks",
            image_root.parent / "sam_v2_dyn_mask",
            image_root.parent / "masks",
        ):
            if candidate not in roots:
                roots.append(candidate)
        return roots

@dataclass
class AlignMoGEConfig:
    seq_name: str
    target_name: str | None = None
    raw_moge_root: Path = _DEFAULT_DATA_ROOT / "raw_moge_depth"
    dust3r_root: Path = _DEFAULT_DATA_ROOT / "dust3r"
    mask_root: Path = _DEFAULT_DATA_ROOT / "sam_v2_dyn_mask"
    image_root: Path = _DEFAULT_DATA_ROOT / "images"
    output_root: Path = _DEFAULT_DATA_ROOT / "aligned_moge_depth"
    raw_root: Path = _DEFAULT_RAW_ROOT
    min_valid_pixels: int = 500
    epsilon: float = 1e-6
    min_depth_quantile: float = 0.01
    store_pointcloud: bool = True
    overwrite: bool = False


def _resolve_sequence_name(seq_name: str, target_override: str | None) -> tuple[str, str]:
    seq_key = seq_name.lstrip("_")
    if not seq_key:
        raise ValueError("Sequence name must contain non-underscore characters")
    if target_override:
        return seq_key, target_override
    parts = [part for part in seq_key.split("_") if part]
    if len(parts) >= 2:
        return seq_key, parts[1]
    return seq_key, parts[0]


def _resolve_image_path(img_dir: Path, frame_name: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = img_dir / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _camera_uid(cam_name: str) -> str:
    match = re.search(r"(\d+)$", cam_name)
    if match is None:
        raise ValueError(f"Cannot infer camera uid from '{cam_name}'")
    return f"cam{match.group(1)}"


def _load_calibrations(raw_root: Path, seq_key: str) -> dict[str, dict[str, float]]:
    csv_path = raw_root / seq_key / "trajectory" / "gopro_calibs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing calibration file at {csv_path}")
    calibrations: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["cam_uid"]
            cal: dict[str, float] = {}
            for field, value in row.items():
                try:
                    cal[field] = float(value)
                except (TypeError, ValueError):
                    cal[field] = value  # type: ignore[assignment]
            calibrations[key] = cal
    if not calibrations:
        raise ValueError(f"No calibration entries found in {csv_path}")
    return calibrations


def _quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("Quaternion has zero length")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _build_intrinsics(cal: dict[str, float], width: int, height: int) -> np.ndarray:
    img_w = cal.get("image_width", float(width)) or float(width)
    img_h = cal.get("image_height", float(height)) or float(height)
    fx = cal.get("intrinsics_0", 1.0) / img_w * width
    fy = cal.get("intrinsics_1", 1.0) / img_h * height
    cx = cal.get("intrinsics_2", img_w / 2) / img_w * width
    cy = cal.get("intrinsics_3", img_h / 2) / img_h * height
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _build_extrinsics(cal: dict[str, float]) -> np.ndarray:
    rotation = _quaternion_to_rotation_matrix(
        cal.get("qw_world_cam", 1.0),
        cal.get("qx_world_cam", 0.0),
        cal.get("qy_world_cam", 0.0),
        cal.get("qz_world_cam", 0.0),
    )
    translation = np.array(
        [cal.get("tx_world_cam", 0.0), cal.get("ty_world_cam", 0.0), cal.get("tz_world_cam", 0.0)],
        dtype=np.float32,
    ).reshape(3, 1)
    extrinsic = np.concatenate([np.concatenate([rotation, translation], axis=1), np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
    return extrinsic


def _depth_to_points(depth_map: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth_map.shape
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_map
    x = (grid_x - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (grid_y - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1)
    return points


def _load_rgb(image_path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if image.size != (size[1], size[0]):
        image = image.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _load_npz_mask(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        mask = data["dyn_mask"]
    if mask.ndim == 4:
        mask = mask[0, 0]
    return mask.astype(bool)


def _save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run(cfg: AlignMoGEConfig) -> None:
    seq_key, target_name = _resolve_sequence_name(cfg.seq_name, cfg.target_name)

    image_root = cfg.image_root.resolve()
    mask_root = cfg.mask_root.resolve()
    raw_moge_root = cfg.raw_moge_root.resolve()
    dust_root = cfg.dust3r_root.resolve()
    output_root = cfg.output_root.resolve()
    raw_root = cfg.raw_root.resolve()

    seq_root = output_root / seq_key
    seq_root.mkdir(parents=True, exist_ok=True)

    calibrations = _load_calibrations(raw_root, seq_key)

    camera_names = sorted({path.name for path in image_root.glob(f"{target_name}_undist_cam*") if path.is_dir()})
    if not camera_names:
        raise FileNotFoundError(f"No camera directories found for target '{target_name}' in {image_root}")

    dust_seq_root = dust_root / seq_key
    if not dust_seq_root.exists():
        raise FileNotFoundError(f"DUSt3R outputs not found at {dust_seq_root}")

    camera_outputs: dict[str, dict[str, Path]] = {}
    camera_stats: dict[str, dict[str, Any]] = {}
    last_scales: dict[str, float] = {}

    for cam_name in camera_names:
        raw_depth_dir = raw_moge_root / cam_name / "depth"

        dust_cam_root = dust_seq_root / cam_name
        if not dust_cam_root.exists():
            dust_cam_root = None
            try:
                candidates = sorted(
                    [path for path in dust_seq_root.iterdir() if path.is_dir()],
                    key=lambda path: path.name,
                )
            except FileNotFoundError:
                candidates = []
            for candidate_parent in candidates:
                candidate = candidate_parent / cam_name
                if candidate.exists():
                    dust_cam_root = candidate
                    break

        if dust_cam_root is not None:
            dust_depth_dir = dust_cam_root / "conf_bg_depth"
            dust_conf_dir = dust_cam_root / "confidence"
        else:
            dust_depth_dir = None
            dust_conf_dir = None

        mask_dir = mask_root / cam_name
        if not mask_dir.exists():
            mask_dir = None
            for candidate_root in _candidate_mask_roots(mask_root, image_root):
                potential = candidate_root / cam_name
                if potential.exists():
                    mask_dir = potential
                    break

        if (
            mask_dir is None
            or not raw_depth_dir.exists()
            or dust_depth_dir is None
            or dust_conf_dir is None
            or not dust_depth_dir.exists()
            or not dust_conf_dir.exists()
        ):
            print(f"[AlignMoGE] Skipping camera {cam_name}: missing inputs")
            continue

        output_depth_dir = seq_root / cam_name / "depth"
        output_depth_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_depth_dir.parent / "metadata.json"

        camera_outputs[cam_name] = {
            "raw_depth": raw_depth_dir,
            "dust_depth": dust_depth_dir,
            "dust_conf": dust_conf_dir,
            "mask": mask_dir,
            "image": image_root / cam_name,
            "output_depth": output_depth_dir,
            "metadata": metadata_path,
        }

        camera_stats[cam_name] = {"frames": set(), "writes": 0, "scales": {}, "fallbacks": 0}

    if not camera_outputs:
        raise ValueError("No cameras with both raw MoGE depth and DUSt3R outputs were found")

    # Aggregate all frame names available in dust3r background depth directories
    frame_names: set[str] = set()
    for info in camera_outputs.values():
        for npy in info["dust_depth"].glob("*.npy"):
            frame_names.add(npy.stem)
    frame_names = set(sorted(frame_names))

    pointcloud_dir = seq_root / "pointcloud"
    if cfg.store_pointcloud:
        pointcloud_dir.mkdir(parents=True, exist_ok=True)

    sequence_frames: set[str] = set()

    for frame_name in tqdm(sorted(frame_names), desc="[AlignMoGE] Frames"):
        frame_pointclouds: list[np.ndarray] = []
        for cam_name, info in camera_outputs.items():
            dust_depth_path = info["dust_depth"] / f"{frame_name}.npy"
            raw_depth_path = info["raw_depth"] / f"{frame_name}.npy"
            conf_path = info["dust_conf"] / f"{frame_name}.npy"
            mask_path = info["mask"] / f"dyn_mask_{frame_name}.npz"

            if not dust_depth_path.exists() or not raw_depth_path.exists() or not conf_path.exists() or not mask_path.exists():
                continue

            dust_depth = np.load(dust_depth_path)
            raw_depth = np.load(raw_depth_path)
            confidence_map = np.load(conf_path)
            fg_mask = _load_npz_mask(mask_path)

            if dust_depth.shape != raw_depth.shape:
                if raw_depth.size == 0:
                    continue
                raw_depth = np.resize(raw_depth, dust_depth.shape)
            if confidence_map.shape != dust_depth.shape:
                confidence_map = np.resize(confidence_map, dust_depth.shape)
            if fg_mask.shape != dust_depth.shape:
                fg_mask = np.resize(fg_mask, dust_depth.shape)

            valid_mask = (dust_depth > 0) & (raw_depth > 0) & (~fg_mask) & (confidence_map > 0)
            valid_pixels = int(np.count_nonzero(valid_mask))

            fallback = False
            if valid_pixels < cfg.min_valid_pixels:
                scale = last_scales.get(cam_name, 1.0)
                fallback = True
            else:
                ratios = dust_depth[valid_mask] / (raw_depth[valid_mask] + cfg.epsilon)
                scale = float(np.median(ratios))
                if not np.isfinite(scale):
                    scale = last_scales.get(cam_name, 1.0)
                    fallback = True
                else:
                    last_scales[cam_name] = scale

            scale = float(np.clip(scale, 1e-3, 1e3))

            aligned_depth = raw_depth * scale
            aligned_depth[raw_depth <= 0] = 0.0

            positive = aligned_depth > 0
            if np.any(positive):
                threshold = min(1e-6, float(np.quantile(aligned_depth[positive], cfg.min_depth_quantile)))
                aligned_depth[aligned_depth < threshold] = 0.0

            output_depth_path = info["output_depth"] / f"{frame_name}.npy"
            if cfg.overwrite or not output_depth_path.exists():
                np.save(output_depth_path, aligned_depth.astype(np.float32))
                camera_stats[cam_name]["writes"] = int(camera_stats[cam_name]["writes"]) + 1

            frames_set = camera_stats[cam_name]["frames"]
            assert isinstance(frames_set, set)
            frames_set.add(frame_name)
            camera_stats[cam_name]["scales"][frame_name] = {
                "scale": scale,
                "valid_pixels": valid_pixels,
                "fallback": fallback,
            }
            if fallback:
                camera_stats[cam_name]["fallbacks"] = int(camera_stats[cam_name]["fallbacks"]) + 1

            if cfg.store_pointcloud and np.any(aligned_depth > 0):
                cam_uid = _camera_uid(cam_name)
                if cam_uid not in calibrations:
                    continue
                cal = calibrations[cam_uid]
                intr = _build_intrinsics(cal, aligned_depth.shape[1], aligned_depth.shape[0])
                extr = _build_extrinsics(cal)
                image_path = _resolve_image_path(info["image"], frame_name)
                if image_path is None:
                    continue
                rgb = _load_rgb(image_path, aligned_depth.shape)
                points_cam = _depth_to_points(aligned_depth, intr)
                points_cam_flat = points_cam.reshape(-1, 3)
                rgb_flat = rgb.reshape(-1, 3)
                valid_points = aligned_depth.reshape(-1) > 0
                if not np.any(valid_points):
                    continue
                homo = np.concatenate([points_cam_flat[valid_points], np.ones((valid_points.sum(), 1), dtype=np.float32)], axis=1)
                world = (extr @ homo.T).T[:, :3]
                pc = np.concatenate([world.astype(np.float32), rgb_flat[valid_points].astype(np.float32)], axis=1)
                if pc.size > 0:
                    frame_pointclouds.append(pc)
        if frame_pointclouds and cfg.store_pointcloud:
            stacked_pc = np.concatenate(frame_pointclouds, axis=0)
            np.savez(pointcloud_dir / f"{frame_name}.npz", data=stacked_pc)
        if frame_pointclouds:
            sequence_frames.add(frame_name)

    for cam_name, stats in camera_stats.items():
        metadata_path = camera_outputs[cam_name]["metadata"]
        existing = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = {}
        frames = set(existing.get("frames", [])) | set(stats["frames"])
        metadata = {
            "sequence": cfg.seq_name,
            "target": target_name,
            "camera": cam_name,
            "frames": sorted(frames),
            "total_frames": len(frames),
            "new_writes": int(stats["writes"]),
            "fallback_scales": int(stats["fallbacks"]),
            "scales": stats["scales"],
        }
        _save_metadata(metadata_path, metadata)

    seq_metadata_path = seq_root / "metadata.json"
    existing_seq = {}
    if seq_metadata_path.exists():
        with seq_metadata_path.open("r", encoding="utf-8") as f:
            try:
                existing_seq = json.load(f)
            except json.JSONDecodeError:
                existing_seq = {}
    frames = set(existing_seq.get("frames", [])) | sequence_frames
    seq_metadata = {
        "sequence": cfg.seq_name,
        "target": target_name,
        "frames": sorted(frames),
        "total_frames": len(frames),
        "cameras": sorted(camera_outputs.keys()),
    }
    _save_metadata(seq_metadata_path, seq_metadata)


if __name__ == "__main__":
    run(tyro.cli(AlignMoGEConfig))
