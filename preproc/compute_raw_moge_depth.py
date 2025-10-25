from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm

import compute_moge
from compute_moge import _collect_image_paths, _resize_map  # type: ignore


MoGeModel = compute_moge.MoGeModel


_REPO_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _REPO_ROOT.parent.resolve()
_DEFAULT_IMAGE_ROOT = (_PROJECT_ROOT / "data" / "images").resolve()
_DEFAULT_OUTPUT_ROOT = (_PROJECT_ROOT / "data" / "raw_moge_depth").resolve()
_DEFAULT_RAW_ROOT = (
    (_PROJECT_ROOT / "_raw_data").resolve()
    if (_PROJECT_ROOT / "_raw_data").exists()
    else (_PROJECT_ROOT.parent / "raw_data").resolve()
)


@dataclass
class RawMoGEDepthConfig:
    seq_name: str
    target_name: str | None = None
    camera_ids: Sequence[int] = (1, 2, 3, 4)
    image_root: Path = _DEFAULT_IMAGE_ROOT
    output_root: Path = _DEFAULT_OUTPUT_ROOT
    raw_root: Path = _DEFAULT_RAW_ROOT
    device: str = "cuda"
    model_name: str = "Ruicheng/moge-vitl"
    frame_step: int = 3
    max_frames: int | None = 300
    target_height: int | None = None
    target_width: int | None = None
    overwrite: bool = False
    camera_dir_template: str = "{target}_undist_cam{cam:02d}"


def _resolve_sequence_key(seq_name: str) -> str:
    seq_key = seq_name.lstrip("_")
    if not seq_key:
        raise ValueError(f"Cannot derive sequence key from '{seq_name}'")
    return seq_key


def _resolve_target_name(seq_name: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    seq_key = _resolve_sequence_key(seq_name)
    parts = [part for part in seq_key.split("_") if part]
    if not parts:
        raise ValueError(f"Cannot infer target name from sequence '{seq_name}'")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def _ordered_camera_ids(camera_ids: Iterable[int]) -> list[int]:
    seen = set()
    ordered: list[int] = []
    for cam in camera_ids:
        if cam in seen:
            continue
        if cam < 0:
            raise ValueError(f"Camera id must be non-negative, got {cam}")
        seen.add(cam)
        ordered.append(cam)
    ordered.sort()
    if not ordered:
        raise ValueError("camera_ids cannot be empty")
    return ordered


def _find_camera_dir(image_root: Path, template: str, target: str, cam: int) -> Path:
    candidate = image_root / template.format(target=target, cam=cam)
    if candidate.exists():
        return candidate
    pattern = f"*undist_cam{cam:02d}"
    matches = [path for path in image_root.glob(pattern) if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    filtered = [path for path in matches if target in path.name]
    if len(filtered) == 1:
        return filtered[0]
    raise FileNotFoundError(
        f"Unable to locate camera directory for cam={cam} under '{image_root}'. "
        f"Tried template '{template}'."
    )


def _camera_uid_from_dirname(name: str) -> str:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Cannot infer camera uid from '{name}'")
    return f"cam{int(match.group(1)):02d}"


def _load_calibrations(raw_root: Path, seq_key: str) -> tuple[dict[str, dict[str, float | str]], Path]:
    csv_path = raw_root / seq_key / "trajectory" / "gopro_calibs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing calibration file at {csv_path}")

    calibrations: dict[str, dict[str, float | str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_uid = row.get("cam_uid")
            if not cam_uid:
                continue
            cal: dict[str, float | str] = {}
            for field, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    cal[field] = float(value)
                except (TypeError, ValueError):
                    cal[field] = value
            calibrations[cam_uid] = cal

    if not calibrations:
        raise ValueError(f"No calibration entries found in {csv_path}")
    return calibrations, csv_path


def _horizontal_fov(calibration: dict[str, float | str]) -> float:
    width = float(calibration.get("image_width", 0.0) or 0.0)
    fx = float(calibration.get("intrinsics_0", 0.0) or 0.0)
    if width <= 0 or fx <= 0:
        raise ValueError("Invalid calibration values for FoV computation")
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def _load_existing_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _infer_camera(
    *,
    model: MoGeModel,
    device: torch.device,
    img_dir: Path,
    out_dir: Path,
    frame_step: int,
    max_frames: int | None,
    target_height: int | None,
    target_width: int | None,
    overwrite: bool,
    fov_x: float | None,
) -> tuple[int, list[str]]:
    image_paths = _collect_image_paths(img_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{img_dir}'")

    selected = image_paths
    if max_frames is not None:
        selected = selected[:max_frames]
    if frame_step > 1:
        selected = selected[::frame_step]
    if not selected:
        return 0, []

    depth_dir = out_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    writes = 0
    for img_path in tqdm(selected, desc=f"[MoGE] {img_dir.name}"):
        frame_name = img_path.stem
        depth_path = depth_dir / f"{frame_name}.npy"
        if depth_path.exists() and not overwrite:
            processed.append(frame_name)
            continue

        image = Image.open(img_path).convert("RGB")
        tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)

        with torch.no_grad():
            output = model.infer(tensor, fov_x=fov_x)

        depth = output["depth"].detach().cpu().numpy().astype(np.float32)
        mask_tensor = output.get("mask")
        if mask_tensor is not None:
            mask = mask_tensor.detach().cpu().numpy() > 0.5
            depth = np.where(mask, depth, 0.0)
        if target_height is not None and target_width is not None:
            depth = _resize_map(depth, target_height, target_width)
        np.save(depth_path, depth)
        processed.append(frame_name)
        writes += 1

    return writes, processed


def run(cfg: RawMoGEDepthConfig) -> None:
    image_root = cfg.image_root.resolve()
    output_root = cfg.output_root.resolve()
    raw_root = cfg.raw_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seq_key = _resolve_sequence_key(cfg.seq_name)
    target = _resolve_target_name(cfg.seq_name, cfg.target_name)
    calibrations, calibration_path = _load_calibrations(raw_root, seq_key)

    camera_ids = _ordered_camera_ids(cfg.camera_ids)

    device = torch.device(cfg.device)
    model = MoGeModel.from_pretrained(cfg.model_name).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    for cam in camera_ids:
        cam_dir = _find_camera_dir(image_root, cfg.camera_dir_template, target, cam)
        cam_uid = _camera_uid_from_dirname(cam_dir.name)
        calibration = calibrations.get(cam_uid)
        if calibration is None:
            print(f"[MoGE] Warning: calibration for camera '{cam_uid}' not found. Using implicit FoV.")
            fov_x = None
        else:
            try:
                fov_x = _horizontal_fov(calibration)
            except ValueError:
                print(
                    f"[MoGE] Warning: invalid calibration for camera '{cam_uid}'. Using implicit FoV."
                )
                fov_x = None

        out_dir = output_root / cam_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing_metadata = _load_existing_metadata(out_dir / "metadata.json")
        written_count, frames = _infer_camera(
            model=model,
            device=device,
            img_dir=cam_dir,
            out_dir=out_dir,
            frame_step=cfg.frame_step,
            max_frames=cfg.max_frames,
            target_height=cfg.target_height,
            target_width=cfg.target_width,
            overwrite=cfg.overwrite,
            fov_x=fov_x,
        )

        existing_frames: set[str] = set(existing_metadata.get("frames", [])) if existing_metadata else set()
        existing_frames.update(frames)

        metadata = {
            "sequence": cfg.seq_name,
            "sequence_key": seq_key,
            "target": target,
            "camera": cam_dir.name,
            "camera_uid": cam_uid,
            "model_name": cfg.model_name,
            "device": cfg.device,
            "frame_step": cfg.frame_step,
            "max_frames": cfg.max_frames,
            "target_height": cfg.target_height,
            "target_width": cfg.target_width,
            "overwrite": cfg.overwrite,
            "requested_frame_count": len(frames),
            "selected_frame_count": len(existing_frames),
            "new_writes": written_count,
            "fov_x_deg": fov_x,
            "calibration_file": str(calibration_path),
            "frames": sorted(existing_frames),
        }
        _save_metadata(out_dir / "metadata.json", metadata)

        fov_msg = f", FoV {fov_x:.2f}°" if fov_x is not None else ""
        print(
            f"[MoGE] camera {cam_dir.name}: wrote {written_count} depth maps to {out_dir / 'depth'}{fov_msg}"
        )


if __name__ == "__main__":
    run(tyro.cli(RawMoGEDepthConfig))
