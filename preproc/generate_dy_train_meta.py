from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tyro


@dataclass
class MetaConfig:
    raw_root: Path = Path("_raw_data")
    """Root directory containing the raw sequence folders."""

    camera_folder: str = "undist_processed_frames"
    """Sub-directory under each sequence containing per-camera frames."""

    output_path: Path | None = None
    """Optional explicit output path for Dy_train_meta.json."""

    include_ego: bool = True
    ego_hw: tuple[int, int] = (1408, 1408)
    ego_intrinsics: tuple[float, float, float, float] = (670.0, 670.0, 703.5, 703.5)
    ego_prefix: str = "0"
    ego_padding: int = 6

    camera_prefix: str = "undist_data"
    camera_padding: int = 5
    subtract_half_pixel: bool = True


def _numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return (int(stem), stem)
    except ValueError:
        return (10**9, stem)


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero length")
    q /= norm
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _make_extrinsic(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = np.asarray(list(translation), dtype=np.float64)
    return extrinsic


def generate_metadata(seq: str, config: MetaConfig = MetaConfig()) -> Path:
    raw_root = config.raw_root.expanduser().resolve()
    seq_dir = raw_root / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence '{seq}' not found under {raw_root}")

    frames_root = seq_dir / config.camera_folder
    if not frames_root.exists():
        raise FileNotFoundError(
            f"Sequence '{seq}' is missing '{config.camera_folder}' under {seq_dir}"
        )

    camera_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    if not camera_dirs:
        raise RuntimeError(f"No camera folders found under {frames_root}")

    # Use the first camera to determine frame count
    primary_frames = sorted([p for p in camera_dirs[0].iterdir() if p.is_file()], key=_numeric_key)
    frame_count = len(primary_frames)
    if frame_count == 0:
        raise RuntimeError(f"Camera directory {camera_dirs[0]} does not contain images")

    traj_dir = seq_dir / "trajectory"
    calib_path = traj_dir / "gopro_calibs.csv"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {calib_path}")

    with calib_path.open() as f:
        reader = csv.DictReader(f)
        calib_rows = [row for row in reader]

    if len(calib_rows) != len(camera_dirs):
        raise ValueError(
            f"Calibration count ({len(calib_rows)}) does not match camera directories ({len(camera_dirs)})"
        )

    # Sort calibrations to match camera order (cam01, cam02, ...)
    calib_rows.sort(key=lambda row: row["cam_uid"])  # cam_uid is cam01, cam02, ...

    intrinsics_list: list[list[list[float]]] = []
    hw_list: list[list[int]] = []
    extrinsics_list: list[np.ndarray] = []

    if config.include_ego:
        fx, fy, cx, cy = config.ego_intrinsics
        intrinsics_list.append(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ]
        )
        hw_list.append([int(config.ego_hw[0]), int(config.ego_hw[1])])

    for row in calib_rows:
        width = int(float(row["image_width"]))
        height = int(float(row["image_height"]))

        fx = float(row["intrinsics_0"])
        fy = float(row["intrinsics_1"])
        cx = float(row["intrinsics_2"])
        cy = float(row["intrinsics_3"])

        if config.subtract_half_pixel:
            cx -= 0.5
            cy -= 0.5

        intrinsics_list.append(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )
        hw_list.append([height, width])

        rotation = _quat_to_matrix(
            float(row["qx_world_cam"]),
            float(row["qy_world_cam"]),
            float(row["qz_world_cam"]),
            float(row["qw_world_cam"]),
        )
        translation = (
            float(row["tx_world_cam"]),
            float(row["ty_world_cam"]),
            float(row["tz_world_cam"]),
        )
        extrinsics_list.append(_make_extrinsic(rotation, translation))

    if config.include_ego:
        # Duplicate the first camera extrinsic for the ego slot by convention.
        extrinsics = [extrinsics_list[0]] + extrinsics_list
    else:
        extrinsics = extrinsics_list

    if len(intrinsics_list) != len(extrinsics):
        raise AssertionError("Mismatch between intrinsics and extrinsics counts")

    camera_count = len(intrinsics_list)

    # Prepare filename entries
    frame_entries: list[list[str]] = []
    for idx in range(frame_count):
        entry: list[str] = []
        if config.include_ego:
            entry.append(f"{config.ego_prefix}/{idx:0{config.ego_padding}d}.jpg")

        for cam_dir in camera_dirs:
            entry.append(
                f"{config.camera_prefix}/{cam_dir.name}/{idx:0{config.camera_padding}d}.jpg"
            )
        frame_entries.append(entry)

    cam_ids = [[i for i in range(camera_count)] for _ in range(frame_count)]

    # Broadcast intrinsics and extrinsics per frame
    k_multi = [
        [[row[:] for row in mat] for mat in intrinsics_list]
        for _ in range(frame_count)
    ]
    w2c_multi = [[mat.tolist() for mat in extrinsics] for _ in range(frame_count)]

    json_dict = {
        "hw": hw_list,
        "k": k_multi,
        "w2c": w2c_multi,
        "fn": frame_entries,
        "cam_id": cam_ids,
    }

    if config.output_path is None:
        output_path = traj_dir / "Dy_train_meta.json"
    else:
        output_path = config.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(json_dict, f)

    print(f"Wrote metadata to {output_path}")
    return output_path


def main(seq: str, config: MetaConfig = MetaConfig()) -> None:
    generate_metadata(seq, config)


if __name__ == "__main__":
    tyro.cli(main)
