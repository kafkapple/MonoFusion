"""Convert m5t2 mouse data format to MonoFusion's expected format.

Creates:
  1. gopro_calibs.csv from Dy_train_meta.json (calibration)
  2. timestep.txt (start frame index)
  3. dyn_mask_XXXXX.npz symlinks (mask naming convention)
  4. XXXXX.png symlinks for images (5-digit from 6-digit)

This is a pure data format conversion — no pipeline logic modified.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Invert 4x4 world-to-camera matrix to camera-to-world."""
    c2w = np.eye(4)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def rotation_to_quat(rot_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    r = Rotation.from_matrix(rot_matrix)
    # scipy returns (x, y, z, w) by default
    qx, qy, qz, qw = r.as_quat()
    return qw, qx, qy, qz


def create_gopro_calibs_csv(meta_path: Path, output_path: Path) -> None:
    """Create gopro_calibs.csv from Dy_train_meta.json."""
    with open(meta_path) as f:
        meta = json.load(f)

    hw = np.array(meta["hw"])       # (4, 2)
    k = np.array(meta["k"])         # (60, 4, 3, 3)
    w2c = np.array(meta["w2c"])     # (60, 4, 4, 4)

    num_cams = hw.shape[0]

    # Use first frame for static camera poses
    rows = []
    for cam_idx in range(num_cams):
        c2w = w2c_to_c2w(w2c[0, cam_idx])
        tx, ty, tz = c2w[:3, 3]
        qw, qx, qy, qz = rotation_to_quat(c2w[:3, :3])

        K = k[0, cam_idx]  # first frame intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = int(hw[cam_idx, 0]), int(hw[cam_idx, 1])

        rows.append({
            "cam_uid": f"cam{cam_idx:02d}",
            "tx_world_cam": tx,
            "ty_world_cam": ty,
            "tz_world_cam": tz,
            "qw_world_cam": qw,
            "qx_world_cam": qx,
            "qy_world_cam": qy,
            "qz_world_cam": qz,
            "image_width": w,
            "image_height": h,
            "intrinsics_0": fx,
            "intrinsics_1": fy,
            "intrinsics_2": cx,
            "intrinsics_3": cy,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[prepare] Created {output_path} ({len(rows)} cameras)")


def create_timestep_txt(output_path: Path, start_index: int = 0) -> None:
    """Create timestep.txt with start frame index."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(start_index))
    print(f"[prepare] Created {output_path} (start_index={start_index})")


def create_mask_symlinks(mask_dir: Path) -> None:
    """Create dyn_mask_XXXXX.npz symlinks pointing to XXXXXX.npz files.

    DUSt3R expects: dyn_mask_{frame_idx:05d}.npz
    Our format:     {frame_idx:06d}.npz
    """
    npz_files = sorted(mask_dir.glob("??????.npz"))
    if not npz_files:
        print(f"[prepare] No 6-digit .npz files in {mask_dir}, skipping")
        return

    created = 0
    for npz_path in npz_files:
        frame_num = int(npz_path.stem)
        link_name = f"dyn_mask_{frame_num:05d}.npz"
        link_path = mask_dir / link_name
        if link_path.exists():
            continue
        os.symlink(npz_path.name, str(link_path))
        created += 1

    print(f"[prepare] Created {created} mask symlinks in {mask_dir}")


def create_image_symlinks(img_dir: Path) -> None:
    """Create 5-digit symlinks pointing to 6-digit image files.

    DUSt3R expects: {frame_idx:05d}.png
    Our format:     {frame_idx:06d}.png
    """
    img_files = sorted(img_dir.glob("??????.png"))
    if not img_files:
        img_files = sorted(img_dir.glob("??????.jpg"))
    if not img_files:
        print(f"[prepare] No 6-digit image files in {img_dir}, skipping")
        return

    created = 0
    for img_path in img_files:
        frame_num = int(img_path.stem)
        ext = img_path.suffix
        link_name = f"{frame_num:05d}{ext}"
        link_path = img_dir / link_name
        if link_path.exists():
            continue
        os.symlink(img_path.name, str(link_path))
        created += 1

    print(f"[prepare] Created {created} image symlinks in {img_dir}")


def create_npy_5digit_symlinks(npy_dir: Path) -> None:
    """Create 5-digit .npy symlinks from 6-digit originals."""
    npy_files = sorted(npy_dir.glob("??????.npy"))
    if not npy_files:
        return
    created = 0
    for npy_path in npy_files:
        frame_num = int(npy_path.stem)
        link_name = f"{frame_num:05d}.npy"
        link_path = npy_dir / link_name
        if link_path.exists():
            continue
        os.symlink(npy_path.name, str(link_path))
        created += 1
    if created:
        print(f"[prepare] Created {created} 5-digit .npy symlinks in {npy_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert m5t2 data format for MonoFusion pipeline")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root data directory (e.g. /node_data/joon/data/monofusion/m5t2_poc)")
    parser.add_argument("--seq_name", type=str, default="m5t2",
                        help="Sequence name")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    seq = args.seq_name
    raw_root = data_root / "_raw_data"

    # 1. Create gopro_calibs.csv
    meta_path = raw_root / seq / "trajectory" / "Dy_train_meta.json"
    csv_path = raw_root / seq / "trajectory" / "gopro_calibs.csv"
    if meta_path.exists():
        create_gopro_calibs_csv(meta_path, csv_path)
    else:
        print(f"[prepare] WARNING: {meta_path} not found, skipping CSV creation")

    # 2. Create timestep.txt
    timestep_dir = raw_root / seq / "frame_aligned_videos"
    create_timestep_txt(timestep_dir / "timestep.txt", start_index=0)

    # 3. Create image symlinks (6-digit → 5-digit) for all camera dirs
    image_root = data_root / "images"
    if image_root.exists():
        for cam_dir in sorted(image_root.iterdir()):
            if cam_dir.is_dir():
                create_image_symlinks(cam_dir)

    # 4. Create mask symlinks for all camera dirs
    mask_root = data_root / "masks"
    if mask_root.exists():
        for cam_dir in sorted(mask_root.iterdir()):
            if cam_dir.is_dir():
                create_mask_symlinks(cam_dir)

    # 5. Create undist_cam directory symlinks (DUSt3R expects {seq}_undist_cam*)
    symlink_dirs = [data_root / "images", data_root / "masks",
                    data_root / "raw_moge_depth"]
    for root_dir in symlink_dirs:
        if not root_dir.exists():
            continue
        for cam_dir in sorted(root_dir.iterdir()):
            if not cam_dir.is_dir() or cam_dir.is_symlink():
                continue
            # m5t2_cam00 → m5t2_undist_cam00
            if f"{seq}_cam" in cam_dir.name and "_undist_" not in cam_dir.name:
                undist_name = cam_dir.name.replace(f"{seq}_cam", f"{seq}_undist_cam")
                undist_link = root_dir / undist_name
                if not undist_link.exists():
                    os.symlink(cam_dir.name, str(undist_link))
                    print(f"[prepare] Created symlink {undist_link} -> {cam_dir.name}")

    # 5b. Create 5-digit .npy symlinks in MoGe depth dirs (DUSt3R uses 5-digit names)
    moge_root = data_root / "raw_moge_depth"
    if moge_root.exists():
        for cam_dir in sorted(moge_root.iterdir()):
            if not cam_dir.is_dir() or cam_dir.is_symlink():
                continue
            depth_dir = cam_dir / "depth"
            if depth_dir.exists():
                create_npy_5digit_symlinks(depth_dir)

    # 6. Create sam_v2_dyn_mask symlink if not exists
    sam_link = data_root / "sam_v2_dyn_mask"
    if not sam_link.exists() and mask_root.exists():
        os.symlink("masks", str(sam_link))
        print(f"[prepare] Created symlink {sam_link} -> masks")

    print("[prepare] Done! Data format conversion complete.")


if __name__ == "__main__":
    main()
