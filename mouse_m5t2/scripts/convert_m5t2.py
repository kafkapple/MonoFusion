"""
Convert M5t2 FaceLift dataset to MonoFusion format.

Input:  /node_data/joon/data/preprocessed/FaceLift_mouse/M5/
        Per-frame dirs: {frame_id}/images/cam_XXX.png + opencv_cameras.json

Output: /node_data/joon/data/monofusion/m5t2/
        Dy_train_meta.json, images/, masks/ in MonoFusion casual_dataset format.

Usage:
    python convert_m5t2.py --src /node_data/joon/data/preprocessed/FaceLift_mouse/M5 \
                           --dst /node_data/joon/data/monofusion/m5t2 \
                           --num_cams 4 \
                           --frame_step 3 \
                           --max_frames 0
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_opencv_cameras(json_path: Path) -> dict:
    """Load opencv_cameras.json and extract per-camera parameters."""
    with open(json_path) as f:
        data = json.load(f)
    cameras = {}
    for entry in data["frames"]:
        view_id = entry["view_id"]
        w2c = np.array(entry["w2c"])  # 4x4
        K = np.array([
            [entry["fx"], 0, entry["cx"]],
            [0, entry["fy"], entry["cy"]],
            [0, 0, 1]
        ])
        cameras[view_id] = {
            "w2c": w2c,
            "K": K,
            "w": entry["w"],
            "h": entry["h"],
        }
    return cameras


def select_cameras_by_spread(cameras: dict, n_select: int = 4) -> list:
    """Select cameras maximizing angular spread around subject."""
    cam_ids = sorted(cameras.keys())
    if len(cam_ids) <= n_select:
        return cam_ids

    # Extract camera positions from w2c
    positions = {}
    for cid in cam_ids:
        w2c = cameras[cid]["w2c"]
        c2w = np.linalg.inv(w2c)
        positions[cid] = c2w[:3, 3]

    center = np.mean(list(positions.values()), axis=0)
    directions = {k: (v - center) / (np.linalg.norm(v - center) + 1e-8)
                  for k, v in positions.items()}

    # Greedy max-angle selection
    selected = [cam_ids[0]]
    remaining = set(cam_ids[1:])
    while len(selected) < n_select:
        best_cam = None
        best_min_angle = -1
        for cid in remaining:
            min_angle = min(
                np.arccos(np.clip(np.dot(directions[cid], directions[s]), -1, 1))
                for s in selected
            )
            if min_angle > best_min_angle:
                best_min_angle = min_angle
                best_cam = cid
        selected.append(best_cam)
        remaining.discard(best_cam)

    return sorted(selected)


def verify_reprojection(cameras: dict, selected_cams: list):
    """Sanity check: project scene center into each camera."""
    # Estimate scene center from camera positions
    positions = []
    for cid in selected_cams:
        c2w = np.linalg.inv(cameras[cid]["w2c"])
        positions.append(c2w[:3, 3])
    scene_center = np.mean(positions, axis=0)

    print(f"\nReprojection sanity check (scene center = {scene_center.round(3)}):")
    for cid in selected_cams:
        cam = cameras[cid]
        w2c = cam["w2c"]
        K = cam["K"]
        pt_cam = w2c[:3, :3] @ scene_center + w2c[:3, 3]
        if pt_cam[2] <= 0:
            print(f"  cam_{cid:03d}: BEHIND CAMERA (z={pt_cam[2]:.3f}) -- convention error!")
            return False
        px = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
        py = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
        w, h = cam["w"], cam["h"]
        in_frame = 0 <= px <= w and 0 <= py <= h
        status = "OK" if in_frame else "OUT OF FRAME"
        print(f"  cam_{cid:03d}: ({px:.1f}, {py:.1f}) [{status}]")
    return True


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    num_cams: int = 4,
    frame_step: int = 1,
    max_frames: int = 0,
):
    """Convert M5t2 to MonoFusion format."""
    # Discover frames
    frame_dirs = sorted([
        d for d in src_root.iterdir()
        if d.is_dir() or d.is_symlink()
    ])
    # Filter to directories that have images/
    frame_dirs = [d for d in frame_dirs if (d / "images").exists()]

    if frame_step > 1:
        frame_dirs = frame_dirs[::frame_step]
    if max_frames > 0:
        frame_dirs = frame_dirs[:max_frames]

    print(f"Source: {src_root}")
    print(f"Frames: {len(frame_dirs)} (step={frame_step})")

    # Load cameras from first frame to select views
    cam_data_0 = load_opencv_cameras(frame_dirs[0] / "opencv_cameras.json")
    selected_cams = select_cameras_by_spread(cam_data_0, num_cams)
    all_cams = sorted(cam_data_0.keys())

    print(f"Available cameras: {all_cams}")
    print(f"Selected cameras: {selected_cams}")

    # Verify reprojection
    if not verify_reprojection(cam_data_0, selected_cams):
        print("WARNING: Reprojection check failed! Camera convention may be wrong.")

    # Create output structure
    seq_name = "m5t2"
    (dst_root / "images" / seq_name).mkdir(parents=True, exist_ok=True)
    (dst_root / "masks" / seq_name).mkdir(parents=True, exist_ok=True)

    # Build Dy_train_meta.json
    # MonoFusion format: {hw: [...], k: [per_time][per_cam], w2c: [per_time][per_cam]}
    meta_hw = []
    meta_k = []
    meta_w2c = []

    # hw: one entry per selected camera
    sample_cam = cam_data_0[selected_cams[0]]
    for cid in selected_cams:
        meta_hw.append([cam_data_0[cid]["h"], cam_data_0[cid]["w"]])

    for fi, frame_dir in enumerate(tqdm(frame_dirs, desc="Converting")):
        frame_name = f"{fi:06d}"
        cam_data = load_opencv_cameras(frame_dir / "opencv_cameras.json")

        k_per_cam = []
        w2c_per_cam = []

        for ci, cid in enumerate(selected_cams):
            cam = cam_data[cid]

            # Save RGB image (remove alpha)
            src_img_path = frame_dir / "images" / f"cam_{cid:03d}.png"
            img = Image.open(src_img_path)
            if img.mode == "RGBA":
                rgb = img.convert("RGB")
                alpha = np.array(img)[:, :, 3]
            else:
                rgb = img
                alpha = np.ones((cam["h"], cam["w"]), dtype=np.uint8) * 255

            # Save RGB: images/{seq_name}/{frame_name}.jpg
            # MonoFusion loads per-camera per-frame, but dataset iterates per-camera
            # The seq_name encodes which camera: e.g. "m5t2_cam00"
            cam_seq = f"{seq_name}_cam{ci:02d}"
            img_dir = dst_root / "images" / cam_seq
            mask_dir = dst_root / "masks" / cam_seq
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            rgb.save(img_dir / f"{frame_name}.png")

            # Save mask as npz: 1=FG, -1=BG, 0=border
            fg_mask = (alpha > 128).astype(np.float32)
            # Erode slightly for border region
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(fg_mask, iterations=2).astype(np.float32)
            mask_out = np.full_like(fg_mask, -1.0)  # default BG
            mask_out[eroded > 0] = 1.0  # FG
            mask_out[(fg_mask > 0) & (eroded == 0)] = 0.0  # border

            np.savez_compressed(
                mask_dir / f"{frame_name}.npz",
                dyn_mask=mask_out
            )

            # Collect camera params
            k_per_cam.append(cam["K"].tolist())
            w2c_per_cam.append(cam["w2c"].tolist())

        meta_k.append(k_per_cam)
        meta_w2c.append(w2c_per_cam)

    # Save Dy_train_meta.json
    # MonoFusion loads: md['w2c'][t][c] where t=time, c=camera_index
    raw_dir = dst_root / "_raw_data" / seq_name / "trajectory"
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "hw": meta_hw,
        "k": meta_k,
        "w2c": meta_w2c,
    }
    with open(raw_dir / "Dy_train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save conversion info
    info = {
        "source": str(src_root),
        "selected_cameras": selected_cams,
        "all_cameras": all_cams,
        "num_frames": len(frame_dirs),
        "frame_step": frame_step,
        "resolution": [sample_cam["h"], sample_cam["w"]],
        "camera_count": len(selected_cams),
    }
    with open(dst_root / "conversion_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Output: {dst_root}")
    print(f"  Frames: {len(frame_dirs)}")
    print(f"  Cameras: {len(selected_cams)} ({selected_cams})")
    print(f"  Resolution: {sample_cam['h']}x{sample_cam['w']}")
    print(f"  Camera sequences: {[f'{seq_name}_cam{i:02d}' for i in range(len(selected_cams))]}")
    print(f"  Meta: {raw_dir / 'Dy_train_meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert M5t2 to MonoFusion format")
    parser.add_argument("--src", type=str, required=True, help="M5t2 source directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    parser.add_argument("--num_cams", type=int, default=4, help="Number of cameras to select")
    parser.add_argument("--frame_step", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames (0=all)")
    args = parser.parse_args()

    convert_dataset(
        src_root=Path(args.src),
        dst_root=Path(args.dst),
        num_cams=args.num_cams,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
