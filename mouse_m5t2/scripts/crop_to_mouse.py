"""
Object-centric crop: extract mouse bbox from masks, crop images + adjust K.

Creates a new dataset where mouse fills ~40-60% of frame with correct per-camera K.

Usage:
    python crop_to_mouse.py \
        --data_root /node_data/joon/data/monofusion/markerless_v7 \
        --output_dir /node_data/joon/data/monofusion/markerless_v7_crop \
        --target_size 512 --padding_factor 1.5
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mouse_bbox_from_mask(mask_path):
    """Extract mouse bounding box from mask file (.npz with dyn_mask)."""
    data = np.load(mask_path)
    mask = data["dyn_mask"] if "dyn_mask" in data else data[list(data.keys())[0]]
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None
    ys, xs = np.where(fg > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_global_bbox(data_root, cam_names):
    """Compute a single global bbox per camera across all frames."""
    cam_bboxes = {}
    for cam in cam_names:
        mask_dir = data_root / "masks" / cam
        if not mask_dir.exists():
            continue
        all_x1, all_y1, all_x2, all_y2 = [], [], [], []
        for mask_file in sorted(mask_dir.glob("*.npz")):
            bbox = get_mouse_bbox_from_mask(mask_file)
            if bbox:
                all_x1.append(bbox[0])
                all_y1.append(bbox[1])
                all_x2.append(bbox[2])
                all_y2.append(bbox[3])
        if all_x1:
            # Union of all frame bboxes (conservative)
            cam_bboxes[cam] = [
                min(all_x1), min(all_y1), max(all_x2), max(all_y2)
            ]
            print("  %s: bbox=[%d,%d,%d,%d] size=%dx%d" % (
                cam, cam_bboxes[cam][0], cam_bboxes[cam][1],
                cam_bboxes[cam][2], cam_bboxes[cam][3],
                cam_bboxes[cam][2] - cam_bboxes[cam][0],
                cam_bboxes[cam][3] - cam_bboxes[cam][1]))
    return cam_bboxes


def compute_crop_params(bbox, img_w, img_h, target_size, padding_factor):
    """Compute square crop parameters from bbox with padding."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    half_w = (bbox[2] - bbox[0]) / 2 * padding_factor
    half_h = (bbox[3] - bbox[1]) / 2 * padding_factor
    half_side = max(half_w, half_h)

    x1 = max(0, int(cx - half_side))
    y1 = max(0, int(cy - half_side))
    x2 = min(img_w, int(cx + half_side))
    y2 = min(img_h, int(cy + half_side))

    # Ensure square
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)

    # Re-center
    cx_int = (x1 + x2) // 2
    cy_int = (y1 + y2) // 2
    x1 = max(0, cx_int - side // 2)
    y1 = max(0, cy_int - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)

    # Adjust if clamped
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)

    scale_x = target_size / (x2 - x1)
    scale_y = target_size / (y2 - y1)

    return x1, y1, x2, y2, scale_x, scale_y


def adjust_intrinsics(K, crop_x1, crop_y1, scale_x, scale_y):
    """Adjust K for crop + resize."""
    K_new = K.copy()
    K_new[0, 0] *= scale_x           # fx
    K_new[1, 1] *= scale_y           # fy
    K_new[0, 2] = (K[0, 2] - crop_x1) * scale_x  # cx
    K_new[1, 2] = (K[1, 2] - crop_y1) * scale_y  # cy
    return K_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--padding_factor", type=float, default=1.5)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    target = args.target_size

    # Detect cameras
    img_root = data_root / "images"
    cam_names = sorted([d.name for d in img_root.iterdir()
                        if d.is_dir() and "cam" in d.name and not d.is_symlink()])
    print("Cameras:", cam_names)

    # Step 1: Compute global bbox per camera
    print("\n[1] Computing mouse bboxes from masks...")
    cam_bboxes = compute_global_bbox(data_root, cam_names)

    # Get image dimensions
    sample_img = Image.open(next((img_root / cam_names[0]).glob("*.png")))
    img_w, img_h = sample_img.size
    print("  Image size: %dx%d" % (img_w, img_h))

    # Step 2: Compute crop params per camera
    print("\n[2] Computing crop parameters...")
    crop_params = {}
    for cam in cam_names:
        if cam not in cam_bboxes:
            continue
        x1, y1, x2, y2, sx, sy = compute_crop_params(
            cam_bboxes[cam], img_w, img_h, target, args.padding_factor)
        crop_params[cam] = (x1, y1, x2, y2, sx, sy)
        print("  %s: crop=[%d,%d,%d,%d] (%dx%d) → %dx%d, scale=%.2f" % (
            cam, x1, y1, x2, y2, x2 - x1, y2 - y1, target, target, sx))

    # Step 3: Crop images
    print("\n[3] Cropping images...")
    for cam in cam_names:
        if cam not in crop_params:
            continue
        x1, y1, x2, y2, sx, sy = crop_params[cam]
        src_dir = img_root / cam
        dst_dir = output_dir / "images" / cam
        dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(sorted(src_dir.glob("*.png")), desc=cam, leave=False):
            img = cv2.imread(str(img_path))
            crop = img[y1:y2, x1:x2]
            resized = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst_dir / img_path.name), resized)

    # Step 4: Crop masks
    print("\n[4] Cropping masks...")
    for cam in cam_names:
        if cam not in crop_params:
            continue
        x1, y1, x2, y2, sx, sy = crop_params[cam]
        src_dir = data_root / "masks" / cam
        dst_dir = output_dir / "masks" / cam
        dst_dir.mkdir(parents=True, exist_ok=True)

        for mask_path in tqdm(sorted(src_dir.glob("*.npz")), desc=cam, leave=False):
            data = np.load(mask_path)
            mask = data["dyn_mask"] if "dyn_mask" in data else data[list(data.keys())[0]]
            crop = mask[y1:y2, x1:x2]
            resized = cv2.resize(crop, (target, target), interpolation=cv2.INTER_NEAREST)
            np.savez_compressed(str(dst_dir / mask_path.name), dyn_mask=resized)

    # Step 5: Adjust camera meta
    print("\n[5] Adjusting camera intrinsics...")
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    meta = json.load(open(meta_path))

    n_frames = len(meta["k"])
    n_cams = len(meta["hw"])

    new_hw = []
    new_k = [[None] * n_cams for _ in range(n_frames)]
    new_w2c = meta["w2c"]  # w2c unchanged (world coordinates unchanged)

    for ci, cam in enumerate(cam_names[:n_cams]):
        if cam not in crop_params:
            continue
        x1, y1, x2, y2, sx, sy = crop_params[cam]
        new_hw.append([target, target])

        for t in range(n_frames):
            K_orig = np.array(meta["k"][t][ci])
            K_new = adjust_intrinsics(K_orig, x1, y1, sx, sy)
            new_k[t][ci] = K_new.tolist()
            if t == 0:
                print("  %s: fx=%.1f→%.1f cx=%.1f→%.1f cy=%.1f→%.1f" % (
                    cam, K_orig[0, 0], K_new[0, 0],
                    K_orig[0, 2], K_new[0, 2],
                    K_orig[1, 2], K_new[1, 2]))

    new_meta = {
        "hw": new_hw,
        "k": new_k,
        "w2c": new_w2c,
        "camera_convention": meta.get("camera_convention", "w2c"),
    }

    out_meta_dir = output_dir / "_raw_data" / "m5t2" / "trajectory"
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(out_meta_dir / "Dy_train_meta.json", "w") as f:
        json.dump(new_meta, f)

    # Step 6: Save conversion info
    info = {
        "source": str(data_root),
        "crop_method": "object-centric from mask bbox",
        "padding_factor": args.padding_factor,
        "target_size": target,
        "camera_count": n_cams,
        "frame_count": n_frames,
        "crop_params": {cam: list(crop_params[cam][:4]) for cam in crop_params},
    }
    with open(output_dir / "conversion_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n[6] Creating symlinks for m5t2_cam* compatibility...")
    for subdir in ["images", "masks"]:
        base = output_dir / subdir
        for ci, cam in enumerate(cam_names[:n_cams]):
            link = base / ("m5t2_cam%02d" % ci)
            if not link.exists():
                link.symlink_to(cam)

    print("\nDone! Output: %s" % output_dir)
    print("\nNext steps:")
    print("  1. DINOv2:  run_dinov2.py --data_root %s" % output_dir)
    print("  2. Depth:   run_moge_depth.py --data_root %s" % output_dir)
    print("  3. RAFT:    generate_raft_tracks.py --data_root %s ..." % output_dir)
    print("  4. PCA32:   (inline script)")
    print("  5. Train:   train_m5t2.py --data_root %s" % output_dir)


if __name__ == "__main__":
    main()
