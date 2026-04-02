"""
Convert raw markerless mouse data to MonoFusion format.

Input:  /node_data/joon/data/raw/markerless_mouse_1_nerf/
        videos_undist/*.mp4 + camera_params.h5 + simpleclick_undist/*.mp4

Output: /node_data/joon/data/monofusion/markerless_v6/
        images/, masks/, _raw_data/.../Dy_train_meta.json, conversion_info.json

Usage:
    python convert_raw_markerless.py \
        --src /node_data/joon/data/raw/markerless_mouse_1_nerf \
        --dst /node_data/joon/data/monofusion/markerless_v6 \
        --start_frame 1000 --num_frames 80 --stride 5 \
        --num_cams 6
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Frame jump boundaries (from readme.md)
JUMP_FRAMES = [5900, 11800, 17700]
VALID_SEGMENTS = [(0, 5899), (5901, 11799), (11801, 17699), (17701, 17999)]


def load_camera_params_h5(h5_path: str):
    """Load camera parameters from DANNCE h5 format."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        K = np.array(f["camera_parameters"]["intrinsic"])    # (6, 3, 3)
        R = np.array(f["camera_parameters"]["rotation"])     # (6, 3, 3)
        t = np.array(f["camera_parameters"]["translation"])  # (6, 3)
    return K, R, t


def verify_camera_convention(R, t, verbose=True):
    """Verify w2c convention by checking camera positions.

    DANNCE convention: X_cam = R @ X_world + t (w2c).
    Camera center in world = -R^T @ t.
    """
    n_cams = R.shape[0]
    centers = []
    for c in range(n_cams):
        center = -R[c].T @ t[c]
        centers.append(center)
        if verbose:
            print(f"  cam{c}: center={center.round(1)}, |t|={np.linalg.norm(t[c]):.1f}")

    centers = np.array(centers)
    spread = centers.max(axis=0) - centers.min(axis=0)
    if verbose:
        print(f"  Camera spread: {spread.round(1)} (should be ~200-600mm for arena)")
        print(f"  Mean center: {centers.mean(axis=0).round(1)}")

    # Sanity: cameras should surround the origin
    mean_dist = np.linalg.norm(centers, axis=1).mean()
    if verbose:
        print(f"  Mean distance from origin: {mean_dist:.1f}mm")

    return centers


def validate_frame_window(start, num_frames, stride):
    """Ensure selected frames don't cross jump boundaries."""
    end_frame = start + (num_frames - 1) * stride
    for seg_start, seg_end in VALID_SEGMENTS:
        if seg_start <= start and end_frame <= seg_end:
            print(f"  Frame window [{start}:{end_frame}] within segment [{seg_start}:{seg_end}] ✓")
            return True
    print(f"  ERROR: Frame window [{start}:{end_frame}] crosses segment boundary!")
    for jf in JUMP_FRAMES:
        if start <= jf <= end_frame:
            print(f"    Jump at frame {jf} is within window")
    return False


def extract_frames(video_path, frame_ids, output_dir, resize_wh=None):
    """Extract specific frames from video, optionally resize."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i, fid in enumerate(frame_ids):
        if fid >= total:
            print(f"  WARNING: frame {fid} >= total {total}, skipping")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: failed to read frame {fid}")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_wh:
            frame_rgb = cv2.resize(frame_rgb, resize_wh, interpolation=cv2.INTER_AREA)
        # Save as PNG
        out_path = os.path.join(output_dir, f"{i:06d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    return len(frame_ids)


def extract_masks(mask_video_path, frame_ids, output_dir, resize_wh=None,
                  threshold=128, erosion_iters=2):
    """Extract mask frames and convert to MonoFusion dyn_mask format."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(mask_video_path))

    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize_wh:
            gray = cv2.resize(gray, resize_wh, interpolation=cv2.INTER_NEAREST)

        # Binary threshold
        fg = (gray > threshold).astype(np.float32)

        # Erode for border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_eroded = cv2.erode(fg, kernel, iterations=erosion_iters)

        # MonoFusion format: 1=FG, -1=BG, 0=border
        dyn_mask = np.full_like(fg, -1.0)
        dyn_mask[fg_eroded > 0] = 1.0
        dyn_mask[(fg > 0) & (fg_eroded == 0)] = 0.0

        out_path = os.path.join(output_dir, f"{i:06d}.npz")
        np.savez_compressed(out_path, dyn_mask=dyn_mask)

    cap.release()


def build_meta_json(K, R, t, num_frames, num_cams, output_path, resize_wh=None,
                    orig_wh=(1152, 1024)):
    """Build Dy_train_meta.json for MonoFusion."""
    # Scale intrinsics if resized
    K_scaled = K.copy()
    if resize_wh:
        sx = resize_wh[0] / orig_wh[0]
        sy = resize_wh[1] / orig_wh[1]
        for c in range(num_cams):
            K_scaled[c, 0, :] *= sx  # fx, cx
            K_scaled[c, 1, :] *= sy  # fy, cy

    # Build w2c matrices (DANNCE convention: X_cam = R @ X_world + t)
    hw = []
    for c in range(num_cams):
        if resize_wh:
            hw.append([resize_wh[1], resize_wh[0]])  # [H, W]
        else:
            hw.append([orig_wh[1], orig_wh[0]])

    # Static cameras: same for all timesteps
    k_per_time = []
    w2c_per_time = []
    for t_idx in range(num_frames):
        k_per_cam = []
        w2c_per_cam = []
        for c in range(num_cams):
            k_per_cam.append(K_scaled[c].tolist())
            w2c_mat = np.eye(4)
            w2c_mat[:3, :3] = R[c]
            w2c_mat[:3, 3] = t[c]
            w2c_per_cam.append(w2c_mat.tolist())
        k_per_time.append(k_per_cam)
        w2c_per_time.append(w2c_per_cam)

    meta = {"hw": hw, "k": k_per_time, "w2c": w2c_per_time, "camera_convention": "w2c"}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f)
    print(f"  Meta saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw markerless → MonoFusion")
    parser.add_argument("--src", type=str, required=True,
                        help="Raw data root (markerless_mouse_1_nerf)")
    parser.add_argument("--dst", type=str, required=True,
                        help="MonoFusion output root")
    parser.add_argument("--start_frame", type=int, default=1000,
                        help="First frame index (default: 1000, skip stabilization)")
    parser.add_argument("--num_frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame stride (5 = 20fps from 100fps)")
    parser.add_argument("--num_cams", type=int, default=6)
    parser.add_argument("--resize", type=str, default=None,
                        help="Resize to WxH (e.g., '768x768'). None=keep original")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    resize_wh = tuple(int(x) for x in args.resize.split("x")) if args.resize else None

    print("=" * 60)
    print("Raw Markerless → MonoFusion Converter")
    print("=" * 60)

    # 1. Validate frame window
    print("\n[1] Frame selection")
    if not validate_frame_window(args.start_frame, args.num_frames, args.stride):
        sys.exit(1)
    frame_ids = list(range(args.start_frame,
                           args.start_frame + args.num_frames * args.stride,
                           args.stride))
    print(f"  Frames: {len(frame_ids)} ({frame_ids[0]}:{frame_ids[-1]}:{args.stride})")
    print(f"  Effective FPS: {100 / args.stride}")

    # 2. Load and verify cameras
    print("\n[2] Camera parameters")
    h5_path = src / "camera_params.h5"
    K, R, t = load_camera_params_h5(str(h5_path))
    print(f"  Loaded: K={K.shape}, R={R.shape}, t={t.shape}")
    cam_indices = list(range(min(args.num_cams, K.shape[0])))
    K, R, t = K[cam_indices], R[cam_indices], t[cam_indices]
    print(f"  Using cameras: {cam_indices}")
    verify_camera_convention(R, t)

    # Validate intrinsics
    for c in cam_indices:
        assert abs(K[c, 2, 2] - 1.0) < 1e-6, f"K[{c}][2,2] = {K[c,2,2]} (expected 1.0)"

    # Get original resolution from first video
    vid0 = src / "videos_undist" / "0.mp4"
    cap = cv2.VideoCapture(str(vid0))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Original resolution: {orig_w}×{orig_h}")
    if resize_wh:
        print(f"  Resize to: {resize_wh[0]}×{resize_wh[1]}")

    # 3. Extract frames per camera
    print("\n[3] Extracting frames")
    seq_name = "markerless"
    for ci in tqdm(cam_indices, desc="Cameras"):
        cam_seq = f"{seq_name}_cam{ci:02d}"
        vid_path = src / "videos_undist" / f"{ci}.mp4"
        img_dir = dst / "images" / cam_seq
        extract_frames(vid_path, frame_ids, str(img_dir), resize_wh)

    # 4. Extract masks
    print("\n[4] Extracting masks")
    mask_dir_src = src / "simpleclick_undist"
    for ci in tqdm(cam_indices, desc="Masks"):
        cam_seq = f"{seq_name}_cam{ci:02d}"
        mask_vid = mask_dir_src / f"{ci}.mp4"
        mask_dir = dst / "masks" / cam_seq
        extract_masks(str(mask_vid), frame_ids, str(mask_dir), resize_wh)

    # 5. Build camera meta
    print("\n[5] Camera metadata")
    meta_path = dst / "_raw_data" / seq_name / "trajectory" / "Dy_train_meta.json"
    build_meta_json(K, R, t, len(frame_ids), len(cam_indices), str(meta_path),
                    resize_wh, (orig_w, orig_h))

    # 6. Save conversion info
    info = {
        "source": str(src),
        "camera_count": len(cam_indices),
        "camera_indices": cam_indices,
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "original_resolution": [orig_w, orig_h],
        "output_resolution": list(resize_wh) if resize_wh else [orig_w, orig_h],
        "seq_name": seq_name,
        "fps_effective": 100 / args.stride,
    }
    info_path = dst / "conversion_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Output: {dst}")
    print(f"  Frames: {len(frame_ids)} @ {100/args.stride}fps")
    print(f"  Cameras: {len(cam_indices)}")
    print(f"  Resolution: {resize_wh or (orig_w, orig_h)}")
    print(f"\nNext steps:")
    print(f"  1. DINOv2: python run_dinov2.py --data_root {dst}")
    print(f"  2. RAFT:   python generate_raft_tracks.py --data_root {dst} ...")
    print(f"  3. Depth:  python run_moge_depth.py --data_root {dst}")
    print(f"  4. Train:  python train_m5t2.py --data_root {dst} --num_bg 0")


if __name__ == "__main__":
    main()
