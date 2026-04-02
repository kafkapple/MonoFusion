"""
Generate consistent dense depth maps using COLMAP MVS with GT camera poses.

Approach B for depth alignment: replace per-view MoGe relative depth
with multi-view stereo depth that is geometrically consistent across cameras.

Usage (on gpu03):
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate monofusion
    CUDA_VISIBLE_DEVICES=6 python ~/dev/MonoFusion/mouse_m5t2/scripts/run_colmap_mvs.py \
        --data_root /node_data/joon/data/monofusion/m5t2_v5 \
        --output_dir /node_data/joon/data/monofusion/m5t2_v5/colmap_depth
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz] (COLMAP convention)."""
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # [qx, qy, qz, qw] scipy convention
    return [q[3], q[0], q[1], q[2]]  # COLMAP: [qw, qx, qy, qz]


def write_colmap_cameras(cameras_txt, fx, fy, cx, cy, w, h, n_cameras):
    """Write COLMAP cameras.txt with PINHOLE model."""
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {n_cameras}\n")
        for cam_id in range(1, n_cameras + 1):
            f.write(f"{cam_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")


def write_colmap_images(images_txt, image_entries):
    """Write COLMAP images.txt.

    Each image needs 2 lines:
    Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    Line 2: (empty — no 2D points for triangulation input)
    """
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_entries)}\n")
        for entry in image_entries:
            img_id = entry["image_id"]
            qw, qx, qy, qz = entry["quat"]
            tx, ty, tz = entry["translation"]
            cam_id = entry["camera_id"]
            name = entry["name"]
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n")
            f.write("\n")  # empty 2D points line


def write_colmap_points3d(points3d_txt):
    """Write empty points3D.txt (COLMAP will triangulate)."""
    with open(points3d_txt, "w") as f:
        f.write("# 3D point list (empty — COLMAP will triangulate)\n")


def setup_colmap_workspace(data_root, output_dir):
    """Convert GT cameras to COLMAP format and setup workspace."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # Load GT camera parameters
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    meta = json.load(open(meta_path))

    n_frames = len(meta["k"])
    n_cameras = len(meta["hw"])
    h, w = meta["hw"][0]
    K = np.array(meta["k"][0][0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print(f"Dataset: {n_frames} frames, {n_cameras} cameras, {w}x{h}")
    print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Setup COLMAP directories
    colmap_dir = output_dir / "colmap"
    sparse_dir = colmap_dir / "sparse" / "0"
    image_dir = colmap_dir / "images"
    dense_dir = colmap_dir / "dense"

    for d in [sparse_dir, image_dir, dense_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images to flat structure: cam{C}_frame{T}.png
    cam_dirs = sorted((data_root / "images").iterdir())
    cam_names = [d.name for d in cam_dirs if d.is_dir() and "cam" in d.name]
    print(f"Cameras: {cam_names}")

    image_entries = []
    img_id = 1

    for ci, cam_name in enumerate(cam_names):
        cam_dir = data_root / "images" / cam_name
        frames = sorted(cam_dir.glob("*.png"))

        for fi, frame_path in enumerate(frames):
            if fi >= n_frames:
                break

            # Flat image name for COLMAP
            flat_name = f"{cam_name}_{frame_path.stem}.png"
            dst = image_dir / flat_name
            if not dst.exists():
                dst.symlink_to(frame_path)

            # w2c matrix for this camera and frame
            w2c = np.array(meta["w2c"][fi][ci], dtype=np.float64)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            quat = rotation_matrix_to_quaternion(R)

            image_entries.append({
                "image_id": img_id,
                "quat": quat,
                "translation": t.tolist(),
                "camera_id": ci + 1,  # COLMAP 1-indexed
                "name": flat_name,
            })
            img_id += 1

    print(f"Total images: {len(image_entries)}")

    # Write COLMAP files
    write_colmap_cameras(sparse_dir / "cameras.txt", fx, fy, cx, cy, w, h, n_cameras)
    write_colmap_images(sparse_dir / "images.txt", image_entries)
    write_colmap_points3d(sparse_dir / "points3D.txt")

    print(f"COLMAP sparse model written to {sparse_dir}")
    return colmap_dir, image_dir, sparse_dir, dense_dir


def run_colmap_pipeline(colmap_dir, image_dir, sparse_dir, dense_dir):
    """Run COLMAP feature extraction, matching, triangulation, and MVS."""
    db_path = colmap_dir / "database.db"

    # Step 1: Feature extraction
    print("\n[1/4] Feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera_per_folder", "0",
        "--SiftExtraction.use_gpu", "1",
    ], check=True)

    # Step 2: Exhaustive matching
    print("\n[2/4] Exhaustive matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], check=True)

    # Step 3: Triangulation with GT poses (not SfM — we trust GT cameras)
    print("\n[3/4] Point triangulation with GT poses...")
    triangulated_dir = colmap_dir / "sparse" / "triangulated"
    triangulated_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "colmap", "point_triangulator",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(triangulated_dir),
    ], check=True)

    # Step 4: Dense MVS (patch_match_stereo)
    print("\n[4/4] Dense stereo (patch_match)...")
    # Undistort images first (required by patch_match)
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(triangulated_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], check=True)

    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.gpu_index", "0",
    ], check=True)

    # Fusion
    fused_path = dense_dir / "fused.ply"
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(fused_path),
    ], check=True)

    print(f"\nFused point cloud: {fused_path}")
    return dense_dir


def extract_depth_maps(dense_dir, data_root, output_dir):
    """Extract per-view depth maps from COLMAP dense output."""
    output_dir = Path(output_dir)
    depth_maps_dir = dense_dir / "stereo" / "depth_maps"

    if not depth_maps_dir.exists():
        print(f"ERROR: {depth_maps_dir} not found")
        return

    # COLMAP outputs .geometric.bin depth maps
    depth_files = sorted(depth_maps_dir.glob("*.geometric.bin"))
    print(f"Found {len(depth_files)} depth maps")

    data_root = Path(data_root)
    cam_dirs = sorted((data_root / "images").iterdir())
    cam_names = [d.name for d in cam_dirs if d.is_dir()]

    for cam_name in cam_names:
        cam_depth_dir = output_dir / cam_name
        cam_depth_dir.mkdir(parents=True, exist_ok=True)

    for depth_file in depth_files:
        # Parse filename: cam_name_frameid.png.geometric.bin
        stem = depth_file.name.replace(".geometric.bin", "")
        # stem = "m5t2_cam00_000000.png"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        cam_name = parts[0]  # m5t2_cam00
        frame_id = parts[1].replace(".png", "")  # 000000

        # Read COLMAP binary depth map
        depth = read_colmap_depth_map(depth_file)
        if depth is None:
            continue

        # Save as .npy (same format as MoGe)
        out_path = output_dir / cam_name / f"{frame_id}.npy"
        np.save(out_path, depth.astype(np.float32))

    print(f"Depth maps saved to {output_dir}")


def read_colmap_depth_map(path):
    """Read COLMAP binary depth map (.geometric.bin)."""
    with open(path, "rb") as f:
        width = int.from_bytes(f.read(4), "little")
        height = int.from_bytes(f.read(4), "little")
        channels = int.from_bytes(f.read(4), "little")
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(height, width) if channels == 1 else data.reshape(height, width, channels)


def main():
    parser = argparse.ArgumentParser(description="COLMAP MVS with GT cameras")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("COLMAP MVS Depth Generation (Approach B)")
    print("=" * 60)

    colmap_dir, image_dir, sparse_dir, dense_dir = setup_colmap_workspace(
        args.data_root, args.output_dir
    )

    run_colmap_pipeline(colmap_dir, image_dir, sparse_dir, dense_dir)
    extract_depth_maps(dense_dir, args.data_root, args.output_dir)

    print("\nDone! Replace aligned_moge_depth with colmap_depth in training config.")


if __name__ == "__main__":
    main()
