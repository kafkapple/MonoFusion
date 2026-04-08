# no-split: Phase 2 — implement MoGe depth alignment via triangulation reference
"""
Phase 2 — Generate aligned MoGe depth using triangulation-based scale.

Replaces the missing compute_aligned_moge_depth.py step with our own approach
that uses known DANNCE metric camera poses (no DUSt3R).

Algorithm per frame:
  1. Compute mask centroids in all valid cams
  2. Triangulate to get metric 3D world point (DLT from N views)
  3. For each cam: expected_depth = z-coordinate of world_point in cam frame
  4. Compute scale[cam] = expected_depth / moge_raw[cam](centroid_pixel)
  5. Apply: aligned_depth[cam][frame] = moge_raw[cam][frame] * scale[cam]
  6. Save aligned depth to new directory

Output:
  /node_data/joon/data/monofusion/markerless_v8/aligned_moge_depth/m5t2/{cam}/depth/{frame:06d}.npy
  /node_data/joon/data/monofusion/markerless_v8/metadata.json
  /node_data/joon/data/monofusion/markerless_v8/alignment_log.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[align] {msg}", flush=True)


def load_Ks_w2cs(data_root, cam_names, n_frames):
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    Ks, w2cs = {}, {}
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])
        with open(meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json") as f:
            m = json.load(f)
        K = np.array(m["k"], dtype=np.float32).squeeze()
        w2c = np.array(m["w2c"], dtype=np.float32).squeeze()
        if m.get("camera_convention") == "c2w":
            w2c = np.array([np.linalg.inv(w) for w in w2c]) if w2c.ndim == 3 else np.linalg.inv(w2c)
        if K.ndim == 2: K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2: w2c = np.tile(w2c[None], (n_frames, 1, 1))
        Ks[cam_name] = K
        w2cs[cam_name] = w2c
    return Ks, w2cs


def load_mask(data_root, cam_name, t):
    paths = sorted((Path(data_root) / "masks" / cam_name).glob("*.npz"))
    return (np.load(paths[t])["dyn_mask"] > 0).astype(np.float32)


def mask_centroid(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0: return None
    return np.array([xs.mean(), ys.mean()])


def mask_multi_centroids(mask, n_slices=5):
    """Return n_slices centroids: one for each horizontal slice of the mask.
    These approximately correspond to body parts visible from multiple cameras.
    Returns list of [x, y] coords (length n_slices), or None if not enough coverage."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) < n_slices * 5:  # at least 5 px per slice
        return None
    y_min, y_max = ys.min(), ys.max()
    if y_max - y_min < n_slices:
        return None
    edges = np.linspace(y_min, y_max + 1, n_slices + 1)
    centroids = []
    for i in range(n_slices):
        in_slice = (ys >= edges[i]) & (ys < edges[i + 1])
        if in_slice.sum() < 3:
            return None
        cx = xs[in_slice].mean()
        cy = ys[in_slice].mean()
        centroids.append([cx, cy])
    return np.array(centroids)


def triangulate_DLT(pts_2d, K_list, w2c_list):
    A = []
    for pt, K, w2c in zip(pts_2d, K_list, w2c_list):
        P = K @ w2c[:3]
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


_moge_model = None
_moge_transform = None


def get_moge_model():
    global _moge_model, _moge_transform
    if _moge_model is None:
        import torch
        import torchvision.transforms as T
        sys.path.insert(0, "/home/joon/dev/MonoFusion/preproc/MoGE")
        from moge.model import MoGeModel
        log("Loading MoGe model...")
        _moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda").eval()
        _moge_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _moge_model, _moge_transform


def run_moge_on_image(img_path, target_size=(512, 512)):
    """Run MoGe, return depth at target resolution."""
    import torch
    from PIL import Image as PILImage
    model, tf = get_moge_model()
    img = tf(PILImage.open(img_path).convert("RGB")).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model.infer(img)
    depth = out["depth"][0].cpu().numpy().astype(np.float32)
    if depth.shape != target_size:
        d = PILImage.fromarray(depth)
        d = d.resize(target_size[::-1], resample=PILImage.BILINEAR)
        depth = np.array(d, dtype=np.float32)
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="markerless_v7 root")
    ap.add_argument("--dst_root", required=True, help="markerless_v8 root (to create)")
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    log(f"src: {src}")
    log(f"dst: {dst}")

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]

    # Resolve image roots (markerless_camXX or m5t2_camXX)
    img_roots = {}
    for cam in cam_names:
        cand = src / "images" / cam.replace("m5t2_", "markerless_")
        if not cand.exists():
            cand = src / "images" / cam
        img_roots[cam] = cand
        log(f"  image root {cam}: {cand}")

    log("Loading cameras...")
    Ks, w2cs = load_Ks_w2cs(args.src_root, cam_names, args.n_frames)

    # Output dirs
    for cam in cam_names:
        (dst / "aligned_moge_depth" / "m5t2" / cam / "depth").mkdir(parents=True, exist_ok=True)

    # Step 1: Run MoGe on all frames, all cams (cache in memory)
    log("Step 1: Run MoGe on all frames...")
    moge_raw = {cam: [] for cam in cam_names}
    for cam in cam_names:
        img_paths = sorted(img_roots[cam].glob("*.png"))[: args.n_frames]
        for i, p in enumerate(img_paths):
            depth = run_moge_on_image(str(p), tuple(args.img_wh))
            moge_raw[cam].append(depth)
            if (i + 1) % 20 == 0:
                log(f"  {cam}: {i+1}/{len(img_paths)}")
    log(f"  MoGe done. shape per cam: {len(moge_raw[cam_names[0]])} × {moge_raw[cam_names[0]][0].shape}")

    # Step 2: For each frame, triangulate and compute scales
    log("Step 2: Triangulate + compute per-cam per-frame scales...")
    alignment_log = {"per_frame": [], "per_cam_median_scale": {}}
    all_scales_per_cam = {cam: [] for cam in cam_names}

    N_SLICES = 5  # multi-point reference
    for t in range(args.n_frames):
        # Multi-point reference: 5 horizontal slices of mask
        slice_centroids = {}  # cam -> [N_SLICES, 2]
        for cam in cam_names:
            mask = load_mask(args.src_root, cam, t)
            mc = mask_multi_centroids(mask, n_slices=N_SLICES)
            slice_centroids[cam] = mc

        valid_cams = [c for c in cam_names if slice_centroids[c] is not None]
        if len(valid_cams) < 2:
            log(f"  t={t}: insufficient valid cams, skipping")
            continue

        # Triangulate each of N_SLICES separately
        # Each slice gives 1 reference 3D world point
        # Note: cross-cam slice correspondence is APPROXIMATE — different cams see
        # the mouse from different angles, so "slice 0" might be head in cam0
        # but neck in cam1. This adds noise but provides more reference points.
        world_pts = []
        for s in range(N_SLICES):
            pts_2d = [slice_centroids[c][s] for c in valid_cams]
            K_list = [Ks[c][t] for c in valid_cams]
            w2c_list = [w2cs[c][t] for c in valid_cams]
            try:
                wp = triangulate_DLT(pts_2d, K_list, w2c_list)
                world_pts.append(wp)
            except Exception:
                pass
        if not world_pts:
            continue

        # Compute per-cam scale by averaging ratios across all reference points
        frame_log = {"t": t, "n_world_pts": len(world_pts), "per_cam": {}}
        for cam in cam_names:
            w2c = w2cs[cam][t]
            ratios = []
            for s, world_pt in enumerate(world_pts):
                pt_cam = w2c @ np.concatenate([world_pt, [1.0]])
                expected_depth_mm = float(pt_cam[2])
                if expected_depth_mm <= 0:
                    continue
                if slice_centroids[cam] is None:
                    continue
                c = slice_centroids[cam][s]
                px, py = int(c[0]), int(c[1])
                if not (0 <= px < args.img_wh[0] and 0 <= py < args.img_wh[1]):
                    continue
                raw = float(moge_raw[cam][t][py, px])
                if raw <= 1e-6:
                    continue
                ratios.append(expected_depth_mm / raw)
            if not ratios:
                frame_log["per_cam"][cam] = {"skipped": "no_valid_refs"}
                continue
            scale = float(np.median(ratios))
            frame_log["per_cam"][cam] = {
                "n_refs": len(ratios),
                "scale_median": scale,
                "scale_std": float(np.std(ratios)),
                "scale": scale,
            }
            all_scales_per_cam[cam].append(scale)
        alignment_log["per_frame"].append(frame_log)

    # Step 3: Per-cam median scale (also compute per-frame for stability)
    log("Step 3: Per-cam median scales:")
    for cam in cam_names:
        slist = all_scales_per_cam[cam]
        if slist:
            med = float(np.median(slist))
            std = float(np.std(slist))
            alignment_log["per_cam_median_scale"][cam] = med
            log(f"  {cam}: median={med:.2f}, std={std:.2f}, n={len(slist)}")

    # Step 4: Apply scales — per-cam per-frame (verified empirically: per-frame > global)
    # Per-frame variation is SIGNAL not noise (mouse depth varies per frame).
    # Global median MVC=19.5%, per-frame MVC=74.3%.
    log("Step 4: Apply scales (per-cam per-frame, fallback to global median)...")
    saved = 0
    for t in range(args.n_frames):
        frame_entry = next((f for f in alignment_log["per_frame"] if f["t"] == t), None)
        for cam in cam_names:
            scale = None
            if frame_entry and cam in frame_entry["per_cam"]:
                pc = frame_entry["per_cam"][cam]
                if "scale" in pc:
                    scale = pc["scale"]
            if scale is None:
                scale = alignment_log["per_cam_median_scale"].get(cam, 130.0)
            aligned = moge_raw[cam][t] * scale
            out_path = dst / "aligned_moge_depth" / "m5t2" / cam / "depth" / f"{t:06d}.npy"
            np.save(out_path, aligned)
            saved += 1
    log(f"  Saved {saved} aligned depth files (per-cam per-frame)")

    # Step 5: Write metadata
    import subprocess
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            cwd="/home/joon/dev/MonoFusion").decode().strip()
    except Exception:
        pass

    metadata = {
        "version": "markerless_v8",
        "preprocessing_date": "2026-04-08",
        "git_commit": git_hash,
        "depth_source": "MoGe (Ruicheng/moge-vitl)",
        "alignment_method": "triangulation-based per-cam per-frame scale",
        "world_units": "mm",
        "depth_units": "mm (after alignment)",
        "alignment_reference": "DLT triangulation of mask centroids across 4 cams",
        "n_frames": args.n_frames,
        "n_cams": len(cam_names),
        "cam_names": cam_names,
        "img_wh": args.img_wh,
        "median_scale_per_cam": alignment_log["per_cam_median_scale"],
        "notes": "Replaces v7 broken DepthAnything fallback. Source: convert_aligned_moge_depth.py logic with triangulation reference instead of DUSt3R.",
    }
    with open(dst / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(dst / "alignment_log.json", "w") as f:
        json.dump(alignment_log, f, indent=2)
    log(f"  Wrote metadata.json and alignment_log.json")

    log("")
    log("Phase 2 alignment complete. Run Phase 3 (MVC validation) next.")


if __name__ == "__main__":
    main()
