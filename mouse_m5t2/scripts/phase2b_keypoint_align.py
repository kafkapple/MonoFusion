# no-split: Phase 2b — depth alignment using DANNCE 22 2D keypoints triangulated to 3D
"""
Phase 2b — Use DANNCE 2D keypoints (22 per frame, 6 cams) as dense reference for alignment.

These are EXACT cross-camera correspondences (each kp_id = same physical body part).
22 keypoints per frame × 4 cams = much denser than mask centroid (1) or slices (5).

Steps per frame:
  1. Load 22 2D keypoints per cam from result_view_{cam}.pkl at original resolution (1152×1024)
  2. Filter by confidence > threshold
  3. Anisotropically scale to 512×512 (sx=512/1152, sy=512/1024)
  4. Triangulate each keypoint visible in ≥2 cams → 3D world point
  5. For each cam, compute MoGe raw depth at the keypoint pixel
  6. Compute scale = expected_depth / raw_depth
  7. Per cam, take median over all valid keypoints

Output: aligned depth in markerless_v8/aligned_moge_depth/
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[kp-align] {msg}", flush=True)


def load_2d_keypoints(raw_root, n_cams=4):
    """Load 2D keypoints for cams 0..n_cams-1. Returns array (n_cams, 18000, 22, 3)."""
    kp_list = []
    for cam in range(n_cams):
        path = Path(raw_root) / "keypoints2d_undist" / f"result_view_{cam}.pkl"
        with open(path, "rb") as f:
            kp = pickle.load(f)
        kp_list.append(kp)
    return np.stack(kp_list, axis=0)


def load_Ks_w2cs(data_root, cam_names, n_frames):
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    Ks, w2cs = {}, {}
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])
        with open(meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json") as f:
            m = json.load(f)
        K = np.array(m["k"], dtype=np.float32).squeeze()
        w2c = np.array(m["w2c"], dtype=np.float32).squeeze()
        if K.ndim == 2: K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2: w2c = np.tile(w2c[None], (n_frames, 1, 1))
        Ks[cam_name] = K
        w2cs[cam_name] = w2c
    return Ks, w2cs


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


def get_moge():
    global _moge_model, _moge_transform
    if _moge_model is None:
        import torch
        import torchvision.transforms as T
        sys.path.insert(0, "/home/joon/dev/MonoFusion/preproc/MoGE")
        from moge.model import MoGeModel
        log("Loading MoGe...")
        _moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda").eval()
        _moge_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _moge_model, _moge_transform


def run_moge(img_path, target_size=(512, 512)):
    import torch
    from PIL import Image as PILImage
    model, tf = get_moge()
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
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--raw_root", required=True, help="raw markerless_mouse_1_nerf dir")
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--start_frame", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--orig_wh", type=int, nargs=2, default=[1152, 1024])
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--conf_thresh", type=float, default=0.5)
    args = ap.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    n_cams = len(cam_names)

    # Output dirs
    for cam in cam_names:
        (dst / "aligned_moge_depth" / "m5t2" / cam / "depth").mkdir(parents=True, exist_ok=True)

    log("Loading 2D keypoints...")
    kp_all = load_2d_keypoints(args.raw_root, n_cams=n_cams)  # (4, 18000, 22, 3)
    log(f"  shape: {kp_all.shape}")

    log("Loading cameras (already at 512×512 K)...")
    Ks, w2cs = load_Ks_w2cs(args.src_root, cam_names, args.n_frames)

    # Image roots
    img_roots = {}
    for cam in cam_names:
        cand = src / "images" / cam.replace("m5t2_", "markerless_")
        if not cand.exists():
            cand = src / "images" / cam
        img_roots[cam] = cand

    sx = args.img_wh[0] / args.orig_wh[0]
    sy = args.img_wh[1] / args.orig_wh[1]
    log(f"Anisotropic scaling: sx={sx:.4f}, sy={sy:.4f}")

    # Step 1: Run MoGe on all images
    log("Step 1: Run MoGe...")
    moge_raw = {cam: [] for cam in cam_names}
    for cam in cam_names:
        img_paths = sorted(img_roots[cam].glob("*.png"))[: args.n_frames]
        for i, p in enumerate(img_paths):
            d = run_moge(str(p), tuple(args.img_wh))
            moge_raw[cam].append(d)
        log(f"  {cam}: {len(moge_raw[cam])} frames")

    # Step 2: Per frame, triangulate keypoints + compute scales
    log("Step 2: Triangulate keypoints + compute scales...")
    alignment_log = {"per_frame": []}
    all_scales_per_cam = {cam: [] for cam in cam_names}

    for t in range(args.n_frames):
        orig_frame_id = args.start_frame + t * args.stride

        # Load 22 keypoints per cam at this orig_frame_id
        kps_per_cam = {}  # cam -> [22, 3] (x_512, y_512, conf)
        for cam_idx, cam in enumerate(cam_names):
            kp_orig = kp_all[cam_idx, orig_frame_id]  # (22, 3)
            kp_512 = kp_orig.copy()
            kp_512[:, 0] *= sx
            kp_512[:, 1] *= sy
            kps_per_cam[cam] = kp_512

        # For each keypoint id (0..21), find cameras with high confidence
        per_frame_log = {"t": t, "orig_frame": orig_frame_id, "n_kps_used": 0,
                         "world_pts": [], "per_cam": {}}
        per_cam_ratios = {cam: [] for cam in cam_names}

        for kp_id in range(22):
            valid_cams = [cam for cam in cam_names if kps_per_cam[cam][kp_id, 2] > args.conf_thresh]
            if len(valid_cams) < 2:
                continue
            pts_2d = [kps_per_cam[c][kp_id, :2] for c in valid_cams]
            K_list = [Ks[c][t] for c in valid_cams]
            w2c_list = [w2cs[c][t] for c in valid_cams]
            world_pt = triangulate_DLT(pts_2d, K_list, w2c_list)
            # Filter outliers: world point should be within reasonable arena bounds
            if np.any(np.abs(world_pt) > 2000) or np.any(np.isnan(world_pt)):
                continue
            per_frame_log["world_pts"].append(world_pt.tolist())

            # For each cam (not just valid_cams), compute scale
            for cam in cam_names:
                w2c = w2cs[cam][t]
                pt_cam = w2c @ np.concatenate([world_pt, [1.0]])
                expected = float(pt_cam[2])
                if expected <= 0:
                    continue
                # Project to cam to get pixel
                K = Ks[cam][t]
                proj = K @ pt_cam[:3]
                if proj[2] <= 0:
                    continue
                px = int(proj[0] / proj[2])
                py = int(proj[1] / proj[2])
                if not (0 <= px < args.img_wh[0] and 0 <= py < args.img_wh[1]):
                    continue
                raw = float(moge_raw[cam][t][py, px])
                if raw <= 1e-6:
                    continue
                ratio = expected / raw
                # Sanity range: reasonable scale should be within 50-300
                if 50 < ratio < 300:
                    per_cam_ratios[cam].append(ratio)

        per_frame_log["n_kps_used"] = len(per_frame_log["world_pts"])

        for cam in cam_names:
            ratios = per_cam_ratios[cam]
            if not ratios:
                per_frame_log["per_cam"][cam] = {"skipped": "no_valid_kps"}
                continue
            scale = float(np.median(ratios))
            per_frame_log["per_cam"][cam] = {
                "n_refs": len(ratios),
                "scale": scale,
                "scale_std": float(np.std(ratios)),
            }
            all_scales_per_cam[cam].append(scale)

        alignment_log["per_frame"].append(per_frame_log)

        if t % 20 == 0:
            log(f"  t={t}: {per_frame_log['n_kps_used']} world pts, "
                f"per-cam refs: {[per_frame_log['per_cam'][c].get('n_refs', 0) for c in cam_names]}")

    # Step 3: Per-cam median across frames
    log("")
    log("Step 3: Per-cam median scales:")
    medians = {}
    for cam in cam_names:
        slist = all_scales_per_cam[cam]
        if slist:
            med = float(np.median(slist))
            std = float(np.std(slist))
            medians[cam] = med
            log(f"  {cam}: median={med:.2f}, std={std:.2f}, n={len(slist)}")
    alignment_log["per_cam_median_scale"] = medians

    # Step 4: Apply per-frame per-cam scales
    log("Step 4: Apply scales and save aligned depth...")
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
                scale = medians.get(cam, 130.0)
            aligned = moge_raw[cam][t] * scale
            np.save(dst / "aligned_moge_depth" / "m5t2" / cam / "depth" / f"{t:06d}.npy", aligned)
            saved += 1
    log(f"  Saved {saved} files")

    # Metadata
    metadata = {
        "version": "markerless_v8 (keypoint-aligned)",
        "alignment_method": "DANNCE 22 2D keypoints triangulated, per-cam per-frame median scale",
        "world_units": "mm",
        "n_frames": args.n_frames,
        "n_cams": n_cams,
        "n_keypoints": 22,
        "median_scale_per_cam": medians,
    }
    with open(dst / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(dst / "alignment_log.json", "w") as f:
        json.dump(alignment_log, f, indent=2)
    log(f"Wrote metadata and alignment log")


if __name__ == "__main__":
    main()
