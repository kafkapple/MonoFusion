# no-split: Phase 1 — run real MoGe + measure per-cam scale via triangulation
"""
Phase 1 — measure depth alignment parameters without destruction.

Steps:
  1. Run real MoGe (not DepthAnything fallback) on 4 cams × selected frames
  2. For each frame, triangulate 3D point(s) using known metric camera poses
     (e.g., mouse mask centroid across 4 cams)
  3. At that 3D point, compute what depth each cam SHOULD see (metric mm)
  4. Compare to MoGe's raw depth at the projected pixel → solve per-cam scale
  5. Report: is scale cam-specific? frame-specific? stable?

NO destruction. NO checkpoint changes. Just measurement + report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[phase1] {msg}", flush=True)


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
    """2D centroid pixel of mask."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def triangulate_DLT(pts_2d, K_list, w2c_list):
    """
    Linear DLT triangulation from N views.
    pts_2d: list of [2] pixel coords
    K_list, w2c_list: length-N lists of 3x3 and 4x4
    Returns 3D world point [3].
    """
    A = []
    for pt, K, w2c in zip(pts_2d, K_list, w2c_list):
        P = K @ w2c[:3]  # 3x4 projection
        x, y = pt
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def run_moge_on_frame(img_path):
    """Run MoGe on a single image, return raw depth (H, W) and points (H, W, 3)."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    if not hasattr(run_moge_on_frame, "model"):
        sys.path.insert(0, "/home/joon/dev/MonoFusion/preproc/MoGE")
        from moge.model import MoGeModel
        log("Loading MoGe model...")
        run_moge_on_frame.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda").eval()
        run_moge_on_frame.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    img = run_moge_on_frame.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = run_moge_on_frame.model.infer(img)
    depth = out["depth"][0].cpu().numpy()
    points = out["points"][0].cpu().numpy() if "points" in out else None
    intrinsics = out["intrinsics"][0].cpu().numpy() if "intrinsics" in out else None
    return depth, points, intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 10, 30, 50, 70])
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    log("=" * 60)
    log("Phase 1: MoGe + triangulation-based scale measurement")
    log("=" * 60)

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    # Resolve image path — prefer markerless_* then m5t2_*
    img_roots = {}
    for cam in cam_names:
        cand1 = Path(args.data_root) / "images" / cam.replace("m5t2_", "markerless_")
        cand2 = Path(args.data_root) / "images" / cam
        if cand1.exists():
            img_roots[cam] = cand1
        elif cand2.exists():
            img_roots[cam] = cand2
        else:
            raise FileNotFoundError(f"No images for {cam}")
        log(f"  image root {cam}: {img_roots[cam]}")

    Ks, w2cs = load_Ks_w2cs(args.data_root, cam_names, 80)

    results = {}

    for t in args.frames:
        log("")
        log(f"=== Frame t={t} ===")

        # 1) Load masks and compute centroid
        centroids_2d = {}
        masks = {}
        for cam in cam_names:
            mask = load_mask(args.data_root, cam, t)
            masks[cam] = mask
            c = mask_centroid(mask)
            centroids_2d[cam] = c
            log(f"  {cam} mask centroid: {c.tolist() if c is not None else None}")

        valid_cams = [c for c in cam_names if centroids_2d[c] is not None]
        if len(valid_cams) < 2:
            log(f"  [skip] insufficient valid cams ({len(valid_cams)})")
            continue

        # 2) Triangulate 3D world point from multi-view centroids
        pts_2d = [centroids_2d[c] for c in valid_cams]
        K_list = [Ks[c][t] for c in valid_cams]
        w2c_list = [w2cs[c][t] for c in valid_cams]
        world_pt = triangulate_DLT(pts_2d, K_list, w2c_list)
        log(f"  triangulated world 3D centroid: {world_pt.tolist()}")

        # 3) For each cam, compute the EXPECTED depth at the centroid pixel
        #    (depth_expected = z in cam frame after transforming world_pt)
        expected_depths = {}
        for cam in cam_names:
            w2c = w2cs[cam][t]
            pt_cam = w2c @ np.concatenate([world_pt, [1.0]])
            expected_depth_mm = pt_cam[2]  # z in cam frame = depth
            expected_depths[cam] = float(expected_depth_mm)
            log(f"  {cam} expected metric depth at centroid: {expected_depth_mm:.2f} mm")

        # 4) Run MoGe on each image, get raw depth at centroid pixel
        moge_depths = {}
        for cam in cam_names:
            img_path = sorted(img_roots[cam].glob("*.png"))[t]
            depth, points, moge_intr = run_moge_on_frame(str(img_path))
            c = centroids_2d[cam]
            if c is None:
                continue
            px, py = int(c[0]), int(c[1])
            # MoGe depth is at its own resolution; assume same as image (512x512)
            if depth.shape != tuple(args.img_wh):
                # resize if needed
                from PIL import Image as PILImage
                depth_img = PILImage.fromarray(depth)
                depth_img = depth_img.resize(args.img_wh, resample=PILImage.BILINEAR)
                depth = np.array(depth_img)
            moge_raw = float(depth[py, px])
            moge_depths[cam] = moge_raw
            log(f"  {cam} MoGe raw depth at centroid ({px},{py}): {moge_raw:.4f}")

        # 5) Per-cam scale = expected / moge_raw
        log("")
        log("  Per-cam scale factors (expected_mm / moge_raw):")
        scales = {}
        for cam in cam_names:
            if cam in expected_depths and cam in moge_depths:
                exp = expected_depths[cam]
                raw = moge_depths[cam]
                if raw > 1e-6:
                    s = exp / raw
                    scales[cam] = s
                    log(f"    {cam}: scale = {s:.2f}  (expected {exp:.2f} mm / moge {raw:.4f})")
        scale_vals = list(scales.values())
        if scale_vals:
            log(f"  → median scale: {np.median(scale_vals):.2f}")
            log(f"  → std of scales: {np.std(scale_vals):.2f}")
            log(f"  → max/min ratio: {max(scale_vals)/min(scale_vals):.2f}")

        results[str(t)] = {
            "centroids_2d": {k: v.tolist() if v is not None else None for k, v in centroids_2d.items()},
            "world_point_mm": world_pt.tolist(),
            "expected_depths_mm": expected_depths,
            "moge_raw_depths": moge_depths,
            "per_cam_scales": scales,
            "median_scale": float(np.median(scale_vals)) if scale_vals else None,
        }

    # Cross-frame summary
    log("")
    log("=" * 60)
    log("Cross-frame summary")
    log("=" * 60)
    all_scales_per_cam = {cam: [] for cam in cam_names}
    for t_key, r in results.items():
        for cam, s in r.get("per_cam_scales", {}).items():
            all_scales_per_cam[cam].append(s)

    for cam, slist in all_scales_per_cam.items():
        if slist:
            log(f"  {cam}: n={len(slist)}, median={np.median(slist):.2f}, "
                f"std={np.std(slist):.2f}, range=[{min(slist):.2f}, {max(slist):.2f}]")

    # Save
    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
