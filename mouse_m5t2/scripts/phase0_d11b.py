# no-split: D11b — fixed warp formula using actual compute_transforms logic
"""
D11b — corrected warp: blend 6D rots BEFORE converting to rmat, match actual code

D11 used wrong formula: Σ coef_k * (R_k @ x + t_k)
D11b uses correct: convert (Σ coef_k * rot6d_k) → rmat, then rmat @ x + Σ coef_k * t_k

Uses torch for consistency with actual scene_model code.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def log(msg):
    print(f"[d11b] {msg}", flush=True)


def cont_6d_to_rmat(rot6d):
    """6D rotation → 3x3 rmat. Torch version matching flow3d."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def compute_transforms_torch(rots, transls, coefs, t):
    """
    rots: (K, T, 6)  motion_bases params
    transls: (K, T, 3)
    coefs: (G, K)  softmax-activated
    t: int
    returns (G, 3, 4)
    """
    transls_t = transls[:, t]  # (K, 3)
    rots_t = rots[:, t]        # (K, 6)
    # Blend with coefs
    transls_blend = torch.einsum("pk,ki->pi", coefs, transls_t)  # (G, 3)
    rots_blend = torch.einsum("pk,ki->pi", coefs, rots_t)        # (G, 6)
    rmat = cont_6d_to_rmat(rots_blend)                            # (G, 3, 3)
    return rmat, transls_blend  # (G, 3, 3), (G, 3)


def warp_means(means, rmat, transl):
    """means: (G, 3), rmat: (G, 3, 3), transl: (G, 3) → (G, 3)"""
    return torch.einsum("pij,pj->pi", rmat, means) + transl


def project(points_world, K, w2c, img_wh):
    W, H = img_wh
    N = points_world.shape[0]
    pts_h = torch.cat([points_world, torch.ones(N, 1)], dim=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    proj = (K @ pts_cam.T).T
    pixels = proj[:, :2] / torch.clamp(proj[:, 2:3], min=1e-6)
    in_view = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < W) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < H) & (depth > 0)
    )
    return pixels.numpy(), depth.numpy(), in_view.numpy()


def load_gt_mask(data_root, cam_name, frame_idx):
    mask_dir = Path(data_root) / "masks" / cam_name
    paths = sorted(mask_dir.glob("*.npz"))
    raw = np.load(paths[frame_idx])["dyn_mask"]
    return (raw > 0).astype(np.float32)


def load_Ks_w2cs(data_root, cam_names, n_frames):
    meta_dir = Path(data_root) / "_raw_data" / "markerless" / "trajectory"
    Ks_list, w2cs_list = [], []
    for cam_name in cam_names:
        cam_idx = int(cam_name[-2:])
        with open(meta_dir / f"Dy_train_meta_cam{cam_idx:02d}.json") as f:
            meta = json.load(f)
        K = np.array(meta["k"], dtype=np.float32).squeeze()
        w2c = np.array(meta["w2c"], dtype=np.float32).squeeze()
        if meta.get("camera_convention") == "c2w":
            w2c = np.array([np.linalg.inv(w) for w in w2c]) if w2c.ndim == 3 else np.linalg.inv(w2c)
        if K.ndim == 2: K = np.tile(K[None], (n_frames, 1, 1))
        if w2c.ndim == 2: w2c = np.tile(w2c[None], (n_frames, 1, 1))
        Ks_list.append(K)
        w2cs_list.append(w2c)
    return (
        torch.from_numpy(np.concatenate(Ks_list, 0)),
        torch.from_numpy(np.concatenate(w2cs_list, 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--viz_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60])
    ap.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("D11b: Corrected warp formula (matches scene_model.compute_transforms)")
    log("=" * 60)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    fg_means = sd["fg.params.means"].float()          # (G, 3)
    motion_coefs_raw = sd["fg.params.motion_coefs"].float()  # (G, K)
    motion_coefs = F.softmax(motion_coefs_raw, dim=-1)
    rots = sd["motion_bases.params.rots"].float()     # (K, T, 6)
    transls = sd["motion_bases.params.transls"].float()  # (K, T, 3)
    G, K = motion_coefs.shape
    T = rots.shape[1]
    log(f"  G={G}, K={K}, T={T}")
    log(f"  motion_coefs (softmax) stats: min={motion_coefs.min():.3f}, max={motion_coefs.max():.3f}, mean={motion_coefs.mean():.3f}")
    log(f"  motion_coefs (softmax) row max: mean={motion_coefs.max(dim=-1).values.mean():.3f} (≈1 means hard assignment)")

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    Ks, w2cs = load_Ks_w2cs(args.data_root, cam_names, T)
    log(f"  Ks={Ks.shape}, w2cs={w2cs.shape}")

    results = {}

    # Raw canonical baseline
    log("")
    log("--- RAW CANONICAL (no warp) @ t=0 ---")
    raw_results = []
    for cam_idx, cam_name in enumerate(cam_names):
        idx = cam_idx * T + 0
        pixels, _, in_view = project(fg_means, Ks[idx], w2cs[idx], args.img_wh)
        mask = load_gt_mask(args.data_root, cam_name, 0)
        W, H = args.img_wh
        px = pixels[in_view].astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, W - 1)
        px[:, 1] = np.clip(px[:, 1], 0, H - 1)
        hit = (mask[px[:, 1], px[:, 0]] > 0.5).sum()
        log(f"  raw {cam_name}: in_view={int(in_view.sum())}/{G}, in_mask={int(hit)} ({100*hit/max(in_view.sum(),1):.1f}%)")
        raw_results.append({"cam": cam_name, "in_view": int(in_view.sum()), "in_mask": int(hit),
                            "hit_rate": float(hit/max(in_view.sum(),1))})
    results["raw"] = raw_results

    # Warped at each target frame
    results["warped"] = {}
    for t in args.frames:
        log("")
        log(f"--- WARPED (corrected) @ t={t} ---")
        rmat, transl = compute_transforms_torch(rots, transls, motion_coefs, t)
        warped = warp_means(fg_means, rmat, transl)  # (G, 3)
        log(f"  warped bbox: min={warped.min(0).values.tolist()}, max={warped.max(0).values.tolist()}")

        wresults = []
        for cam_idx, cam_name in enumerate(cam_names):
            idx = cam_idx * T + t
            pixels, _, in_view = project(warped, Ks[idx], w2cs[idx], args.img_wh)
            mask = load_gt_mask(args.data_root, cam_name, t)
            W, H = args.img_wh
            px = pixels[in_view].astype(int)
            px[:, 0] = np.clip(px[:, 0], 0, W - 1)
            px[:, 1] = np.clip(px[:, 1], 0, H - 1)
            hit = (mask[px[:, 1], px[:, 0]] > 0.5).sum()
            hit_rate = float(hit/max(in_view.sum(),1))
            log(f"  warped_t{t:02d} {cam_name}: in_view={int(in_view.sum())}/{G}, in_mask={int(hit)} ({100*hit_rate:.1f}%)")
            wresults.append({"cam": cam_name, "t": t, "in_view": int(in_view.sum()),
                             "in_mask": int(hit), "hit_rate": hit_rate})

            # Viz
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100, facecolor="black")
                ax.set_facecolor("black")
                ax.imshow(mask, cmap="Greens", alpha=0.5, origin="upper")
                ax.scatter(pixels[in_view, 0], pixels[in_view, 1], s=0.5, c="red", alpha=0.6)
                ax.set_xlim(0, W); ax.set_ylim(H, 0)
                ax.set_title(f"{cam_name} t={t} (warped) — hit={100*hit_rate:.1f}%", color="white")
                ax.axis("off")
                fig.savefig(viz_dir / f"d11b_warped_t{t:02d}_{cam_name}.png",
                            facecolor="black", bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                log(f"    viz err: {e}")
        results["warped"][str(t)] = wresults

    log("")
    log("=" * 60)
    log("SUMMARY:")
    log(f"  raw canonical:  {100*np.mean([r['hit_rate'] for r in raw_results]):.1f}%")
    for t in args.frames:
        rs = results["warped"][str(t)]
        log(f"  warped @ t={t:>2}: {100*np.mean([r['hit_rate'] for r in rs]):.1f}%")

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
