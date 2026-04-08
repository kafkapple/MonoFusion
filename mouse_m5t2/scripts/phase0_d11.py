# no-split: D11 — Apply motion warp at t=0 and re-project, vs raw canonical
"""
D11 — Warped FG Gaussian positions vs GT mouse mask (at t=0, cano_t)

D10 measured raw canonical means → 0/3/0/0 % mask hit rate (cam0-3)
D10 found motion_bases[cano_t=0] is NOT identity (rot L1=0.242, trans norms up to 665)

This script computes warped positions at t=0:
    warped[p] = Σ_k motion_coefs[p, k] * (R_k(t=0) @ canonical[p] + trans_k(t=0))
and re-projects them to cam views. Compare to raw projection.

Pre-specified falsification targets:
  IF warped hit rate >> raw hit rate → canonical is not the reference;
     motion does the heavy lifting; blob problem is in motion bases / warping
  IF warped hit rate ≈ raw hit rate (both low) → projection itself is wrong
     (camera bug) or Gaussians are genuinely not near mouse
  IF warped hit rate = ~mask FG fraction (random) → Gaussians are scattered
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[d11] {msg}", flush=True)


def rot6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 matrix (Zhou et al. 2019). rot6d: [..., 6]"""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def warp_means_at_t(canonical, motion_coefs, R_kt, trans_kt):
    """
    canonical: [N, 3]
    motion_coefs: [N, K]  (soft assignment)
    R_kt: [K, 3, 3]
    trans_kt: [K, 3]
    Returns warped [N, 3]
    """
    # per-basis transform: [K, N, 3]
    per_basis = np.einsum("kij,nj->kni", R_kt, canonical) + trans_kt[:, None, :]  # [K, N, 3]
    # weighted sum: [N, 3]
    warped = np.einsum("nk,kni->ni", motion_coefs, per_basis)
    return warped


def project(points_world, K, w2c, img_wh):
    W, H = img_wh
    N = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((N, 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    proj = (K @ pts_cam.T).T
    pixels = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    in_view = (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] >= 0) & (pixels[:, 1] < H) & (depth > 0)
    return pixels, depth, in_view


def load_gt_mask(data_root, cam_name, frame_idx):
    mask_dir = Path(data_root) / "masks" / cam_name
    paths = sorted(mask_dir.glob("*.npz"))
    raw = np.load(paths[frame_idx])["dyn_mask"]
    return (raw > 0).astype(np.float32)


def load_Ks_w2cs(data_root, cam_names, n_frames=80):
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
    return np.concatenate(Ks_list, 0), np.concatenate(w2cs_list, 0)


def measure_for_positions(positions, cam_names, Ks, w2cs, data_root, img_wh, n_frames, t,
                          label, viz_dir, fg_means_raw=None):
    """Measure mask hit rate for given 3D positions at time t per cam. Optionally save viz."""
    results = []
    for cam_idx, cam_name in enumerate(cam_names):
        idx = cam_idx * n_frames + t
        K = Ks[idx]
        w2c = w2cs[idx]
        pixels, depth, in_view = project(positions, K, w2c, img_wh)
        n_in_view = int(in_view.sum())
        try:
            mask = load_gt_mask(data_root, cam_name, t)
            px = pixels[in_view].astype(int)
            W, H = img_wh
            px[:, 0] = np.clip(px[:, 0], 0, W - 1)
            px[:, 1] = np.clip(px[:, 1], 0, H - 1)
            mask_vals = mask[px[:, 1], px[:, 0]]
            n_in_mask = int((mask_vals > 0.5).sum())
            hit_rate = n_in_mask / max(n_in_view, 1)
            log(f"  [{label}] {cam_name} t={t}: in_view={n_in_view}/{positions.shape[0]}, in_mask={n_in_mask} ({100*hit_rate:.1f}%)")
            results.append({
                "cam": cam_name, "label": label,
                "n_in_view": n_in_view, "n_in_mask": n_in_mask, "hit_rate": hit_rate,
            })

            # Viz overlay
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100, facecolor="black")
                ax.set_facecolor("black")
                ax.imshow(mask, cmap="Greens", alpha=0.5, origin="upper")
                if fg_means_raw is not None:
                    # also show raw projection for comparison
                    px_raw, _, iv_raw = project(fg_means_raw, K, w2c, img_wh)
                    ax.scatter(px_raw[iv_raw, 0], px_raw[iv_raw, 1], s=0.5, c="cyan", alpha=0.3, label="raw canonical")
                ax.scatter(pixels[in_view, 0], pixels[in_view, 1], s=0.5, c="red", alpha=0.6, label=label)
                ax.set_xlim(0, W); ax.set_ylim(H, 0)
                ax.set_title(f"{cam_name} t={t} — {label} vs GT mask\nhit_rate={100*hit_rate:.1f}%", color="white")
                ax.legend(loc="upper right", facecolor="black", labelcolor="white")
                ax.axis("off")
                fig.savefig(viz_dir / f"d11_{label}_{cam_name}_t{t:02d}.png",
                            facecolor="black", bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                log(f"    viz error: {e}")
        except Exception as e:
            log(f"  [{label}] {cam_name}: mask error {e}")
    return results


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
    log("D11: Warped FG positions vs GT mask")
    log("=" * 60)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    fg_means = ckpt["fg.params.means"].float().numpy()           # [N, 3]
    motion_coefs = ckpt["fg.params.motion_coefs"].float().numpy() # [N, K]
    # Important: softmax or raw?
    log(f"  motion_coefs stats: min={motion_coefs.min():.3f}, max={motion_coefs.max():.3f}, "
        f"mean={motion_coefs.mean():.3f}")
    # Look at how they are used in code: typically softmax(motion_coefs) is applied before blending
    motion_coefs_sm = np.exp(motion_coefs) / np.exp(motion_coefs).sum(axis=-1, keepdims=True)
    log(f"  motion_coefs after softmax: row sums ≈ 1 (sample: {motion_coefs_sm[0].sum():.3f})")

    rots_6d = ckpt["motion_bases.params.rots"].float().numpy()   # [K, T, 6]
    transls = ckpt["motion_bases.params.transls"].float().numpy() # [K, T, 3]
    K_bases, T_frames, _ = rots_6d.shape
    N = fg_means.shape[0]
    log(f"  N={N}, K_bases={K_bases}, T_frames={T_frames}")

    cam_names = ["m5t2_cam00", "m5t2_cam01", "m5t2_cam02", "m5t2_cam03"]
    Ks, w2cs = load_Ks_w2cs(args.data_root, cam_names, T_frames)
    log(f"  Loaded Ks={Ks.shape}, w2cs={w2cs.shape}")

    results = {"raw": [], "warped": {}}

    # --- Raw canonical baseline ---
    log("")
    log("--- RAW CANONICAL (no motion warp) @ t=0 ---")
    results["raw"] = measure_for_positions(
        fg_means, cam_names, Ks, w2cs, args.data_root, args.img_wh, T_frames, t=0,
        label="raw_cano", viz_dir=viz_dir, fg_means_raw=None,
    )

    # --- Apply motion warp at each target frame ---
    for t in args.frames:
        log("")
        log(f"--- WARPED (motion_bases applied) @ t={t} ---")
        R_kt = rot6d_to_matrix(rots_6d[:, t])   # [K, 3, 3]
        trans_kt = transls[:, t]                # [K, 3]
        warped = warp_means_at_t(fg_means, motion_coefs_sm, R_kt, trans_kt)
        log(f"  warped bbox: min={warped.min(0).tolist()}, max={warped.max(0).tolist()}")
        results["warped"][str(t)] = measure_for_positions(
            warped, cam_names, Ks, w2cs, args.data_root, args.img_wh, T_frames, t=t,
            label=f"warped_t{t:02d}", viz_dir=viz_dir, fg_means_raw=fg_means,
        )

    # Summary
    log("")
    log("=" * 60)
    log("SUMMARY: mean mask hit rate across cams")
    log("=" * 60)
    def mean_hit(rs):
        vals = [r["hit_rate"] for r in rs if "hit_rate" in r]
        return np.mean(vals) if vals else float("nan")
    log(f"  raw canonical:       {100*mean_hit(results['raw']):.1f}%")
    for t in args.frames:
        log(f"  warped @ t={t:>2}:    {100*mean_hit(results['warped'][str(t)]):.1f}%")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"")
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
