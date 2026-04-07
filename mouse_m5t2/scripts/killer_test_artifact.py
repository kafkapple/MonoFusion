# no-split: single diagnostic experiment — load → frame-shift → metrics
"""
Killer test for V8/V9 PSNR-vs-quality artifact hypothesis.

Tests two things:
1. Frame-shift consistency: render frame t, evaluate against gt[t+k] for k in [0,1,5,10,20]
   - Real reconstruction → loss(k=0) << loss(k=20) (ratio > 1.5)
   - Temporal artifact → loss flat across k (ratio ≈ 1.0)

2. FG-only / BG-only PSNR split using gt mask
   - Real → both FG and BG PSNR are high
   - Artifact → BG high, FG low

Usage (on gpu03):
    cd ~/dev/MonoFusion
    PYTHONPATH=. python mouse_m5t2/scripts/killer_test_artifact.py \
        --checkpoint /node_data/joon/data/monofusion/markerless_v7/results_v9c/checkpoints/best.ckpt \
        --data_root /node_data/joon/data/monofusion/markerless_v7 \
        --output /node_data/joon/data/monofusion/markerless_v7/results_v9c/killer_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_model(ckpt_path: Path, device: str):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SceneModel.init_from_state_dict(ckpt["model"]).to(device).eval()
    print(f"  Loaded: FG={model.num_fg_gaussians}, BG={model.num_bg_gaussians}, total={model.num_gaussians}")
    return model


def load_gt_image(data_root: Path, cam_name: str, frame_idx: int) -> np.ndarray:
    img_dir = data_root / "images" / cam_name
    paths = sorted(img_dir.glob("*.png"))
    img = np.array(Image.open(paths[frame_idx]).convert("RGB")).astype(np.float32) / 255.0
    return img  # (H, W, 3)


def load_gt_mask(data_root: Path, cam_name: str, frame_idx: int) -> np.ndarray:
    """Load FG mask. Stored as .npz with key 'dyn_mask', values in [-1,1]. FG = (>0)."""
    mask_dir = data_root / "masks" / cam_name
    paths = sorted(mask_dir.glob("*.npz"))
    raw = np.load(paths[frame_idx])["dyn_mask"]  # float32, [-1, 1]
    return (raw > 0).astype(np.float32)  # binary FG mask


@torch.no_grad()
def render_at(model, t_render: int, cam_idx: int, device: str, n_frames: int = 80, img_wh=(512, 512)):
    """Render scene at time t_render from camera cam_idx."""
    idx = cam_idx * n_frames + t_render
    w2c = model.w2cs[idx:idx+1].to(device)
    K = model.Ks[idx:idx+1].to(device)
    out = model.render(
        t=t_render, w2cs=w2c, Ks=K, img_wh=img_wh,
        return_color=True, return_feat=False, return_depth=False, return_mask=True,
    )
    rgb = out["img"][0].clamp(0, 1).cpu().numpy()  # (H, W, 3)
    rendered_mask = out.get("mask", torch.zeros(1, img_wh[1], img_wh[0], 1))[0, :, :, 0].cpu().numpy()
    return rgb, rendered_mask


def psnr(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR. If mask provided, computed only over masked pixels."""
    diff = (x - y) ** 2
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[..., None]
        valid = mask > 0.5
        if valid.sum() == 0:
            return float("nan")
        mse = (diff * valid).sum() / (valid.sum() * x.shape[-1])
    else:
        mse = diff.mean()
    if mse < 1e-10:
        return 99.0
    return float(10 * np.log10(1.0 / mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--n_cams", type=int, default=4)
    p.add_argument("--img_wh", type=int, nargs=2, default=[512, 512])
    args = p.parse_args()

    device = "cuda"
    data_root = Path(args.data_root)
    model = load_model(Path(args.checkpoint), device)

    n_frames = args.n_frames
    n_cams = args.n_cams
    cam_names = [f"m5t2_cam{i:02d}" for i in range(n_cams)]

    results: dict = {
        "checkpoint": str(args.checkpoint),
        "n_frames": n_frames,
        "n_cams": n_cams,
        "tests": {},
    }

    # ─────────────────────────────────────────────────────────────────
    # TEST 1: Frame-shift consistency
    # Render frame t, evaluate against gt[t+k] for k in [0,1,5,10,20]
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 1: Frame-shift consistency ===")
    print("If loss is flat across k → temporal average artifact (BAD)")
    print("If loss(k=0) << loss(k=20) → real reconstruction (GOOD)\n")

    test_frames = [10, 30, 50]  # render at these t values
    shifts = [0, 1, 5, 10, 20]
    cam_idx = 0  # cam00 has the best mouse coverage

    shift_results = []
    for t in test_frames:
        for k in shifts:
            t_eval = t + k
            if t_eval >= n_frames:
                continue
            # Render at time t (uses model's t-th time pose)
            pred_rgb, _ = render_at(model, t, cam_idx, device, n_frames, tuple(args.img_wh))
            gt_rgb = load_gt_image(data_root, cam_names[cam_idx], t_eval)
            mse = float(((pred_rgb - gt_rgb) ** 2).mean())
            l1 = float(np.abs(pred_rgb - gt_rgb).mean())
            psnr_val = psnr(pred_rgb, gt_rgb)
            shift_results.append({"t_render": t, "k": k, "t_eval": t_eval, "mse": mse, "l1": l1, "psnr": psnr_val})
            print(f"  t={t}, k={k:2d}, t_eval={t_eval:2d}: MSE={mse:.5f}, L1={l1:.5f}, PSNR={psnr_val:.2f}")

    # Compute ratio for diagnostic
    ratios = []
    for t in test_frames:
        same = next((r for r in shift_results if r["t_render"] == t and r["k"] == 0), None)
        far = next((r for r in shift_results if r["t_render"] == t and r["k"] == 20), None)
        if same and far and same["mse"] > 0:
            ratio = far["mse"] / same["mse"]
            ratios.append(ratio)
            print(f"  >> t={t}: MSE(k=20) / MSE(k=0) = {ratio:.3f}")
    if ratios:
        avg_ratio = float(np.mean(ratios))
        verdict = "REAL (good)" if avg_ratio > 1.5 else ("ARTIFACT (bad)" if avg_ratio < 1.1 else "MIXED")
        print(f"\n  >> Avg ratio: {avg_ratio:.3f} → Verdict: {verdict}")
        results["tests"]["frame_shift"] = {
            "shifts": shifts, "test_frames": test_frames, "results": shift_results,
            "avg_ratio": avg_ratio, "verdict": verdict,
        }

    # ─────────────────────────────────────────────────────────────────
    # TEST 2: FG-only / BG-only PSNR split (per camera × per frame)
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 2: FG-only vs BG-only PSNR ===")
    print("Real reconstruction → both high")
    print("Artifact → BG high, FG low (BG averages over mouse, FG ghost)\n")

    eval_frames = list(range(0, n_frames, 10))  # every 10 frames
    fg_psnrs, bg_psnrs, full_psnrs = [], [], []
    per_frame = []

    for t in eval_frames:
        for c in range(n_cams):
            try:
                pred_rgb, rendered_mask = render_at(model, t, c, device, n_frames, tuple(args.img_wh))
                gt_rgb = load_gt_image(data_root, cam_names[c], t)
                gt_mask = load_gt_mask(data_root, cam_names[c], t)
                full = psnr(pred_rgb, gt_rgb)
                fg = psnr(pred_rgb, gt_rgb, mask=gt_mask)
                bg = psnr(pred_rgb, gt_rgb, mask=(1.0 - gt_mask))
                fg_pixel_pct = float((gt_mask > 0.5).mean())
                full_psnrs.append(full); fg_psnrs.append(fg); bg_psnrs.append(bg)
                per_frame.append({
                    "t": t, "cam": c, "full_psnr": full, "fg_psnr": fg, "bg_psnr": bg,
                    "fg_pct": fg_pixel_pct,
                })
                print(f"  t={t:2d} cam{c}: full={full:5.2f} | FG={fg:5.2f} | BG={bg:5.2f} | FG%={fg_pixel_pct*100:.1f}")
            except Exception as e:
                print(f"  t={t} cam{c}: FAILED — {e}")

    if fg_psnrs:
        full_avg = float(np.nanmean(full_psnrs))
        fg_avg = float(np.nanmean(fg_psnrs))
        bg_avg = float(np.nanmean(bg_psnrs))
        gap = bg_avg - fg_avg
        print(f"\n  >> Avg Full PSNR: {full_avg:.2f}")
        print(f"  >> Avg FG PSNR:   {fg_avg:.2f}")
        print(f"  >> Avg BG PSNR:   {bg_avg:.2f}")
        print(f"  >> BG−FG gap:     {gap:.2f}  (large gap → BG dominates → artifact)")
        verdict2 = "ARTIFACT (BG dominates)" if gap > 5.0 else ("REAL (balanced)" if gap < 2.0 else "MIXED")
        print(f"  >> Verdict: {verdict2}")
        results["tests"]["fg_bg_psnr"] = {
            "per_frame": per_frame,
            "avg_full_psnr": full_avg, "avg_fg_psnr": fg_avg, "avg_bg_psnr": bg_avg,
            "gap_bg_minus_fg": gap, "verdict": verdict2,
        }

    # ─────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
