# no-split: single diagnostic — load model → fg-only render → fg-only PSNR
"""
FG-only diagnostic: render an existing trained model with only FG Gaussians,
measure FG PSNR. This answers the question:

  "Does the model's FG learn a recognizable mouse, or is the visual quality
   in the killer test entirely from BG bleeding into the FG region?"

Usage:
    cd ~/dev/MonoFusion
    PYTHONPATH=. python mouse_m5t2/scripts/fg_only_diagnostic.py \
        --checkpoint <best.ckpt> \
        --data_root <markerless_v7> \
        --output <results_xxx>/fg_only_diag.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model(ckpt_path, device):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SceneModel.init_from_state_dict(ckpt["model"]).to(device).eval()
    print(f"  Loaded: FG={model.num_fg_gaussians}, BG={model.num_bg_gaussians}, total={model.num_gaussians}")
    return model


def load_gt_image(data_root, cam_name, frame_idx):
    img_dir = data_root / "images" / cam_name
    paths = sorted(img_dir.glob("*.png"))
    return np.array(Image.open(paths[frame_idx]).convert("RGB")).astype(np.float32) / 255.0


def load_gt_mask(data_root, cam_name, frame_idx):
    mask_dir = data_root / "masks" / cam_name
    paths = sorted(mask_dir.glob("*.npz"))
    raw = np.load(paths[frame_idx])["dyn_mask"]
    return (raw > 0).astype(np.float32)


def psnr(x, y, mask=None):
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


@torch.no_grad()
def render_fg_only(model, t, cam_idx, device, n_frames=80, img_wh=(512, 512)):
    """Render at time t from camera cam_idx using FG Gaussians only."""
    idx = cam_idx * n_frames + t
    w2c = model.w2cs[idx:idx+1].to(device)
    K = model.Ks[idx:idx+1].to(device)
    out = model.render(
        t=t, w2cs=w2c, Ks=K, img_wh=img_wh,
        return_color=True, return_feat=False, return_depth=False, return_mask=False,
        fg_only=True,  # ★ KEY: render FG Gaussians only
    )
    return out["img"][0].clamp(0, 1).cpu().numpy()


@torch.no_grad()
def render_bg_only(model, t, cam_idx, device, n_frames=80, img_wh=(512, 512)):
    """Render at time t using BG Gaussians only (BG is static, t doesn't matter for pose)."""
    if model.num_bg_gaussians == 0:
        return None
    idx = cam_idx * n_frames + t
    w2c = model.w2cs[idx:idx+1].to(device)
    K = model.Ks[idx:idx+1].to(device)
    out = model.render(
        t=t, w2cs=w2c, Ks=K, img_wh=img_wh,
        return_color=True, return_feat=False, return_depth=False, return_mask=False,
        bg_only=True,
    )
    return out["img"][0].clamp(0, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--n_cams", type=int, default=4)
    p.add_argument("--save_previews", type=str, default=None,
                   help="Optional dir to save fg-only and bg-only PNG previews")
    args = p.parse_args()

    device = "cuda"
    data_root = Path(args.data_root)
    model = load_model(Path(args.checkpoint), device)

    n_frames = args.n_frames
    n_cams = args.n_cams
    cam_names = [f"m5t2_cam{i:02d}" for i in range(n_cams)]

    eval_frames = list(range(0, n_frames, 10))
    fg_psnrs = []      # PSNR of fg_only render vs GT, masked to FG region
    bg_psnrs = []      # PSNR of bg_only render vs GT, masked to BG region
    fg_full = []       # PSNR of fg_only render vs GT, full image (for comparison)
    per_frame = []

    print("\n=== FG-only / BG-only render evaluation ===")
    print("FG PSNR (FG-only render at FG region) — direct measure of FG quality")
    print("BG PSNR (BG-only render at BG region) — direct measure of BG quality")
    print()

    for t in eval_frames:
        for c in range(n_cams):
            try:
                fg_img = render_fg_only(model, t, c, device, n_frames)
                bg_img = render_bg_only(model, t, c, device, n_frames) if model.num_bg_gaussians > 0 else None
                gt_img = load_gt_image(data_root, cam_names[c], t)
                gt_mask = load_gt_mask(data_root, cam_names[c], t)
                # FG-only render evaluated only at FG region
                fg_psnr_fg = psnr(fg_img, gt_img, mask=gt_mask)
                fg_psnr_full = psnr(fg_img, gt_img)
                # BG-only render evaluated only at BG region
                if bg_img is not None:
                    bg_psnr_bg = psnr(bg_img, gt_img, mask=(1.0 - gt_mask))
                else:
                    bg_psnr_bg = float("nan")
                fg_psnrs.append(fg_psnr_fg)
                bg_psnrs.append(bg_psnr_bg)
                fg_full.append(fg_psnr_full)
                per_frame.append({
                    "t": t, "cam": c,
                    "fg_only_psnr_at_fg": fg_psnr_fg,
                    "fg_only_psnr_full": fg_psnr_full,
                    "bg_only_psnr_at_bg": bg_psnr_bg,
                })
                print(f"  t={t:2d} cam{c}: FG-only@FG={fg_psnr_fg:5.2f} | FG-only@full={fg_psnr_full:5.2f} | BG-only@BG={bg_psnr_bg:5.2f}")

                if args.save_previews:
                    Path(args.save_previews).mkdir(parents=True, exist_ok=True)
                    if t in [0, 30, 50] and c == 0:
                        from PIL import Image as PI
                        side = np.concatenate([
                            (gt_img * 255).astype(np.uint8),
                            (fg_img * 255).astype(np.uint8),
                            (bg_img * 255).astype(np.uint8) if bg_img is not None else (gt_img * 255).astype(np.uint8),
                        ], axis=1)
                        PI.fromarray(side).save(f"{args.save_previews}/t{t:02d}_cam{c}_gt_fg_bg.png")
            except Exception as e:
                print(f"  t={t} cam{c}: FAILED — {e}")

    avg_fg = float(np.nanmean(fg_psnrs))
    avg_bg = float(np.nanmean(bg_psnrs))
    avg_fg_full = float(np.nanmean(fg_full))

    print(f"\n  >> FG-only render @ FG region:  avg PSNR = {avg_fg:.2f}")
    print(f"  >> FG-only render @ full image: avg PSNR = {avg_fg_full:.2f}  (full = mostly white BG)")
    print(f"  >> BG-only render @ BG region:  avg PSNR = {avg_bg:.2f}")

    # Interpretation
    print()
    print("=== INTERPRETATION ===")
    if avg_fg > 18:
        print(f"  FG-only renders the mouse well (FG PSNR={avg_fg:.2f} > 18).")
        print("  → FG Gaussians have learned the mouse. Any visual ghost in combined render")
        print("    is due to BG bleeding into the displayed image, NOT due to FG itself.")
        print("  → Fix direction: hard mask BG opacity at FG region, OR exclude BG from FG region preprocessing.")
    elif avg_fg > 14:
        print(f"  FG-only is partial (PSNR={avg_fg:.2f}). FG learned a rough mouse but not detail.")
        print("  → Both FG architecture AND BG interference are issues.")
        print("  → Fix: combine V10a's loss balance with motion bases improvement.")
    else:
        print(f"  FG-only is poor (PSNR={avg_fg:.2f} < 14). FG itself is not learning the mouse.")
        print("  → BG interference is NOT the only issue.")
        print("  → Fix: investigate FG initialization, motion bases, or feature quality.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "fg_only_psnr_at_fg_avg": avg_fg,
            "fg_only_psnr_full_avg": avg_fg_full,
            "bg_only_psnr_at_bg_avg": avg_bg,
            "per_frame": per_frame,
        }, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
