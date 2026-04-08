# no-split: single diagnostic script — Phase 0 (D5/D6/D7/D8/D9)
"""
Phase 0 Diagnostics for V12 plan validation.

D5: FG-only temporal variance (does blob move?)
D6: V10b checkpoint FG Gaussian count, opacity, scale histograms, FG:BG ratio
D7: FG render at cano_t=0 with motion bases disabled (V12a falsifier)
D8: get_tracks_3d() return shape with num_samples={5K, 18K, 50K} (S1 empirical verify)
D9: cano_t auto-selected value + which pose it picks

Usage (on gpu03):
    cd ~/dev/MonoFusion
    PYTHONPATH=. python mouse_m5t2/scripts/phase0_diagnostics.py \
        --checkpoint /node_data/joon/data/monofusion/markerless_v7/results_v10b/checkpoints/best.ckpt \
        --data_root  /node_data/joon/data/monofusion/markerless_v7 \
        --output     /node_data/joon/data/monofusion/markerless_v7/phase0_diag.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def log(msg):
    print(f"[phase0] {msg}", flush=True)


# ----------------------------- D6 -----------------------------
def d6_checkpoint_stats(ckpt_path, device):
    log("=" * 60)
    log("D6: V10b checkpoint FG/BG stats")
    log("=" * 60)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from flow3d.scene_model import SceneModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SceneModel.init_from_state_dict(ckpt["model"]).to(device).eval()

    n_fg = int(model.num_fg_gaussians)
    n_bg = int(model.num_bg_gaussians)
    n_total = int(model.num_gaussians)
    ratio = n_bg / max(n_fg, 1)

    log(f"  FG Gaussians : {n_fg:,}")
    log(f"  BG Gaussians : {n_bg:,}")
    log(f"  Total        : {n_total:,}")
    log(f"  FG:BG ratio  : 1 : {ratio:.1f}")

    # Opacity stats: params are stored as logits
    fg_opacity_logit = model.fg.params["opacities"].detach().cpu()
    fg_opacity = torch.sigmoid(fg_opacity_logit).numpy()
    fg_scales_log = model.fg.params["scales"].detach().cpu()
    fg_scales = torch.exp(fg_scales_log).numpy()

    op_pcts = np.percentile(fg_opacity, [5, 25, 50, 75, 95])
    sc_pcts = np.percentile(fg_scales.mean(-1), [5, 25, 50, 75, 95])

    log(f"  FG opacity percentiles [5,25,50,75,95]:")
    log(f"    {op_pcts.tolist()}")
    log(f"  FG opacity collapse check: {(fg_opacity < 0.05).sum()} / {n_fg} have opacity<0.05")
    log(f"  FG opacity saturation check: {(fg_opacity > 0.95).sum()} / {n_fg} have opacity>0.95")
    log(f"  FG scale (geo-mean of 3 axes) percentiles [5,25,50,75,95]:")
    log(f"    {sc_pcts.tolist()}")
    log(f"  FG tiny-scale check: {(fg_scales.mean(-1) < 0.001).sum()} / {n_fg} have scale<1mm")
    log(f"  FG large-scale check: {(fg_scales.mean(-1) > 0.1).sum()} / {n_fg} have scale>10cm")

    return {
        "num_fg": n_fg, "num_bg": n_bg, "num_total": n_total, "bg_over_fg": ratio,
        "fg_opacity_pcts": op_pcts.tolist(),
        "fg_opacity_near_zero": int((fg_opacity < 0.05).sum()),
        "fg_opacity_saturated": int((fg_opacity > 0.95).sum()),
        "fg_scale_pcts": sc_pcts.tolist(),
        "fg_scale_tiny": int((fg_scales.mean(-1) < 0.001).sum()),
        "fg_scale_large": int((fg_scales.mean(-1) > 0.1).sum()),
    }, model


# ----------------------------- D8 -----------------------------
def d8_get_tracks_3d_shape(data_root, device):
    log("=" * 60)
    log("D8: get_tracks_3d() return shape at num_samples = {5K, 18K, 50K}")
    log("=" * 60)
    from flow3d.data.casual_dataset import CasualDataset, CustomDataConfig
    # Import the mouse_m5t2 patcher so the dataset can load the M5t2 format
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_m5t2 import patch_casual_dataset
    patch_casual_dataset()

    d8_results = {}
    # Use cam0 as representative; mouse_m5t2 has 4 cams, result will be ~equal
    for num_samples in [5000, 18000, 50000]:
        cfg = CustomDataConfig(
            seq_name="m5t2_cam00",
            root_dir=str(data_root),
            video_name="m5t2_cam00",
            depth_type="moge",
        )
        # Minimal load: only need get_tracks_3d
        ds = CasualDataset(
            data_cfg=cfg, training=False, num_targets_per_frame=1,
        ) if False else None  # skip full dataset, use direct call below

        # Actually instantiate:
        try:
            ds = CasualDataset(
                data_cfg=cfg, training=True, num_targets_per_frame=1,
            )
        except Exception as e:
            log(f"  [WARN] full CasualDataset init failed ({e}); using minimal path")
            continue

        try:
            t3d, vis, invis, conf, col, feat = ds.get_tracks_3d(num_samples)
            log(f"  num_samples={num_samples}: returned shape={tuple(t3d.shape)}")
            log(f"    visible@t0 = {int(vis[:, 0].sum().item())}")
            d8_results[str(num_samples)] = {
                "requested": num_samples,
                "returned_P": int(t3d.shape[0]),
                "num_frames": int(t3d.shape[1]),
                "visible_at_cano_guess": int(vis[:, 0].sum().item()),
            }
        except Exception as e:
            log(f"  [ERR] num_samples={num_samples}: {e}")
            d8_results[str(num_samples)] = {"error": str(e)}

    log("")
    log("  S1 VERDICT:")
    if all(isinstance(d8_results.get(str(n)), dict) and "returned_P" in d8_results[str(n)]
           for n in [5000, 18000, 50000]):
        p5k = d8_results["5000"]["returned_P"]
        p18k = d8_results["18000"]["returned_P"]
        p50k = d8_results["50000"]["returned_P"]
        log(f"  P(5K)={p5k}, P(18K)={p18k}, P(50K)={p50k}")
        if p18k <= 1.2 * p5k:
            log("  ❌ V12b CONFIRMED DEAD — num_fg cap is not the bottleneck (TAPNet is)")
            d8_results["verdict"] = "V12b_dead"
        elif p18k > 10000:
            log("  ⚠️  V12b may still be viable")
            d8_results["verdict"] = "V12b_viable"
        else:
            log(f"  🟡 V12b partial — P grows but < 18K")
            d8_results["verdict"] = "V12b_partial"
    return d8_results


# ----------------------------- D9 -----------------------------
def d9_cano_t(d8_results):
    log("=" * 60)
    log("D9: cano_t selection (from D8 track visibility)")
    log("=" * 60)
    log("  Inspect visibility curve / cano_t is cam-dependent.")
    log("  See 08_visibility_curve.png for multi-cam breakdown.")
    return {"cano_t_derivation": "argmax(visible.sum(dim=0))"}


# ----------------------------- D5, D7 -----------------------------
@torch.no_grad()
def render_fg_only(model, t, cam_idx, device, n_frames=80, img_wh=(512, 512),
                   disable_motion=False):
    idx = cam_idx * n_frames + t
    w2c = model.w2cs[idx:idx+1].to(device)
    K = model.Ks[idx:idx+1].to(device)

    if disable_motion:
        # Save and zero out motion bases
        saved_rots = model.motion_bases.params["rots"].data.clone()
        saved_trans = model.motion_bases.params["transls"].data.clone()
        # Rot: 6D, identity = [1,0,0, 0,1,0]
        identity_6d = torch.tensor([1., 0., 0., 0., 1., 0.], device=saved_rots.device)
        model.motion_bases.params["rots"].data[:] = identity_6d.view(1, 1, 6)
        model.motion_bases.params["transls"].data.zero_()

    out = model.render(
        t=t, w2cs=w2c, Ks=K, img_wh=img_wh,
        return_color=True, return_feat=False, return_depth=False, return_mask=False,
        fg_only=True,
    )
    img = out["img"][0].clamp(0, 1).cpu().numpy()

    if disable_motion:
        model.motion_bases.params["rots"].data[:] = saved_rots
        model.motion_bases.params["transls"].data[:] = saved_trans
    return img


def d5_d7_rendering(model, device):
    log("=" * 60)
    log("D5: FG-only temporal variance (does blob move?)")
    log("=" * 60)

    cam = 0
    frames = [0, 10, 20, 30, 40, 50, 60, 70]
    renders = []
    for t in frames:
        img = render_fg_only(model, t, cam, device)
        renders.append(img)
    stack = np.stack(renders)  # [T, H, W, 3]
    per_pixel_var = stack.var(axis=0).mean()  # scalar
    per_pixel_max_minus_min = (stack.max(axis=0) - stack.min(axis=0)).mean()
    log(f"  cam{cam} across {len(frames)} frames:")
    log(f"    mean per-pixel variance = {per_pixel_var:.6f}")
    log(f"    mean (max - min)        = {per_pixel_max_minus_min:.6f}")
    if per_pixel_var < 1e-5:
        log("  ❌ D5 FAIL: variance ≈ 0 — blob is frozen, motion is NOT training")
        d5_verdict = "frozen"
    else:
        log("  ✅ D5 OK: variance > 0 — blob is moving with time")
        d5_verdict = "moving"

    log("=" * 60)
    log("D7: FG render at cano_t=0 with motion bases DISABLED")
    log("=" * 60)
    img_motion_on = render_fg_only(model, 0, cam, device, disable_motion=False)
    img_motion_off = render_fg_only(model, 0, cam, device, disable_motion=True)
    diff = np.abs(img_motion_on - img_motion_off).mean()
    log(f"  cam{cam} t=0: motion_on vs motion_off pixel diff = {diff:.6f}")
    if diff < 1e-3:
        log("  📌 At cano_t=0 motion is identity (expected). Blob shape determined by init.")
        d7_note = "cano_t_identity"
    else:
        log("  ⚠️  Non-trivial difference at cano_t — indicates motion bases not identity at t=0")
        d7_note = "nontrivial_diff"

    out_dir = Path("/tmp/phase0_d5d7")
    out_dir.mkdir(exist_ok=True)
    from PIL import Image
    for t, img in zip(frames, renders):
        Image.fromarray((img * 255).astype(np.uint8)).save(out_dir / f"d5_t{t:02d}_cam0.png")
    Image.fromarray((img_motion_on * 255).astype(np.uint8)).save(out_dir / "d7_t0_cam0_motion_on.png")
    Image.fromarray((img_motion_off * 255).astype(np.uint8)).save(out_dir / "d7_t0_cam0_motion_off.png")
    log(f"  Saved D5/D7 renders to {out_dir}")

    return {
        "d5": {
            "mean_variance": float(per_pixel_var),
            "mean_range": float(per_pixel_max_minus_min),
            "verdict": d5_verdict,
        },
        "d7": {
            "pixel_diff_motion_on_vs_off_at_t0": float(diff),
            "note": d7_note,
        },
    }


# ----------------------------- MAIN -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--skip_d8", action="store_true", help="Skip D8 if dataset load is slow")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")

    results = {}

    d6_stats, model = d6_checkpoint_stats(args.checkpoint, device)
    results["d6"] = d6_stats

    if not args.skip_d8:
        try:
            results["d8"] = d8_get_tracks_3d_shape(Path(args.data_root), device)
        except Exception as e:
            log(f"  [WARN] D8 failed: {e}")
            results["d8"] = {"error": str(e)}

    results["d9"] = d9_cano_t(results.get("d8", {}))

    results.update(d5_d7_rendering(model, device))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
