"""
Five pre-training validation check functions for MonoFusion M5t2.
Called by validate_training_inputs.py.
"""
import json
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────
# Check 1: Coordinate System (Camera Reprojection)
# ─────────────────────────────────────────────

def check_coordinate_system(data_root: Path) -> list[str]:
    """Verify camera K + w2c matrices allow correct 2D→3D→2D round-trip.

    Strategy:
      - Load a visible RAFT track point (u, v) for cam00 at F0
      - Un-project using depth + K + w2c to get world-space 3D point
      - Re-project back to cam00 → should recover (u, v) within 1px
      - Project to cam01/cam03 → check point is within image bounds
    """
    issues = []
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    if not meta_path.exists():
        issues.append(f"CRITICAL [coord]: meta JSON not found: {meta_path}")
        return issues

    with open(meta_path) as f:
        meta = json.load(f)

    Ks   = np.array(meta["k"])    # [T, C, 3, 3]
    w2cs = np.array(meta["w2c"])  # [T, C, 4, 4]
    T, C = Ks.shape[:2]

    track_dir = data_root / "tapir" / "m5t2_cam00"
    if not track_dir.exists():
        issues.append(f"WARNING [coord]: tapir/m5t2_cam00 not found — skipping reprojection check")
        return issues

    track_files = sorted(track_dir.glob("000000_*.npy"))
    if not track_files:
        issues.append(f"WARNING [coord]: no F0 tracks in m5t2_cam00")
        return issues

    tracks_f0 = np.load(track_files[0])  # [N, 4]: x, y, occ_logit, dist
    visible = tracks_f0[:, 2] <= 0.0
    if visible.sum() == 0:
        issues.append(f"WARNING [coord]: no visible tracks at F0 — cannot verify reprojection")
        return issues

    idx = np.where(visible)[0][0]
    u, v = tracks_f0[idx, 0], tracks_f0[idx, 1]

    depth_path = data_root / "aligned_moge_depth" / "m5t2" / "m5t2_cam00" / "depth" / "000000.npy"
    if not depth_path.exists():
        issues.append(f"WARNING [coord]: depth not found — skipping reprojection check")
        return issues

    depth = np.load(depth_path)  # [H, W]
    H, W = depth.shape
    ui, vi = int(np.clip(u, 0, W-1)), int(np.clip(v, 0, H-1))
    z = float(depth[vi, ui])
    if z <= 0:
        issues.append(f"WARNING [coord]: depth=0 at ({ui},{vi}) — choose different point")
        return issues

    K   = Ks[0, 0]    # [3, 3] cam00 F0
    w2c = w2cs[0, 0]  # [4, 4] cam00 F0
    c2w = np.linalg.inv(w2c)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pt_cam = np.array([x_cam, y_cam, z, 1.0])

    pt_world = c2w @ pt_cam
    pt_cam_reproj = w2c @ pt_world
    u_reproj = K[0,0] * pt_cam_reproj[0] / pt_cam_reproj[2] + K[0,2]
    v_reproj = K[1,1] * pt_cam_reproj[1] / pt_cam_reproj[2] + K[1,2]

    reproj_err = np.sqrt((u - u_reproj)**2 + (v - v_reproj)**2)
    if reproj_err > 1.0:
        issues.append(f"CRITICAL [coord]: reprojection round-trip error = {reproj_err:.3f}px (threshold 1px)")
    else:
        print(f"  [coord] ✓ cam00 reprojection round-trip error: {reproj_err:.4f}px")

    for cam_idx, cam_name in [(1, "cam01"), (3, "cam03")]:
        K_c = Ks[0, cam_idx]
        w2c_c = w2cs[0, cam_idx]
        pt_cam_c = w2c_c @ pt_world
        if pt_cam_c[2] <= 0:
            issues.append(f"WARNING [coord]: world point behind {cam_name} at F0 — coordinate system may be wrong")
            continue
        u_c = K_c[0,0] * pt_cam_c[0] / pt_cam_c[2] + K_c[0,2]
        v_c = K_c[1,1] * pt_cam_c[1] / pt_cam_c[2] + K_c[1,2]
        in_frame = 0 <= u_c < W and 0 <= v_c < H
        print(f"  [coord] {cam_name}: world point projects to ({u_c:.0f}, {v_c:.0f}) — {'in frame ✓' if in_frame else 'outside frame ⚠'}")

    return issues


# ─────────────────────────────────────────────
# Check 2: Frame Index Alignment
# ─────────────────────────────────────────────

def check_frame_index_alignment(data_root: Path) -> list[str]:
    """Verify track filenames match expected 6-digit frame convention.

    casual_dataset.py requests tracks by frame_name (e.g. '000000').
    Track files are named {query}_{target}.npy (6-digit).
    An off-by-one (e.g. 5-digit) causes silent empty-track batches.
    """
    issues = []
    track_dir = data_root / "tapir" / "m5t2_cam01"
    if not track_dir.exists():
        issues.append(f"WARNING [index]: tapir/m5t2_cam01 not found")
        return issues

    files = sorted(track_dir.glob("*.npy"))
    if not files:
        issues.append(f"CRITICAL [index]: no track files in {track_dir}")
        return issues

    bad_names = [f.name for f in files[:5] if not (
        len(f.stem.split("_")) == 2 and
        all(len(p) == 6 and p.isdigit() for p in f.stem.split("_"))
    )]
    if bad_names:
        issues.append(f"CRITICAL [index]: track filenames not 6-digit pattern: {bad_names[:3]}")
    else:
        print(f"  [index] ✓ track file naming: 6-digit pattern confirmed (e.g. {files[0].name})")

    expected = 300
    actual = len(files)
    if actual != expected:
        issues.append(f"WARNING [index]: expected {expected} track files, found {actual}")
    else:
        print(f"  [index] ✓ track count: {actual}/{expected} ✓")

    image_dir = data_root / "images" / "m5t2_cam01"
    if image_dir.exists():
        frame_names = sorted([p.stem for p in image_dir.glob("??????.png")])
        expected_queries = [frame_names[i] for i in [0, 15, 30, 45, 59] if i < len(frame_names)]
        for q in expected_queries:
            q_files = list(track_dir.glob(f"{q}_*.npy"))
            if len(q_files) == 0:
                issues.append(f"CRITICAL [index]: no track files for query frame {q}")
            elif len(q_files) != 60:
                issues.append(f"WARNING [index]: query {q} has {len(q_files)} files, expected 60")
        if not issues:
            print(f"  [index] ✓ all 5 query frames present with 60 targets each")

    return issues


# ─────────────────────────────────────────────
# Check 3: occ_logit / expected_dist Interpretation
# ─────────────────────────────────────────────

def check_track_interpretation(data_root: Path) -> list[str]:
    """Verify RAFT track values → parse_tapir_track_info → valid_visible=True.

    parse_tapir_track_info math:
      visibility = 1 - sigmoid(occ_logit)
      confidence = 1 - sigmoid(expected_dist)
      valid_visible = visibility * confidence > 0.5

    Required: occ_logit=-2.0, expected_dist=-2.0:
      visibility = 0.88, confidence = 0.88
      valid_visible = 0.88 * 0.88 = 0.77 > 0.5 ✓

    Bug (old): expected_dist=0.0:
      confidence = 0.5 → valid_visible = 0.44 → NOT VISIBLE!
    """
    issues = []

    track_dir = data_root / "tapir" / "m5t2_cam00"
    files = sorted(track_dir.glob("000000_000000.npy")) if track_dir.exists() else []
    if not files:
        track_dir2 = data_root / "tapir" / "m5t2_cam01"
        files = sorted(track_dir2.glob("*.npy"))[:1] if track_dir2.exists() else []

    if not files:
        issues.append(f"WARNING [occ]: no track files found — skipping interpretation check")
        return issues

    tracks = np.load(files[0])  # [N, 4]
    occ_logits = torch.from_numpy(tracks[:, 2])
    expected_dists = torch.from_numpy(tracks[:, 3])

    visibility   = 1 - torch.sigmoid(occ_logits)
    confidence   = 1 - torch.sigmoid(expected_dists)
    valid_visible = (visibility * confidence > 0.5)
    valid_invisible = ((1 - visibility) * confidence > 0.5)

    n_visible   = valid_visible.sum().item()
    n_invisible = valid_invisible.sum().item()
    n_total     = len(tracks)
    n_neither   = n_total - n_visible - n_invisible

    print(f"  [occ] Track interpretation check ({files[0].name}):")
    print(f"        visible:   {n_visible}/{n_total} ({n_visible/n_total*100:.1f}%)")
    print(f"        invisible: {n_invisible}/{n_total} ({n_invisible/n_total*100:.1f}%)")
    print(f"        neither:   {n_neither}/{n_total} ({n_neither/n_total*100:.1f}%)")

    visible_raw = (tracks[:, 2] <= 0.0)
    if visible_raw.sum() > 0:
        vis_after_parse = valid_visible[visible_raw].sum().item()
        vis_raw_count = int(visible_raw.sum())
        ratio = vis_after_parse / vis_raw_count
        if ratio < 0.5:
            issues.append(
                f"CRITICAL [occ]: {vis_after_parse}/{vis_raw_count} raw-visible tracks "
                f"pass parse_tapir_track_info ({ratio*100:.0f}%). "
                f"expected_dist values may be 0.0 instead of -2.0. "
                f"Regenerate tracks with generate_raft_tracks.py (CONFIDENT_DIST=-2.0)."
            )
        else:
            print(f"  [occ] ✓ {vis_after_parse}/{vis_raw_count} raw-visible tracks → valid_visible ✓")

    if n_neither / n_total > 0.3:
        issues.append(
            f"WARNING [occ]: {n_neither/n_total*100:.0f}% tracks are neither visible nor invisible "
            f"(confidence too low). Check expected_dist values."
        )

    return issues


# ─────────────────────────────────────────────
# Check 4: Scene Scale
# ─────────────────────────────────────────────

def check_scene_scale(data_root: Path) -> list[str]:
    """Verify camera translations and depth values are on compatible scales.

    MonoFusion normalizes scene internally (scene_scale in GaussianParams).
    But if cameras are in mm-scale while depths are in m-scale (or vice versa),
    init_utils.py will produce wrong 3D point cloud initializations.
    """
    issues = []
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    if not meta_path.exists():
        issues.append(f"WARNING [scale]: meta JSON not found")
        return issues

    with open(meta_path) as f:
        meta = json.load(f)

    w2cs = np.array(meta["w2c"])  # [T, C, 4, 4]
    c2ws = np.linalg.inv(w2cs)

    translations = c2ws[:, :, :3, 3]  # [T, C, 3]
    trans_norm = np.linalg.norm(translations, axis=-1)  # [T, C]
    trans_mean = trans_norm.mean()
    trans_max  = trans_norm.max()

    print(f"  [scale] Camera translations: mean_norm={trans_mean:.3f}, max_norm={trans_max:.3f}")

    if trans_mean > 100:
        issues.append(
            f"WARNING [scale]: camera translations are large (mean={trans_mean:.1f}). "
            f"May need scene normalization. Check init_utils.py for scene_scale handling."
        )

    depth_files = sorted((data_root / "aligned_moge_depth" / "m5t2" / "m5t2_cam00" / "depth").glob("*.npy"))
    if depth_files:
        depth_sample = np.load(depth_files[0])
        valid_depths = depth_sample[depth_sample > 0]
        if len(valid_depths) > 0:
            depth_mean = valid_depths.mean()
            depth_max  = valid_depths.max()
            print(f"  [scale] Depth (cam00 F0): mean={depth_mean:.3f}, max={depth_max:.3f}")

            ratio = trans_mean / (depth_mean + 1e-6)
            if ratio > 100 or ratio < 0.01:
                issues.append(
                    f"WARNING [scale]: camera translation scale ({trans_mean:.2f}) vs depth scale "
                    f"({depth_mean:.2f}) differ by {ratio:.0f}x. "
                    f"Verify DUSt3R and MoGe output the same unit (meters)."
                )
            else:
                print(f"  [scale] ✓ camera/depth scale ratio: {ratio:.2f} (within reasonable range)")

    return issues


# ─────────────────────────────────────────────
# Check 5: cam02 Empty Handling
# ─────────────────────────────────────────────

def check_empty_camera(data_root: Path) -> list[str]:
    """Verify cam02 (0% FG) doesn't crash training.

    The empty-camera issue: casual_dataset.py tries to load tracks for all cameras.
    If cam02 has only occluded tracks (occ_logit=+10), the track loss batch is empty.
    Some loss implementations divide by n_visible → NaN if 0.
    """
    issues = []
    track_dir = data_root / "tapir" / "m5t2_cam02"
    if not track_dir.exists():
        issues.append(f"WARNING [cam02]: tapir/m5t2_cam02 not found")
        return issues

    files = sorted(track_dir.glob("000000_*.npy"))
    if not files:
        print(f"  [cam02] ⚠ no F0 tracks for cam02 — may be empty")
        return issues

    mid_files = sorted(track_dir.glob("*_000030.npy"))
    if not mid_files:
        mid_files = files[:1]

    tracks = np.load(mid_files[0])  # [N, 4]
    visible = (tracks[:, 2] <= 0.0).sum()
    print(f"  [cam02] Track {mid_files[0].name}: {visible}/{len(tracks)} visible")
    if visible == 0:
        print(f"  [cam02] ✓ cam02 has 0 visible tracks (expected: mouse left FOV)")
        print(f"          Ensure training loss handles empty-track batches (no division by 0)")
    else:
        issues.append(
            f"WARNING [cam02]: expected 0 visible tracks at F30, found {visible}. "
            f"Verify mask is correct."
        )

    return issues
