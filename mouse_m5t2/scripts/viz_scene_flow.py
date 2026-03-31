"""
Post-training 3D scene flow visualization for MonoFusion M5t2.

Extracts per-Gaussian scene flow vectors from a trained checkpoint using
MonoFusion's SE(3) motion bases (compute_poses_fg), then renders:
  - Figure A: 3-view magnitude heatmap (isometric / top / front)
  - Figure B: 2D projection error vs RAFT flow (verification)
  - Figure C: Trajectory trails for top-K Gaussians

No Open3D / GUI required: outputs are static .png files.

Usage (on gpu03, monofusion_pytorch env):
    python viz_scene_flow.py \\
        --checkpoint /node_data/joon/data/monofusion/m5t2_poc/ckpt/latest.pt \\
        --data_root  /node_data/joon/data/monofusion/m5t2_poc \\
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/viz \\
        --query_t 0 --target_t 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed on server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch


# ──────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: Path, device: str):
    """Load MonoFusion model from checkpoint.

    MonoFusion saves the full trainer state. We extract:
      - model.motion_bases.params
      - model.gaussians.params
    """
    state = torch.load(ckpt_path, map_location=device)

    # MonoFusion saves state_dict under different keys depending on version
    for key in ("model", "model_state_dict", "state_dict"):
        if key in state:
            return state[key]

    # If no wrapper key: the state IS the model dict
    return state


def extract_means_at_frame(model_state: dict, frame_t: int, device: str) -> np.ndarray:
    """Compute per-Gaussian canonical positions at time t using SE(3) motion bases.

    Implements: means_i(t) = SE3_combined_i(t) @ [canonical_μ_i; 1]

    Args:
        model_state: raw state dict from checkpoint
        frame_t:     target frame index (0-indexed)
    Returns:
        means: [G, 3] numpy array of world-space Gaussian positions at t
    """
    import torch.nn.functional as F

    # Extract parameters
    canonical_means = _get_param(model_state, "gaussians.means", "means")  # (G, 3)
    motion_coefs_raw = _get_param(model_state, "gaussians.motion_coefs", "motion_coefs")  # (G, K)
    rots   = _get_param(model_state, "motion_bases.rots",   "rots")    # (K, T, 6)
    transls = _get_param(model_state, "motion_bases.transls", "transls")  # (K, T, 3)

    if canonical_means is None:
        raise KeyError(
            "Could not find Gaussian means in checkpoint. "
            "Keys available: " + str([k for k in model_state.keys() if "means" in k or "motion" in k])
        )

    canonical_means  = torch.tensor(canonical_means,  device=device, dtype=torch.float32)
    motion_coefs_raw = torch.tensor(motion_coefs_raw, device=device, dtype=torch.float32)
    rots   = torch.tensor(rots,    device=device, dtype=torch.float32)
    transls = torch.tensor(transls, device=device, dtype=torch.float32)

    G, K = motion_coefs_raw.shape
    coefs = F.softmax(motion_coefs_raw, dim=-1)  # (G, K)

    # SE3_k(t) from 6D rotation + translation
    rot_at_t   = rots[:, frame_t, :]    # (K, 6)
    transl_at_t = transls[:, frame_t, :]  # (K, 3)

    rot_mats = rot6d_to_matrix(rot_at_t)  # (K, 3, 3)

    # Weighted combination: SE3_combined_i = Σ_k coefs[i,k] * SE3_k(t)
    # rot_combined[i] = Σ_k coefs[i,k] * rot_mats[k]  → (G, 3, 3)
    rot_combined   = torch.einsum("gk,kij->gij", coefs, rot_mats)    # (G, 3, 3)
    transl_combined = torch.einsum("gk,ki->gi",  coefs, transl_at_t)  # (G, 3)

    # Apply: means_i(t) = rot_combined_i @ canonical_μ_i + transl_combined_i
    means_t = torch.einsum("gij,gj->gi", rot_combined, canonical_means) + transl_combined  # (G, 3)

    return means_t.cpu().numpy()


def rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D continuous rotation representation to 3x3 rotation matrix.

    Ref: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)

    Args:
        r6d: (N, 6) 6D rotation vectors
    Returns:
        R:   (N, 3, 3) rotation matrices
    """
    # First two columns of rotation matrix, then compute third via cross product
    a1 = r6d[:, :3]
    a2 = r6d[:, 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)


def _get_param(state: dict, *keys: str):
    """Try multiple key patterns (handles different checkpoint formats)."""
    for key in keys:
        if key in state:
            v = state[key]
            return v.numpy() if hasattr(v, "numpy") else np.array(v)
        # Prefix search
        for k in state:
            if k.endswith(key) or k.endswith("." + key.split(".")[-1]):
                v = state[k]
                return v.numpy() if hasattr(v, "numpy") else np.array(v)
    return None


# ──────────────────────────────────────────
# Figure A: Magnitude Heatmap (3 views)
# ──────────────────────────────────────────

def viz_magnitude_heatmap(
    means_t0: np.ndarray,
    means_t1: np.ndarray,
    output_path: Path,
    t0: int, t1: int,
):
    """3-panel 3D scatter plot colored by scene flow magnitude.

    Views: isometric (30°/45°), top-down, front.
    """
    delta = means_t1 - means_t0              # (G, 3)
    mag   = np.linalg.norm(delta, axis=1)    # (G,)

    # Subsample dense point clouds for clarity
    max_pts = 5000
    if len(means_t0) > max_pts:
        idx = np.random.choice(len(means_t0), max_pts, replace=False)
        pts, mag_s = means_t0[idx], mag[idx]
    else:
        pts, mag_s = means_t0, mag

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"3D Scene Flow Magnitude: F{t0} → F{t1}", fontsize=13)

    views = [(30, 45, "Isometric"), (90, -90, "Top (XY)"), (0, 0, "Front (XZ)")]
    for col, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=mag_s, cmap="viridis", s=2, alpha=0.7
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.tick_params(labelsize=6)

    plt.colorbar(sc, ax=fig.axes, label="|φ| scene flow magnitude", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────
# Figure B: 2D Projection Verification
# ──────────────────────────────────────────

def viz_projection_verification(
    means_t0: np.ndarray,
    means_t1: np.ndarray,
    data_root: Path,
    cam_name: str,
    frame_t: int,
    K_mat: np.ndarray,
    output_path: Path,
):
    """Compare 3D scene flow projected to 2D vs RAFT optical flow.

    Discrepancy reveals depth ambiguity or multi-view inconsistency.
    """
    import json

    delta = means_t1 - means_t0  # (G, 3) 3D flow
    z = means_t0[:, 2]           # depth of each Gaussian at t

    # Project 3D flow → 2D (pinhole approximation)
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    u_flow = fx * delta[:, 0] / (z + 1e-6)  # (G,)
    v_flow = fy * delta[:, 1] / (z + 1e-6)  # (G,)
    proj_uv = np.stack([u_flow, v_flow], axis=1)  # (G, 2) optical flow in pixels

    # Load RAFT flow for this camera and frame
    track_dir = data_root / "tapir" / cam_name
    frame_str = f"{frame_t:06d}"
    # Find query file closest to frame_t
    query_files = sorted(track_dir.glob(f"*_{frame_str}.npy")) if track_dir.exists() else []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"2D Projection Verification: {cam_name} F{frame_t}", fontsize=12)

    # Left: projected 3D flow magnitude (Gaussian positions projected to 2D)
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    u_pos = fx * means_t0[:, 0] / (z + 1e-6) + cx  # 2D positions of Gaussians
    v_pos = fy * means_t0[:, 1] / (z + 1e-6) + cy
    proj_mag = np.linalg.norm(proj_uv, axis=1)
    axes[0].scatter(u_pos, v_pos, c=proj_mag, cmap="viridis", s=1, alpha=0.5)
    axes[0].set_title("3D flow projected to 2D (magnitude)")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("u (px)"); axes[0].set_ylabel("v (px)")

    # Right: RAFT tracks at this frame (if available)
    if query_files:
        tracks = np.load(query_files[0])  # (N, 4)
        visible = tracks[:, 2] <= 0.0
        axes[1].scatter(tracks[visible, 0], tracks[visible, 1],
                        c="green", s=4, label=f"visible ({visible.sum()})", alpha=0.7)
        axes[1].scatter(tracks[~visible, 0], tracks[~visible, 1],
                        c="red", s=1, label=f"occluded ({(~visible).sum()})", alpha=0.3)
        axes[1].set_title(f"RAFT tracks at F{frame_t} ({query_files[0].name})")
        axes[1].legend(fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("u (px)"); axes[1].set_ylabel("v (px)")
    else:
        axes[1].text(0.5, 0.5, f"No RAFT tracks found\nfor {cam_name} F{frame_t}",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("RAFT tracks (not found)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────
# Figure C: Trajectory Trails
# ──────────────────────────────────────────

def viz_trajectory_trails(
    all_means: np.ndarray,
    output_path: Path,
    n_trails: int = 100,
    fg_only_mask: np.ndarray = None,
):
    """3D trajectory trails for top-K Gaussians sorted by total displacement.

    Args:
        all_means:   (T, G, 3) per-frame Gaussian positions
        n_trails:    number of trails to plot (top by total motion)
        fg_only_mask: (G,) boolean mask to restrict to FG Gaussians
    """
    T, G, _ = all_means.shape

    if fg_only_mask is not None:
        all_means = all_means[:, fg_only_mask, :]
        G = all_means.shape[1]

    # Sort by total path length
    path_lengths = np.sum(np.linalg.norm(np.diff(all_means, axis=0), axis=-1), axis=0)  # (G,)
    top_idx = np.argsort(path_lengths)[-n_trails:]

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"Trajectory Trails (top {n_trails} by path length, T={T})", fontsize=12)

    color_map = cm.plasma(np.linspace(0, 1, T))

    for col, (elev, azim, title) in enumerate([(30, 45, "Isometric"), (90, -90, "Top")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for i in top_idx:
            traj = all_means[:, i, :]  # (T, 3)
            # Draw segments colored by time
            for t in range(T - 1):
                ax.plot(
                    traj[t:t+2, 0], traj[t:t+2, 1], traj[t:t+2, 2],
                    color=color_map[t], linewidth=0.6, alpha=0.7
                )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.tick_params(labelsize=6)

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, T-1))
    sm.set_array([])
    plt.colorbar(sm, ax=fig.axes, label="Frame index", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-training 3D scene flow visualization")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root",  type=Path, required=True, help="M5t2 PoC data root")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save .png outputs")
    parser.add_argument("--query_t",    type=int, default=0,  help="Source frame for scene flow")
    parser.add_argument("--target_t",   type=int, default=30, help="Target frame for scene flow")
    parser.add_argument("--all_frames", action="store_true",  help="Extract all T frames for trajectory trails")
    parser.add_argument("--device",     type=str, default="cpu", help="torch device (cpu / cuda:N)")
    parser.add_argument("--cam_name",   type=str, default="m5t2_cam00", help="Camera for 2D projection check")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"CRITICAL: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model_state = load_model_from_checkpoint(args.checkpoint, args.device)

    # ── Figure A: Magnitude heatmap ──
    print(f"\n[A] Computing scene flow F{args.query_t} → F{args.target_t}...")
    means_q = extract_means_at_frame(model_state, args.query_t, args.device)
    means_t = extract_means_at_frame(model_state, args.target_t, args.device)
    print(f"    Gaussians: {len(means_q)}, flow max: {np.linalg.norm(means_t - means_q, axis=1).max():.4f}")
    viz_magnitude_heatmap(
        means_q, means_t,
        args.output_dir / f"sceneflow_A_magnitude_f{args.query_t}_f{args.target_t}.png",
        args.query_t, args.target_t,
    )

    # ── Figure B: 2D projection verification ──
    print(f"\n[B] 2D projection verification ({args.cam_name})...")
    import json
    meta_path = args.data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    K_mat = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        Ks = np.array(meta["k"])  # [T, C, 3, 3]
        cam_idx = int(args.cam_name.split("cam")[-1].lstrip("0") or "0")
        K_mat = Ks[args.query_t, cam_idx]
    if K_mat is not None:
        viz_projection_verification(
            means_q, means_t,
            args.data_root, args.cam_name, args.query_t, K_mat,
            args.output_dir / f"sceneflow_B_projection_{args.cam_name}_f{args.query_t}.png",
        )
    else:
        print(f"  WARNING: meta JSON not found at {meta_path} — skipping Figure B")

    # ── Figure C: Trajectory trails ──
    if args.all_frames:
        print(f"\n[C] Extracting all frames for trajectory trails...")
        # Load T from meta
        T = len(meta["k"]) if meta_path.exists() else 60
        all_means = []
        for t in range(T):
            all_means.append(extract_means_at_frame(model_state, t, args.device))
            if t % 10 == 0:
                print(f"    frame {t}/{T}")
        all_means = np.stack(all_means, axis=0)  # (T, G, 3)
        viz_trajectory_trails(
            all_means,
            args.output_dir / "sceneflow_C_trails.png",
        )

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
