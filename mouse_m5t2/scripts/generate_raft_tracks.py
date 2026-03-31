"""
Option B: RAFT optical flow + per-frame mask validation for FG tracking.

Strategy (addresses audit findings):
  - Input: original RGB (no masking → no distribution shift for RAFT)
  - Per-frame SAM2 masks used as VALIDATION, not input modification
  - Query points seeded from frame-0 FG mask (same as original TAPNet)
  - Consecutive RAFT flows chained to track points over time
  - Points outside FG mask at frame t → marked occluded

Output format: MonoFusion TAPIR-compatible [N, 4] = [x, y, occ_logit, dist]
Saves {query_frame}_{target_frame}.npy for all target frames.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_raft_tracks.py \\
        --data_root /node_data/joon/data/monofusion/m5t2_poc \\
        --output_dir /node_data/joon/data/monofusion/m5t2_poc/tapir_raft \\
        --n_points 512
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from viz_utils import get_cam_dirs, get_frame_names


OCCLUDED_LOGIT = 10.0    # sigmoid(10) ≈ 1.0 (completely occluded)
VISIBLE_LOGIT  = -2.0   # sigmoid(-2) ≈ 0.12 (visible)
OOB_LOGIT      = 10.0   # out-of-bounds = occluded

# CRITICAL: expected_dist (track[N, 3]) controls confidence in parse_tapir_track_info:
#   confidence = 1 - sigmoid(expected_dist)
#   valid_visible = visibility * confidence > 0.5
# If expected_dist = 0.0 → confidence = 0.5 → valid_visible = 0.88 * 0.5 = 0.44 → NOT VISIBLE!
# Fix: expected_dist = -2.0 → confidence = 0.88 → valid_visible = 0.88 * 0.88 = 0.78 ✓
CONFIDENT_DIST = -2.0   # high-confidence marker for all RAFT tracks (visible or occluded)


def load_raft(device: str) -> tuple[torch.nn.Module, object]:
    """Load pretrained RAFT-small with official preprocessing transform.

    IMPORTANT: torchvision RAFT requires uint8 input + OpticalFlow() transform.
    Feeding float32 directly causes ~10x flow magnification errors.
    """
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    transform = weights.transforms()  # uint8 → float32 normalized [-1, 1]
    return model, transform


def load_frames_tensor(cam_dir: Path) -> tuple[torch.Tensor, list[str]]:
    """Load all frames as (T, 3, H, W) uint8 for RAFT preprocessing."""
    names = get_frame_names(cam_dir)
    imgs = []
    for name in names:
        img = np.array(Image.open(cam_dir / f"{name}.png").convert("RGB"))
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))  # uint8
    return torch.stack(imgs), names  # (T, 3, H, W) uint8


def load_masks(mask_dir: Path, names: list[str]) -> list[np.ndarray]:
    """Load per-frame binary FG masks. +1=FG in float32 [-1,1]."""
    masks = []
    for name in names:
        p = mask_dir / f"{name}.npz"
        if p.exists():
            raw = np.load(p)["dyn_mask"]
            mask = (raw > 0)
            if mask.ndim == 3:
                mask = mask.squeeze(-1)
        else:
            mask = None
        masks.append(mask)
    return masks


@torch.no_grad()
def compute_consecutive_flows(
    model: torch.nn.Module,
    transform,
    frames: torch.Tensor,
    device: str,
    batch_size: int = 4,
) -> np.ndarray:
    """Compute forward optical flow for all consecutive frame pairs.

    Args:
        frames: (T, 3, H, W) uint8 tensor (CPU)
    Returns flows: (T-1, H, W, 2) as numpy, units=pixels.
    """
    T = frames.shape[0]
    all_flows = []

    for i in tqdm(range(0, T - 1, batch_size), desc="  RAFT flow", leave=False):
        batch_end = min(i + batch_size, T - 1)
        src = frames[i:batch_end]        # (B, 3, H, W) uint8
        dst = frames[i+1:batch_end+1]    # (B, 3, H, W) uint8

        # Apply official RAFT preprocessing: uint8 → float32 [-1, 1]
        src_t, dst_t = transform(src, dst)
        src_t, dst_t = src_t.to(device), dst_t.to(device)

        # RAFT returns list of flow predictions; take last (most refined)
        flow_preds = model(src_t, dst_t)
        flow = flow_preds[-1]  # (B, 2, H, W)

        # Convert to (B, H, W, 2) numpy
        flow_np = flow.permute(0, 2, 3, 1).cpu().numpy()
        all_flows.append(flow_np)

    return np.concatenate(all_flows, axis=0)  # (T-1, H, W, 2)


def sample_query_points(mask: np.ndarray, n_points: int) -> np.ndarray:
    """Sample query points from FG mask. Returns [N, 2] as [x, y]."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((n_points, 2), dtype=np.float32)

    if len(ys) >= n_points:
        idx = np.random.choice(len(ys), n_points, replace=False)
    else:
        idx = np.random.choice(len(ys), n_points, replace=True)

    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def propagate_tracks(
    query_xy: np.ndarray,       # [N, 2] float32 [x, y]
    flows: np.ndarray,          # [T-1, H, W, 2] float32
    masks: list[np.ndarray],    # per-frame FG masks
    query_frame: int,
    H: int, W: int,
) -> np.ndarray:
    """Propagate query points using chained optical flow.

    Returns track array [T, N, 4] = [x, y, occ_logit, dist].
    At each frame, sample flow at current point position using bilinear interpolation.
    Points outside FG mask → occ_logit = OCCLUDED_LOGIT.
    Points out of image bounds → occ_logit = OOB_LOGIT.
    """
    T = len(flows) + 1
    N = len(query_xy)
    tracks = np.zeros((T, N, 4), dtype=np.float32)
    tracks[:, :, 3] = CONFIDENT_DIST  # -2.0: high confidence → parse_tapir_track_info works

    # Current positions
    cur_xy = query_xy.copy()

    # Set query frame
    tracks[query_frame, :, 0] = cur_xy[:, 0]
    tracks[query_frame, :, 1] = cur_xy[:, 1]
    tracks[query_frame, :, 2] = VISIBLE_LOGIT  # visible at query frame

    def sample_flow(flow_hw2: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Sample flow at sub-pixel positions using bilinear interpolation."""
        # xy: [N, 2] = [x, y], flow_hw2: [H, W, 2]
        # normalize to [-1, 1] for grid_sample
        x_norm = 2.0 * xy[:, 0] / (W - 1) - 1.0
        y_norm = 2.0 * xy[:, 1] / (H - 1) - 1.0
        grid = torch.from_numpy(
            np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

        flow_t = torch.from_numpy(flow_hw2).permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        sampled = F.grid_sample(flow_t, grid, align_corners=True, mode="bilinear",
                                padding_mode="border")  # (1, 2, 1, N)
        return sampled.squeeze().permute(1, 0).numpy()  # (N, 2) = [dx, dy]

    def check_visibility(xy: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Returns occ_logit for each point: visible if in mask, else occluded."""
        # Bounds check
        in_bounds = ((xy[:, 0] >= 0) & (xy[:, 0] < W) &
                     (xy[:, 1] >= 0) & (xy[:, 1] < H))
        occ = np.where(in_bounds, VISIBLE_LOGIT, OOB_LOGIT)

        if mask is not None:
            xi = xy[:, 0].clip(0, W - 1).astype(int)
            yi = xy[:, 1].clip(0, H - 1).astype(int)
            in_mask = mask[yi, xi]
            occ = np.where(in_bounds & in_mask, VISIBLE_LOGIT, OCCLUDED_LOGIT)

        return occ.astype(np.float32)

    # Forward propagation: query_frame → T-1
    cur_xy = query_xy.copy()
    for t in range(query_frame, T - 1):
        delta = sample_flow(flows[t], cur_xy)
        cur_xy = cur_xy + delta
        occ = check_visibility(cur_xy, masks[t + 1])
        tracks[t + 1, :, 0] = cur_xy[:, 0]
        tracks[t + 1, :, 1] = cur_xy[:, 1]
        tracks[t + 1, :, 2] = occ

    # Backward propagation: query_frame-1 → 0 (use negative forward flow)
    cur_xy = query_xy.copy()
    for t in range(query_frame - 1, -1, -1):
        delta = sample_flow(flows[t], cur_xy)
        cur_xy = cur_xy - delta  # reverse direction
        occ = check_visibility(cur_xy, masks[t])
        tracks[t, :, 0] = cur_xy[:, 0]
        tracks[t, :, 1] = cur_xy[:, 1]
        tracks[t, :, 2] = occ

    return tracks  # [T, N, 4]


def save_tracks(
    tracks: np.ndarray,
    query_frame: int,
    frame_names: list[str],
    output_dir: Path,
):
    """Save [T, N, 4] as individual {query}_{target}.npy files."""
    query_name = frame_names[query_frame]
    T = len(frame_names)
    for t in range(T):
        target_name = frame_names[t]
        np.save(output_dir / f"{query_name}_{target_name}.npy", tracks[t])


def process_camera(
    cam_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    transform,
    n_points: int,
    query_frames: list[int],
    device: str,
    batch_size: int = 8,
) -> dict:
    """Process one camera end-to-end."""
    frames_t, names = load_frames_tensor(cam_dir)
    masks = load_masks(mask_dir, names)
    T, C, H, W = frames_t.shape

    print(f"    Computing RAFT flows: {T-1} pairs...")
    flows = compute_consecutive_flows(model, transform, frames_t, device, batch_size)

    stats = {"cam": cam_dir.name, "results": []}
    for q_idx in query_frames:
        if q_idx >= T:
            continue
        mask_q = masks[q_idx]
        if mask_q is None:
            print(f"    WARNING: no mask for query frame {q_idx}, skipping")
            continue

        query_xy = sample_query_points(mask_q, n_points)
        tracks = propagate_tracks(query_xy, flows, masks, q_idx, H, W)

        cam_out = output_dir / cam_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)
        save_tracks(tracks, q_idx, names, cam_out)

        # Stats: average visible ratio at each frame
        vis_per_frame = [(tracks[t, :, 2] <= 0.0).sum() for t in range(T)]
        mid_vis = vis_per_frame[T // 2]
        print(f"    Q={q_idx}: F0={vis_per_frame[0]}/{n_points}, "
              f"F{T//2}={mid_vis}/{n_points} ({mid_vis/n_points*100:.1f}%)")
        stats["results"].append({"q": q_idx, "vis_per_frame": vis_per_frame})

    return stats


def main():
    parser = argparse.ArgumentParser(description="RAFT optical flow FG track generation (Option B)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_points", type=int, default=512,
                        help="Number of query points per camera (default: 512)")
    parser.add_argument("--query_frames", type=int, nargs="+", default=[0],
                        help="Query frame indices (default: [0] for PoC)")
    parser.add_argument("--cameras", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Frames per RAFT batch")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading RAFT-small...")
    model, transform = load_raft(device)

    cam_dirs = get_cam_dirs(data_root)
    if args.cameras:
        cam_dirs = [d for d in cam_dirs if d.name in args.cameras]
    mask_root = data_root / "masks"

    for cam_dir in cam_dirs:
        print(f"\n  Camera: {cam_dir.name}")
        mask_dir = mask_root / cam_dir.name
        process_camera(
            cam_dir, mask_dir, output_dir, model, transform,
            n_points=args.n_points,
            query_frames=args.query_frames,
            device=device,
            batch_size=args.batch_size,
        )

    print(f"\nDone. RAFT tracks saved to: {output_dir}")


if __name__ == "__main__":
    main()
