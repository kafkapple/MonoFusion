"""Render FG-only track video: background darkened, tracks on mouse only."""
import numpy as np
from pathlib import Path
from PIL import Image
import imageio
import matplotlib.pyplot as plt

data_root = Path("/node_data/joon/data/monofusion/m5t2_poc")
cam = "m5t2_cam00"
frames = sorted([p.stem for p in (data_root / "images" / cam).glob("??????.png")])
T = len(frames)

track_root = data_root / "tapir" / cam
raw_tracks = []
for fname in frames:
    tp = track_root / f"{frames[0]}_{fname}.npy"
    raw_tracks.append(np.load(tp) if tp.exists() else None)

N = raw_tracks[0].shape[0]
vis = np.zeros(N)
for r in raw_tracks:
    if r is not None and r.shape[0] == N:
        vis += (r[:, 2] <= 0).astype(float)
top = np.argsort(-vis)[:30]
print(f"Top 30 visibility: min={vis[top].min():.0f}/{T}, max={vis[top].max():.0f}/{T}")

colors = (plt.cm.hsv(np.linspace(0, 0.9, 30))[:, :3] * 255).astype(np.uint8)
H, W = 512, 512

writer = imageio.get_writer(
    str(data_root / "viz" / "vid_04_tracks_fg_only.mp4"), fps=15, quality=8)

for fi in range(T):
    img = np.array(Image.open(data_root / "images" / cam / f"{frames[fi]}.png"))[:, :, :3].copy()

    # Darken background using FG mask
    mask_path = data_root / "masks" / cam / f"{frames[fi]}.npz"
    if mask_path.exists():
        mask = np.load(mask_path)["dyn_mask"]
        bg = mask < 0.5
        img[bg] = (img[bg] * 0.15).astype(np.uint8)

    # Draw tracks
    if raw_tracks[fi] is not None and raw_tracks[fi].shape[0] == N:
        for ti, pi in enumerate(top):
            occ = raw_tracks[fi][pi, 2]
            if occ <= 0:
                x = int(np.clip(raw_tracks[fi][pi, 0], 0, W - 1))
                y = int(np.clip(raw_tracks[fi][pi, 1], 0, H - 1))
                r = 4
                y1, y2 = max(0, y - r), min(H, y + r + 1)
                x1, x2 = max(0, x - r), min(W, x + r + 1)
                img[y1:y2, x1:x2] = colors[ti]

                # Trail (last 8 frames)
                for dt in range(1, min(8, fi + 1)):
                    prev = raw_tracks[fi - dt]
                    if prev is not None and prev.shape[0] == N and prev[pi, 2] <= 0:
                        px = int(np.clip(prev[pi, 0], 0, W - 1))
                        py = int(np.clip(prev[pi, 1], 0, H - 1))
                        tr = 1
                        img[max(0, py - tr):min(H, py + tr + 1),
                            max(0, px - tr):min(W, px + tr + 1)] = colors[ti]

    writer.append_data(img)

writer.close()
print("Done: vid_04_tracks_fg_only.mp4")
