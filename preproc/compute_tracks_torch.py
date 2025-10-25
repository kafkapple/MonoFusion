import argparse
import glob
import os
from pathlib import Path

import imageio.v2 as imageio
import cv2
import mediapy as media
import numpy as np
import torch
from tapnet_torch import tapir_model, transforms
from tqdm import tqdm


def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video


def _candidate_mask_dirs(requested: Path, image_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    _add(requested)
    _add(requested.parent / image_dir.name)
    _add(requested.parent / "sam_v2_dyn_mask" / image_dir.name)
    _add(requested.parent / "masks" / image_dir.name)

    dataset_root = image_dir.parent.parent
    _add(dataset_root / "sam_v2_dyn_mask" / image_dir.name)
    _add(dataset_root / "masks" / image_dir.name)

    return candidates


def _resolve_mask_dir(mask_dir: str, image_dir: str) -> Path:
    mask_path = Path(mask_dir)
    image_path = Path(image_dir).resolve()

    if mask_path.exists():
        return mask_path

    for candidate in _candidate_mask_dirs(mask_path, image_path):
        if candidate.exists():
            print(f"[Tracks] Using mask directory {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Mask directory '{mask_dir}' not found; searched {[str(c) for c in _candidate_mask_dirs(mask_path, image_path)]}"
    )


def read_video_npz(folder_path: Path) -> media._VideoArray:
    frame_paths = sorted(folder_path.glob("*.npz"))
    if not frame_paths:
        raise FileNotFoundError(f"No mask npz files found in {folder_path}")

    first = frame_paths[0]
    with np.load(first) as sample:
        if "dyn_mask" not in sample:
            raise KeyError(f"File {first} missing 'dyn_mask' array")

    target_width, target_height = 512, 288
    resized_masks = []
    for frame_path in frame_paths:
        with np.load(frame_path) as data:
            mask = np.squeeze(data["dyn_mask"]).astype(np.uint8)
        resized = cv2.resize(
            mask * 255,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_masks.append(resized[np.newaxis, ...].astype(bool))

    video = np.stack(resized_masks)
    print(f"MMMMASK_{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    return media._VideoArray(video)


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--grid_size", type=int, default=4, help="grid size")
    parser.add_argument("--resize_height", type=int, default=256, help="resize height")
    parser.add_argument("--resize_width", type=int, default=256, help="resize width")
    parser.add_argument(
        "--model_type", type=str, choices=["tapir", "bootstapir"], help="model type"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="checkpoint dir",
    )
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir_path = _resolve_mask_dir(args.mask_dir, folder_path)
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    print(f"{done=}")
    if done:
        print("Already done")
        return

    ## Load model
    ckpt_file = (
        "tapir_checkpoint_panning.pt"
        if args.model_type == "tapir"
        else "bootstapir_checkpoint_v2.pt"
    )
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)

    resize_height = args.resize_height
    resize_width = args.resize_width
    grid_size = args.grid_size

    video = read_video(folder_path)


    num_frames, height, width = video.shape[0:3]

    #try:
    #  masks = read_video(mask_dir)
    #  print('VIDEO SHAPE OF', masks.shape)
    #except: 
    masks = read_video_npz(mask_dir_path)[:]
    masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)
    print(f"{video.shape=} {masks.shape=} {masks.max()=} {masks.sum()=}")

    frames = media.resize_video(video, (resize_height, resize_width))
    print(f"{frames.shape=}")
    frames = torch.from_numpy(frames).to(device)
    frames = preprocess_frames(frames)[None]
    print(f"preprocessed {frames.shape=}")

    y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
    y_resize, x_resize = y / (height - 1) * (resize_height - 1), x / (width - 1) * (
        resize_width - 1
    )

    for t in tqdm(range(num_frames), desc="query frames"):
        name_t = os.path.splitext(frame_names[t])[0]
        file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
        if len(file_matches) == num_frames:
            print(f"Already computed tracks with query {t=} {name_t=}")
            continue

        all_points = np.stack([t * np.ones_like(y), y_resize, x_resize], axis=-1)
        mask = masks[t]
        in_mask = mask[y, x] > 0.5
        all_points_t = all_points[in_mask]
        print(f"{all_points.shape=} {all_points_t.shape=} {t=}")
        outputs = []
        if len(all_points_t) > 0:
            num_chunks = max(1, len(all_points_t) // 128)
            for points in tqdm(
                np.array_split(all_points_t, axis=0, indices_or_sections=num_chunks),
                leave=False,
                desc="points",
            ):
                points = torch.from_numpy(points.astype(np.float32))[None].to(
                    device
                )  # Add batch dimension
                with torch.inference_mode():
                    preds = model(frames, points)
                tracks, occlusions, expected_dist = (
                    preds["tracks"][0].detach().cpu().numpy(),
                    preds["occlusion"][0].detach().cpu().numpy(),
                    preds["expected_dist"][0].detach().cpu().numpy(),
                )
                tracks = transforms.convert_grid_coordinates(
                    tracks, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
                )
                outputs.append(
                    np.concatenate(
                        [tracks, occlusions[..., None], expected_dist[..., None]], axis=-1
                    )
                )
            outputs = np.concatenate(outputs, axis=0)
        else:
            outputs = np.zeros((0, num_frames, 4), dtype=np.float32)

        for j in range(num_frames):
            if j == t:
                original_query_points = np.stack([x[in_mask], y[in_mask]], axis=-1)
                outputs[:, j, :2] = original_query_points
            name_j = os.path.splitext(frame_names[j])[0]
            np.save(f"{out_dir}/{name_t}_{name_j}.npy", outputs[:, j])


if __name__ == "__main__":
    main()
