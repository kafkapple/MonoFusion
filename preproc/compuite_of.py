import argparse
import glob
import os

import imageio
import cv2
import mediapy as media
import numpy as np
import torch
from tapnet_torch import tapir_model, transforms
from tqdm import tqdm

import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
from of_utils import *
def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video

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
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--grid_size", type=int, default=4, help="grid size")
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAFT.from_pretrained(args.url, args=args)
    model = model.to(device)

    resize_height = args.resize_height
    resize_width = args.resize_width
    grid_size = args.grid_size

    video = read_video(folder_path)
    num_frames, height, width = video.shape[0:3]
    frames = media.resize_video(video, (resize_height, resize_width))
    print(f"{frames.shape=}")
    frames = torch.from_numpy(frames).to(device)
    frames = preprocess_frames(frames)[None]
    print(f"preprocessed {frames.shape=}")


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
