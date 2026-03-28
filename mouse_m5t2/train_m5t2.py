"""
MonoFusion training wrapper for M5t2 mouse dataset.

Patches the CasualDataset camera loader to work with M5t2 format,
then delegates to the standard MonoFusion training pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=4 python train_m5t2.py \
        --data_root /node_data/joon/data/monofusion/m5t2_poc \
        --num_fg 5000 \
        --num_bg 10000 \
        --num_motion_bases 10 \
        --num_epochs 50
"""
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# Add MonoFusion to path
MONOFUSION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MONOFUSION_ROOT))
sys.path.insert(0, str(MONOFUSION_ROOT / "preproc" / "Dust3R"))


def patch_casual_dataset():
    """Monkey-patch CasualDataset to support M5t2 camera format."""
    from flow3d.data.casual_dataset import CasualDataset

    original_init = CasualDataset.__init__

    def patched_init(self, *args, **kwargs):
        # Intercept video_name to detect M5t2
        video_name = kwargs.get("video_name", "")
        if "m5t2" in video_name.lower() or "m5t2" in kwargs.get("seq_name", "").lower():
            kwargs.setdefault("depth_type", "aligned_moge_depth")
            kwargs.setdefault("track_2d_type", "tapir")
            kwargs.setdefault("mask_type", "masks")
            kwargs.setdefault("image_type", "images")

        original_init(self, *args, **kwargs)

    CasualDataset.__init__ = patched_init


def create_m5t2_datasets(data_root: str, glb_step: int = 1):
    """Create CasualDataset instances for each M5t2 camera."""
    from flow3d.data.casual_dataset import CasualDataset, CustomDataConfig

    data_root = Path(data_root)
    info_path = data_root / "conversion_info.json"
    with open(info_path) as f:
        info = json.load(f)

    num_cams = info["camera_count"]
    cam_seqs = [f"m5t2_cam{i:02d}" for i in range(num_cams)]

    # Load camera meta to patch into the format expected by load_known_cameras
    meta_path = data_root / "_raw_data" / "m5t2" / "trajectory" / "Dy_train_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    datasets = []
    for ci, seq_name in enumerate(cam_seqs):
        print(f"\nCreating dataset for {seq_name} (camera index {ci})...")

        # Build per-camera Dy_train_meta.json that load_known_cameras expects
        # It indexes by: md['k'][t][c] where c = int(seq_name[-1])
        # We need to create a meta where camera index = last digit of seq_name
        cam_meta_dir = data_root / "_raw_data" / "m5t2" / "trajectory"
        cam_meta_dir.mkdir(parents=True, exist_ok=True)

        # The camera index extracted by load_known_cameras:
        # c = int(self.seq_name[-1])  -- takes last char
        # But our seq_name is like "m5t2_cam00" -> last char = "0"
        # So we need camera data at index 0 in the meta

        # Remap: put this camera's data at index 0
        T = len(meta["k"])
        remapped_meta = {
            "hw": [meta["hw"][ci]],  # single camera
            "k": [[meta["k"][t][ci]] for t in range(T)],
            "w2c": [[meta["w2c"][t][ci]] for t in range(T)],
        }

        cam_meta_path = cam_meta_dir / f"Dy_train_meta_cam{ci:02d}.json"
        with open(cam_meta_path, "w") as f:
            json.dump(remapped_meta, f)

        try:
            dataset = CasualDataset(
                seq_name=seq_name,
                root_dir=str(data_root),
                start=0,
                end=-1,
                res="",
                image_type="images",
                mask_type="masks",
                depth_type="moge",
                camera_type="droid_recon",
                track_2d_type="tapir",
                mask_erosion_radius=2,
                video_name="_m5t2",
                super_fast=False,
            )
            datasets.append(dataset)
            print(f"  OK: {len(dataset.frame_names)} frames")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_fg", type=int, default=5000)
    parser.add_argument("--num_bg", type=int, default=10000)
    parser.add_argument("--num_motion_bases", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.data_root) / "results")

    print("=" * 60)
    print("MonoFusion M5t2 PoC Training")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"FG Gaussians: {args.num_fg}")
    print(f"BG Gaussians: {args.num_bg}")
    print(f"Motion bases: {args.num_motion_bases}")

    # Verify data exists
    data_root = Path(args.data_root)
    required = ["images", "masks", "dinov2_features", "aligned_moge_depth", "tapir"]
    for d in required:
        path = data_root / d
        if not path.exists():
            print(f"ERROR: Missing {path}")
            sys.exit(1)
        subdirs = list(path.iterdir())
        print(f"  {d}/: {len(subdirs)} subdirs")

    # Patch and create datasets
    patch_casual_dataset()
    datasets = create_m5t2_datasets(args.data_root)

    if not datasets:
        print("ERROR: No datasets created")
        sys.exit(1)

    print(f"\n{len(datasets)} camera datasets loaded")

    # Test basic data loading
    print("\nTesting data loading...")
    for i, ds in enumerate(datasets):
        try:
            img = ds.get_image(0)
            mask = ds.get_mask(0)
            feat = ds.get_feat(0)
            print(f"  cam{i}: img={img.shape}, mask={mask.shape}, feat={feat.shape}")
        except Exception as e:
            print(f"  cam{i}: LOAD ERROR - {e}")

    print("\nData loading test complete. Starting training...")

    # Import training components
    from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
    from flow3d.data import BaseDataset, SynchornizedDataset
    from flow3d.data.utils import to_device
    from flow3d.init_utils import (
        init_bg, init_fg_from_tracks_3d,
        init_motion_params_with_procrustes, run_initial_optim,
    )
    from flow3d.params import GaussianParams, MotionBases
    from flow3d.scene_model import SceneModel
    from flow3d.tensor_dataclass import StaticObservations, TrackObservations
    from flow3d.trainer import Trainer

    device = torch.device("cuda")
    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(work_dir / "checkpoints" / "last.ckpt")

    # Initialize model from tracks
    print("\nInitializing model from unified tracks...")
    from dance_glb import init_model_from_unified_tracks, initialize_and_checkpoint_model

    # Create a minimal TrainConfig-like object
    _work_dir = str(work_dir)
    _num_fg = args.num_fg
    _num_bg = args.num_bg
    _num_bases = args.num_motion_bases

    class M5t2Config:
        num_fg = _num_fg
        num_bg = _num_bg
        num_motion_bases = _num_bases
        port = None
        vis_debug = False

    M5t2Config.work_dir = _work_dir
    cfg = M5t2Config()

    initialize_and_checkpoint_model(
        cfg, datasets, device, ckpt_path,
        vis=False, port=None, seq_name="m5t2"
    )

    # Load trainer from checkpoint
    from flow3d.configs import FGLRConfig, BGLRConfig, MotionLRConfig
    lr_cfg = SceneLRConfig(fg=FGLRConfig(), bg=BGLRConfig(), motion_bases=MotionLRConfig())
    loss_cfg = LossesConfig()
    optim_cfg = OptimizerConfig()

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path, device, lr_cfg, loss_cfg, optim_cfg,
        work_dir=str(work_dir), port=None,
    )

    # Create data loaders
    train_loaders = []
    for ds in datasets:
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, num_workers=2,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )
        train_loaders.append(loader)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  FG Gaussians: {args.num_fg}")
    print(f"  BG Gaussians: {args.num_bg}")
    print(f"  Motion bases: {args.num_motion_bases}")

    from tqdm import tqdm
    for epoch in (pbar := tqdm(range(start_epoch, args.num_epochs),
                                initial=start_epoch, total=args.num_epochs)):
        trainer.set_epoch(epoch)
        for batches in zip(*train_loaders):
            batches = [to_device(batch, device) for batch in batches]
            loss = trainer.train_step(batches)
            loss.backward()
            trainer.op_af_bk()
            pbar.set_description(f"Loss: {loss:.6f}")

        # Save checkpoint periodically
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            save_path = str(work_dir / "checkpoints" / f"epoch_{epoch:04d}.ckpt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model": trainer.model.state_dict(),
                "epoch": epoch,
                "global_step": trainer.global_step,
            }, save_path)
            print(f"\n  Checkpoint saved: {save_path}")

    print(f"\nTraining complete! Final checkpoint at {work_dir}/checkpoints/")


if __name__ == "__main__":
    main()
