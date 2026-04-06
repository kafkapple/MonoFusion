# no-split: single training workflow — init→dataset→train→checkpoint must stay cohesive
"""
MonoFusion training wrapper for M5t2 mouse dataset.

Patches the CasualDataset camera loader to work with M5t2 format,
then delegates to the standard MonoFusion training pipeline.

Usage (V5h example):
    CC=x86_64-conda-linux-gnu-gcc CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 \
    python -u train_m5t2.py \
        --data_root /node_data/joon/data/monofusion/m5t2_v5 \
        --output_dir /node_data/joon/data/monofusion/m5t2_v5/results_v5h \
        --num_fg 5000 --num_bg 0 --num_motion_bases 10 --num_epochs 300 \
        --w_feat 1.5 --w_mask 1.0 --w_depth_reg 0.0 \
        --feat_ramp_start_epoch 5 --feat_ramp_end_epoch 100 \
        --max_gaussians 100000 --stop_densify_pct 0.6 \
        --densify_xys_grad_threshold 0.00015 \
        --feat_dir_name dinov2_features_pca32_norm \
        --disable_opacity_reset \
        --wandb_name v5h_experiment
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
    """Monkey-patch CasualDataset to support M5t2 camera format.

    Fixes in casual_dataset.py for M5t2 data:
    1. Sets correct default kwargs (depth_type, track_type, etc.)
    2. Patches load_known_cameras: range(0,300,3) �� range(0,T,1)
       and camera index extraction for M5t2 naming convention.
    """
    from flow3d.data.casual_dataset import CasualDataset

    original_init = CasualDataset.__init__

    def patched_init(self, *args, **kwargs):
        video_name = kwargs.get("video_name", "")
        if "m5t2" in video_name.lower() or "m5t2" in kwargs.get("seq_name", "").lower():
            kwargs.setdefault("depth_type", "moge")
            kwargs.setdefault("track_2d_type", "tapir")
            kwargs.setdefault("mask_type", "masks")
            kwargs.setdefault("image_type", "images")

        original_init(self, *args, **kwargs)

    CasualDataset.__init__ = patched_init


def create_m5t2_datasets(data_root: str, glb_step: int = 1, feat_dir_name: str = "dinov2_features"):
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
            "camera_convention": meta["camera_convention"],  # MUST exist — silent c2w fallback caused 8 wasted experiments
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
            # Set camera_id for per-camera depth scale alignment
            dataset.camera_id = ci
            # Override feature directory if non-default (e.g., PCA reduced features)
            if feat_dir_name != "dinov2_features":
                custom_feat_dir = data_root / feat_dir_name / seq_name
                if custom_feat_dir.exists():
                    dataset.feat_dir = custom_feat_dir
                    # Clear pre-loaded feature cache so load_feat re-reads from new dir
                    dataset.feats = [None for _ in dataset.frame_names]
                    print(f"  Using custom features: {feat_dir_name}/{seq_name}")
                else:
                    print(f"  WARNING: {custom_feat_dir} not found, using default")

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
    parser.add_argument("--wandb_project", type=str, default="monofusion-m5t2")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--w_feat", type=float, default=0.75,
                        help="Feature loss weight target (paper 1.5 with PCA 32d)")
    parser.add_argument("--feat_ramp_start_epoch", type=int, default=5,
                        help="Epoch to start gradual feature loss ramp (0=immediate)")
    parser.add_argument("--feat_ramp_end_epoch", type=int, default=50,
                        help="Epoch when feature loss reaches full w_feat")
    parser.add_argument("--w_depth_reg", type=float, default=0.0,
                        help="Depth regularization weight (paper default 0.0)")
    parser.add_argument("--max_gaussians", type=int, default=100000,
                        help="Hard cap on Gaussian count during densification (0=no cap)")
    parser.add_argument("--feat_dir_name", type=str, default="dinov2_features",
                        help="Feature directory name (e.g., dinov2_features_pca32)")
    parser.add_argument("--stop_densify_pct", type=float, default=0.6,
                        help="Fraction of total_steps for densification window (V5e: 0.4)")
    parser.add_argument("--densify_xys_grad_threshold", type=float, default=0.00015,
                        help="XY gradient threshold for densification (V5e: 0.0002)")
    parser.add_argument("--w_mask", type=float, default=7.0,
                        help="Mask loss weight (paper 7.0, reduce for small FG ratio)")
    parser.add_argument("--disable_opacity_reset", action="store_true",
                        help="Disable periodic opacity resets (better for dynamic scenes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (sets torch+numpy+cuda)")
    parser.add_argument("--bg_lr_config", type=str, default="gt",
                        choices=["gt", "frozen"],
                        help="BG LR config: 'gt'=BGLRGTConfig (paper spec), 'frozen'=BGLRConfig (~1e-9)")
    args = parser.parse_args()

    # Reproducibility: set seeds before any stochastic operation
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"  Random seed: {args.seed} (deterministic mode)")

    if args.output_dir is None:
        args.output_dir = str(Path(args.data_root) / "results")

    # Initialize wandb (following FaceLift pattern)
    import wandb
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        run_name = args.wandb_name or f"v5_{Path(args.data_root).name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "data_root": args.data_root,
                "num_fg": args.num_fg,
                "num_bg": args.num_bg,
                "num_motion_bases": args.num_motion_bases,
                "num_epochs": args.num_epochs,
                "output_dir": args.output_dir,
                "seed": args.seed,
                "bg_lr_config": args.bg_lr_config,
                "w_feat": args.w_feat,
                "w_mask": args.w_mask,
                "w_depth_reg": args.w_depth_reg,
                "max_gaussians": args.max_gaussians,
            },
            dir=args.output_dir,
        )

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
    datasets = create_m5t2_datasets(args.data_root, feat_dir_name=args.feat_dir_name)

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
    from flow3d.configs import FGLRConfig, BGLRConfig, BGLRGTConfig, MotionLRConfig
    bg_lr = BGLRGTConfig() if args.bg_lr_config == "gt" else BGLRConfig()
    lr_cfg = SceneLRConfig(fg=FGLRConfig(), bg=bg_lr, motion_bases=MotionLRConfig())
    print(f"  BG LR config: {args.bg_lr_config} (means={bg_lr.means}, feats={bg_lr.feats})")
    # V5d settings (MoA+Audit consensus):
    # - w_feat: 0.5 default with PCA 32d + L2 norm (paper 1.5, conservative start)
    # - w_depth_reg: 0.0 (paper default — nonzero was confirmed destabilizer)
    # - Gradual feat ramp replaces hard warmup (prevents gradient shock)
    loss_cfg = LossesConfig(w_feat=args.w_feat, w_depth_reg=args.w_depth_reg, w_mask=args.w_mask)
    # Compute steps/epoch from actual dataset: frames / batch_size
    n_frames = len(datasets[0].frame_names)
    batch_size = 4
    steps_per_epoch = n_frames // batch_size
    total_steps = args.num_epochs * steps_per_epoch
    print(f"  Steps/epoch: {steps_per_epoch} ({n_frames} frames / {batch_size} batch)")
    print(f"  Total steps: {total_steps} ({args.num_epochs} epochs × {steps_per_epoch})")
    stop_densify = int(total_steps * args.stop_densify_pct)
    stop_control = int(total_steps * 0.8)  # paper: 80% of training
    # Disable opacity resets for dynamic scenes (Hybrid 3D-4DGS: +0.73dB)
    reset_n = 999999 if args.disable_opacity_reset else 30
    optim_cfg = OptimizerConfig(
        max_steps=total_steps,
        stop_densify_steps=stop_densify,
        stop_control_steps=stop_control,
        stop_control_by_screen_steps=stop_control,
        densify_xys_grad_threshold=args.densify_xys_grad_threshold,
        max_gaussians=args.max_gaussians,
        reset_opacity_multiplier=1.5,  # Reset above cull threshold (0.15 > 0.1)
        reset_opacity_every_n_controls=reset_n,
    )
    print(f"  Densification: steps 0-{stop_densify} ({args.stop_densify_pct*100:.0f}% of {total_steps})")
    print(f"  Densify grad threshold: {args.densify_xys_grad_threshold}")
    print(f"  Control: steps 0-{stop_control} (80%), reset_opacity_mult=1.5")
    print(f"  w_mask={args.w_mask}, opacity_reset={'DISABLED' if args.disable_opacity_reset else f'every {reset_n} controls'}")

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path, device, lr_cfg, loss_cfg, optim_cfg,
        work_dir=str(work_dir), port=None,
    )

    # Create data loaders
    train_loaders = []
    for ds in datasets:
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=0,
            persistent_workers=False,
            collate_fn=BaseDataset.train_collate_fn,
        )
        train_loaders.append(loader)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  FG Gaussians: {args.num_fg}")
    print(f"  BG Gaussians: {args.num_bg}")
    print(f"  Motion bases: {args.num_motion_bases}")
    print(f"  w_feat={args.w_feat} (ramp {args.feat_ramp_start_epoch}-{args.feat_ramp_end_epoch}), "
          f"w_depth_reg={args.w_depth_reg}, w_rgb={loss_cfg.w_rgb}")
    print(f"  max_gaussians={args.max_gaussians}, num_bg={args.num_bg}")

    from tqdm import tqdm
    import glob
    import imageio

    best_loss = float("inf")
    epoch_losses = []
    ckpt_dir = work_dir / "checkpoints"
    preview_dir = work_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path, epoch, loss_val=None, tag=""):
        ckpt_data = {
            "model": trainer.model.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in trainer.optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in trainer.scheduler.items()},
            "epoch": epoch,
            "global_step": trainer.global_step,
            "avg_loss": loss_val,
        }
        # Save per-camera depth alignment params (not in model.state_dict)
        if hasattr(trainer, 'depth_scales'):
            ckpt_data["depth_scales"] = trainer.depth_scales.data
        if hasattr(trainer, 'depth_shifts'):
            ckpt_data["depth_shifts"] = trainer.depth_shifts.data
        torch.save(ckpt_data, path)
        print(f"  Checkpoint saved: {os.path.basename(path)} {tag}")
        print(f"  Gaussians: {trainer.model.num_gaussians}")

    def cleanup_old_ckpts(keep_n=2):
        """Keep only the most recent N epoch checkpoints + best.ckpt."""
        ckpts = sorted(glob.glob(str(ckpt_dir / "epoch_*.ckpt")))
        for old in ckpts[:-keep_n]:
            os.remove(old)
            print(f"  Removed old checkpoint: {os.path.basename(old)}")

    def save_preview(epoch):
        """Render cam0 frame0: GT vs predicted side-by-side."""
        try:
            with torch.no_grad():
                w2c = trainer.model.w2cs[0:1].to(device)
                K = trainer.model.Ks[0:1].to(device)
                out = trainer.model.render(
                    t=0, w2cs=w2c, Ks=K, img_wh=(512, 512),
                    return_color=True, return_feat=False,
                )
                rendered = out["img"][0].clamp(0, 1).cpu().numpy()
                rendered_u8 = (rendered * 255).astype(np.uint8)

                # Load GT image for comparison
                gt_img = datasets[0].get_image(0)
                if hasattr(gt_img, 'numpy'):
                    gt_img = gt_img.numpy()
                if gt_img.max() <= 1.0:
                    gt_img = (gt_img * 255).astype(np.uint8)
                if gt_img.shape[:2] != rendered_u8.shape[:2]:
                    import cv2
                    gt_img = cv2.resize(gt_img, (rendered_u8.shape[1], rendered_u8.shape[0]))

                # Side-by-side: GT | Rendered
                combined = np.concatenate([gt_img[:, :, :3], rendered_u8], axis=1)
                path = str(preview_dir / f"epoch_{epoch:04d}_gt_vs_pred.png")
                imageio.imwrite(path, combined)
                print(f"  Preview saved: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Preview failed: {e}")

    # Pre-load viz module once (was previously reloaded every save_interval epochs)
    try:
        import importlib.util
        _vzmod_path = str(MONOFUSION_ROOT / "mouse_m5t2" / "scripts" / "viz_scene_flow.py")
        _spec = importlib.util.spec_from_file_location("viz_scene_flow", _vzmod_path)
        _vzmod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vzmod)
        viz_magnitude_heatmap = _vzmod.viz_magnitude_heatmap
        viz_trajectory_trails = _vzmod.viz_trajectory_trails
        _viz_available = True
    except Exception as e:
        print(f"  Motion viz unavailable: {e}")
        _viz_available = False

    w_feat_target = trainer.losses_cfg.w_feat
    ramp_start = args.feat_ramp_start_epoch
    ramp_end = args.feat_ramp_end_epoch
    print(f"  Feature ramp: epochs {ramp_start}-{ramp_end}, "
          f"w_feat 0→{w_feat_target} (gradual)")
    print(f"  Max Gaussians cap: {args.max_gaussians}")
    print(f"  w_depth_reg: {args.w_depth_reg}")

    for epoch in (pbar := tqdm(range(start_epoch, args.num_epochs),
                                initial=start_epoch, total=args.num_epochs)):
        # Gradual feature loss ramp (prevents gradient shock on RGB-optimized structure)
        if epoch < ramp_start:
            trainer.losses_cfg.w_feat = 0.0
        elif epoch < ramp_end:
            progress = (epoch - ramp_start) / (ramp_end - ramp_start)
            trainer.losses_cfg.w_feat = w_feat_target * progress
        else:
            trainer.losses_cfg.w_feat = w_feat_target

        if epoch % 25 == 0 or epoch == ramp_start or epoch == ramp_end:
            print(f"\n  [Epoch {epoch}] w_feat={trainer.losses_cfg.w_feat:.4f}")

        trainer.set_epoch(epoch)
        step_losses = []
        for batches in zip(*train_loaders):
            batches = [to_device(batch, device) for batch in batches]
            loss = trainer.train_step(batches)
            loss.backward()
            trainer.op_af_bk()
            step_losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.6f}")

        avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0
        epoch_losses.append(avg_loss)
        # Extract PSNR from last step's stats for console output
        psnr_str = ""
        if hasattr(trainer, 'stats') and trainer.stats:
            psnr_val = trainer.stats.get("train/psnr")
            if psnr_val is not None:
                psnr_val = psnr_val.item() if hasattr(psnr_val, 'item') else psnr_val
                psnr_str = f", PSNR={psnr_val:.2f}"
        print(f"\n  Epoch {epoch}: avg_loss={avg_loss:.6f} ({len(step_losses)} steps){psnr_str}")

        # Wandb epoch logging (FaceLift pattern: train/* namespace)
        log_dict = {
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch,
            "train/global_step": trainer.global_step,
            "train/num_gaussians": trainer.model.num_gaussians,
            "train/num_fg_gaussians": trainer.model.num_fg_gaussians,
            "train/best_loss": best_loss,
        }
        # Log per-component losses + PSNR from trainer stats (single source of truth)
        if hasattr(trainer, 'stats') and trainer.stats:
            for k, v in trainer.stats.items():
                if isinstance(v, (int, float)):
                    log_dict[k] = v
                elif hasattr(v, 'item'):  # torch.Tensor scalars (PSNR, etc.)
                    log_dict[k] = v.item()

        # Motion metrics (cheap scalars, every epoch)
        try:
            with torch.no_grad():
                t0, t1 = 0, min(1, n_frames - 1)
                ts_pair = torch.tensor([t0, t1], device=device)
                means_pair, _ = trainer.model.compute_poses_fg(ts_pair)  # (G, 2, 3)
                flow_vec = means_pair[:, 1] - means_pair[:, 0]  # (G, 3)
                flow_mag = flow_vec.norm(dim=-1)  # (G,)
                log_dict["motion/flow_mean"] = flow_mag.mean().item()
                log_dict["motion/flow_max"] = flow_mag.max().item()
                log_dict["motion/flow_std"] = flow_mag.std().item()

                # Motion basis utilization + entropy
                coefs_raw = trainer.model.fg.params["motion_coefs"]  # (G, K)
                probs = torch.softmax(coefs_raw, dim=-1)  # (G, K)
                utilization = probs.mean(dim=0)  # (K,)
                for k_idx in range(utilization.shape[0]):
                    log_dict[f"motion/basis_{k_idx}_util"] = utilization[k_idx].item()
                entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                log_dict["motion/coef_entropy"] = entropy.item()
        except Exception as e:
            print(f"  Motion metrics failed: {e}")

        wandb.log(log_dict, step=trainer.global_step)

        # Best checkpoint tracking
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(str(ckpt_dir / "best.ckpt"), epoch, loss_val=avg_loss,
                      tag=f"★ best={best_loss:.4f}")

        # Periodic checkpoint (every 50 epochs) + cleanup
        save_interval = 50 if args.num_epochs > 100 else 10
        if epoch % save_interval == 0 or epoch == args.num_epochs - 1:
            save_ckpt(str(ckpt_dir / f"epoch_{epoch:04d}.ckpt"), epoch, loss_val=avg_loss)
            cleanup_old_ckpts(keep_n=2)
            save_preview(epoch)

            # Wandb image logging: GT vs rendered + motion visualizations
            vis_log = {}
            preview_path = preview_dir / f"epoch_{epoch:04d}_gt_vs_pred.png"
            if preview_path.exists():
                vis_log["vis/gt_vs_rendered"] = wandb.Image(
                    str(preview_path), caption=f"Epoch {epoch} | Loss {avg_loss:.4f}")

            # Motion visualizations
            try:
                if not _viz_available:
                    raise RuntimeError("viz module not loaded")
                with torch.no_grad():
                    # Scene flow heatmap: frame 0 → frame T//4
                    t_mid = n_frames // 4
                    ts_flow = torch.tensor([0, t_mid], device=device)
                    means_flow, _ = trainer.model.compute_poses_fg(ts_flow)
                    m0 = means_flow[:, 0].cpu().numpy()
                    m1 = means_flow[:, 1].cpu().numpy()
                    heatmap_path = str(preview_dir / f"epoch_{epoch:04d}_flow_heatmap.png")
                    viz_magnitude_heatmap(m0, m1, heatmap_path, 0, t_mid)
                    vis_log["vis/scene_flow_heatmap"] = wandb.Image(
                        heatmap_path, caption=f"Flow F0→F{t_mid}")

                    # Trajectory trails (subsample 10 frames for speed)
                    trail_ts = torch.linspace(0, n_frames - 1, min(10, n_frames),
                                              device=device).long()
                    all_means, _ = trainer.model.compute_poses_fg(trail_ts)  # (G, 10, 3)
                    all_means_np = all_means.cpu().numpy().transpose(1, 0, 2)  # (10, G, 3)
                    trails_path = str(preview_dir / f"epoch_{epoch:04d}_trails.png")
                    viz_trajectory_trails(all_means_np, trails_path, n_trails=50)
                    vis_log["vis/trajectory_trails"] = wandb.Image(
                        trails_path, caption=f"Top-50 trails ({len(trail_ts)} frames)")

                    # Flow magnitude histogram
                    flow_mag_all = np.linalg.norm(m1 - m0, axis=1)
                    vis_log["vis/flow_histogram"] = wandb.Histogram(flow_mag_all)
            except Exception as e:
                print(f"  Motion viz failed: {e}")

            if vis_log:
                wandb.log(vis_log, step=trainer.global_step)

    # Save loss curve
    import json
    loss_path = str(work_dir / "loss_curve.json")
    with open(loss_path, "w") as f:
        json.dump({"epochs": list(range(len(epoch_losses))), "losses": epoch_losses}, f)
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {ckpt_dir}/best.ckpt + latest 2 epoch checkpoints")
    print(f"Loss curve: {loss_path}")


if __name__ == "__main__":
    main()
