import functools
import os
import os.path as osp
import time
from dataclasses import asdict
from typing import cast

import imageio as iio
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState, Viewer
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data.utils import normalize_coords, to_device

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ... [Your existing code for loading data and computing estimated_depth]

# After computing estimated_depth, add the following function and code:

import torch
import os
import pickle
import torch
import os

import torch
import os

def unproject_image(img, w2c, K, depth, mask, glb_pc, img_wh, output_dir='output_depth_maps', output_filename='estimated_depth.png'):
    device = img.device
    w2c = w2c.to(device)
    K = K.to(device)
    depth = depth.to(device)
    mask = mask.to(device)
    pc = torch.from_numpy(glb_pc).float().to(device)  # (N, 3)
    ones = torch.ones((pc.shape[0], 1), device=device, dtype=pc.dtype)
    print(w2c.shape, K.shape)
    pc_hom = torch.cat([pc, ones], dim=1)  # (N, 4)
    c_pc_hom = pc_hom @ w2c.T  # (N, 4)
    c_pc = c_pc_hom[:, :3]  # (N, 3)
    print(pc_hom.shape)
    x, y, z = c_pc[:, 0], c_pc[:, 1], c_pc[:, 2]  # (N,)

    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Stack coordinates for matrix multiplication
    coords = torch.stack([x, y, z], dim=0)  # (3, N_valid)
    uv = K @ coords
    u = uv[0, :] / uv[2, :]
    v = uv[1, :] / uv[2, :]

    # Round to nearest pixel indices
    u_pixel = torch.round(u).long()
    v_pixel = torch.round(v).long()

    # Unpack image dimensions
    W, H = img_wh

    # Filter points within image bounds
    in_bounds = (u_pixel >= 0) & (u_pixel < W) & (v_pixel >= 0) & (v_pixel < H)
    u_pixel = u_pixel[in_bounds]
    v_pixel = v_pixel[in_bounds]
    z = z[in_bounds]

    # Compute linear pixel indices
    pixel_indices = v_pixel * W + u_pixel  # (N_in_bounds,)

    # Initialize estimated depth map
    estimated_depth_flat = torch.full((H * W,), float('inf'), device=device, dtype=pc.dtype)

    # Manually assign depth values to the estimated depth map
    # Sort pixel_indices and corresponding z values
    sorted_indices = torch.argsort(pixel_indices)
    sorted_pixel_indices = pixel_indices[sorted_indices]
    sorted_z = z[sorted_indices]

    # Find the indices where pixel_indices change
    change_indices = torch.cat([
        torch.tensor([0], device=sorted_pixel_indices.device),
        (sorted_pixel_indices[1:] != sorted_pixel_indices[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([len(sorted_pixel_indices)], device=sorted_pixel_indices.device)
    ])

    # Loop over unique pixel indices to assign the minimum depth value
    for i in range(len(change_indices) - 1):
        start = change_indices[i].item()
        end = change_indices[i + 1].item()
        idx = sorted_pixel_indices[start].item()
        min_z = sorted_z[start:end].min()
        estimated_depth_flat[idx] = min_z

    # Reshape to (H, W)
    estimated_depth = estimated_depth_flat.view(H, W)

    gt_depth = depth
    valid_mask = (estimated_depth != float('inf')) & (gt_depth > 0)# & mask.bool()

    # Extract valid depth values
    est_depth_valid = estimated_depth[valid_mask]
    gt_depth_valid = gt_depth[valid_mask]

    # Calculate Absolute Relative Error
    abs_rel_error = torch.mean(torch.abs(gt_depth_valid - est_depth_valid) / gt_depth_valid)
    print(f"Absolute Relative Error: {abs_rel_error.item()}")

    # Save the depth map
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    save_depth_map(estimated_depth, output_path)

    return abs_rel_error, estimated_depth

def save_depth_map(depth_map, filename):
    import matplotlib.pyplot as plt
    plt.imshow(depth_map.cpu(), cmap='plasma', vmin=0, vmax=depth_map[depth_map != float('inf')].max())
    plt.colorbar()
    plt.savefig(filename)
    plt.close()



class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_img_loader: DataLoader | None = None,
        val_kpt_loader: DataLoader | None = None,
        save_dir: str | None = None,
        do_the_trick: float | None = None,
        no_fg: bool=0
    ):
        self.model = model
        if do_the_trick:
           self.model.bg.params['scales'] = do_the_trick *  self.model.bg.params['scales']
        #if no_fg:
        '''fg_path = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_nus_cpr_08_1/_algo_depth'
        fg_path = f"{fg_path}/checkpoints/last.ckpt"
        ckpt_fg = torch.load(fg_path)["model"]
        model_fg = SceneModel.init_from_state_dict(ckpt_fg)
        self.model.fg = model_fg.fg'''
        self.device = device
        self.train_loader = train_loader
        self.val_img_loader = val_img_loader
        self.val_kpt_loader = val_kpt_loader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg
    
    @torch.no_grad()
    def save_train_videos(self, epoch: int):
        if self.train_loader is None:
            return
        video_dir = osp.join(self.save_dir, "videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)
        fps = 15.0
        # Render video.
        video = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]
            # (H, W).
            depth = batch["depths"][0]

            mask = batch["masks"][0]

            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
            ref_pred_depth = torch.cat(
                (depth[..., None], rendered["depth"][0]), dim=1
            ).cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
            if rendered["mask"] is not None:
                #masks.append(rendered["mask"][0].cpu().squeeze(-1))
                #print(mask.shape, rendered["mask"][0].shape, 'MASKSHAPE')
                masks.append(torch.cat([mask, rendered["mask"][0].squeeze(-1)], dim=1).cpu())

        # rgb video
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=30,
        )
        # depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        if len(masks) > 0:
            # mask video
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        # Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 10
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4).
        w2c = batch0["w2cs"]
        # (3, 3).
        K = batch0["Ks"]
        # (H, W, 3).
        img = batch0["imgs"]
        # (H, W).
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch in tqdm(
            self.train_loader, desc="Rendering 2D track video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3).
            target_imgs.append(batch["imgs"].cpu())
            # (1,).
            target_ts = batch["ts"]
            # (1, 4, 4).
            target_w2cs = batch["w2cs"]
            # (1, 3, 3).
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())
        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "tracks_2d.mp4"),
            make_video_divisble(np.stack(track_2d_video, 0)),
            fps=fps,
        )
        # Render motion coefficient video.
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            motion_coef_colors = torch.pca_lowrank(
                self.model.fg.get_coefs()[None],
                q=3,
            )[0][0]
        motion_coef_colors = (motion_coef_colors - motion_coef_colors.min(0)[0]) / (
            motion_coef_colors.max(0)[0] - motion_coef_colors.min(0)[0]
        )

        if self.model.bg is None: 
          motion_coef_colors = motion_coef_colors
        else:
          motion_coef_colors = F.pad(
              motion_coef_colors, (0, 0, 0, self.model.bg.num_gaussians), value=0.5
          )
        video = []
        for batch in tqdm(
            self.train_loader, desc="Rendering motion coefficient video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (3, 3).
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "motion_coefs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )

    @torch.no_grad()
    def save_train_videos_images(self, epoch: int):
        if self.train_loader is None:
            return

        # Create directories for videos and images
        video_dir = osp.join(self.save_dir, "videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)
        image_dir = osp.join(self.save_dir, "images", f"epoch_{epoch:04d}")
        os.makedirs(image_dir, exist_ok=True)
        rgb_dir = osp.join(image_dir, "rgbs")
        depth_dir = osp.join(image_dir, "depths")
        mask_dir = osp.join(image_dir, "masks")
        tracks_dir = osp.join(image_dir, "tracks_2d")
        motion_coefs_dir = osp.join(image_dir, "motion_coefs")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(tracks_dir, exist_ok=True)
        os.makedirs(motion_coefs_dir, exist_ok=True)

        fps = 15.0
        # Render video.
        video = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Extract necessary data from the batch
            t = batch["ts"][0]
            w2c = batch["w2cs"][0]
            K = batch["Ks"][0]
            img = batch["imgs"][0]
            depth = batch["depths"][0]
            mask = batch["masks"][0]
            img_wh = img.shape[-2::-1]

            # Render the image
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )

            # Concatenate the original and rendered images
            concat_img = torch.cat([img, rendered["img"][0]], dim=1).cpu()
            video.append(concat_img)

            # Process depth
            ref_pred_depth = torch.cat((depth[..., None], rendered["depth"][0]), dim=1).cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())

            # Process mask
            if rendered["mask"] is not None:
                concat_mask = torch.cat([mask, rendered["mask"][0].squeeze(-1)], dim=1).cpu()
                masks.append(concat_mask)

            # Save RGB image
            rgb_image = (concat_img.numpy() * 255).astype(np.uint8)
            imageio.imwrite(
                osp.join(rgb_dir, f"frame_{batch_idx:04d}.png"),
                rgb_image
            )

            # Save depth image
            depth_image = apply_depth_colormap(
                ref_pred_depth, near_plane=depth_min, far_plane=depth_max
            ).numpy()
            depth_image = (depth_image * 255).astype(np.uint8)
            imageio.imwrite(
                osp.join(depth_dir, f"frame_{batch_idx:04d}.png"),
                depth_image
            )

            # Save mask image
            if rendered["mask"] is not None:
                mask_image = (concat_mask.numpy() * 255).astype(np.uint8)
                imageio.imwrite(
                    osp.join(mask_dir, f"frame_{batch_idx:04d}.png"),
                    mask_image
                )

        # Save RGB video
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )

        # Save depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )

        # Save mask video if masks are available
        if len(masks) > 0:
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        # Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 10
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        t = batch0["ts"]
        w2c = batch0["w2cs"]
        K = batch0["Ks"]
        img = batch0["imgs"]
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering 2D track video", leave=False)
        ):

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_imgs.append(batch["imgs"].cpu())
            target_ts = batch["ts"]
            target_w2cs = batch["w2cs"]
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())

        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "tracks_2d.mp4"),
            make_video_divisble(np.stack(track_2d_video, 0)),
            fps=fps,
        )

        # Save 2D track images
        for idx, frame in enumerate(track_2d_video):
            imageio.imwrite(
                osp.join(tracks_dir, f"frame_{idx:04d}.png"),
                frame
            )

        # Render motion coefficient video.
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            motion_coef_colors = torch.pca_lowrank(
                self.model.fg.get_coefs()[None],
                q=3,
            )[0][0]
        motion_coef_colors = (motion_coef_colors - motion_coef_colors.min(0)[0]) / (
            motion_coef_colors.max(0)[0] - motion_coef_colors.min(0)[0]
        )

        if self.model.bg is not None:
            motion_coef_colors = F.pad(
                motion_coef_colors, (0, 0, 0, self.model.bg.num_gaussians), value=0.5
            )

        video = []
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering motion coefficient video", leave=False)
        ):

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            t = batch["ts"][0]
            w2c = batch["w2cs"][0]
            K = batch["Ks"][0]
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            concat_img = torch.cat([img, rendered["img"][0]], dim=1).cpu()
            video.append(concat_img)

            # Save motion coefficient images
            motion_coef_image = (concat_img.numpy() * 255).astype(np.uint8)
            imageio.imwrite(
                osp.join(motion_coefs_dir, f"frame_{batch_idx:04d}.png"),
                motion_coef_image
            )

        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "motion_coefs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )


    @torch.no_grad()
    def save_int_videos(self, epoch: int, w2c, nnvs='', ego=False, Ks=None):
        if self.train_loader is None:
            return
        # Directories for videos
        video_dir = osp.join(self.save_dir, f"int_videos{nnvs}", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)

        # Directories for images
        image_dir = osp.join(self.save_dir, f"int_images{nnvs}", f"epoch_{epoch:04d}")
        os.makedirs(image_dir, exist_ok=True)

        feat_dir = osp.join(self.save_dir, f"int_feats{nnvs}", f"epoch_{epoch:04d}")
        os.makedirs(feat_dir, exist_ok=True)


        print(f"Saving videos to {video_dir} and images to {image_dir}")

        fps = 15.0


        video = []
        video_dino = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0

        

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):


            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]

            if ego: 
                video = []
                for batch_idx, (K, w2c_ego) in enumerate(zip(Ks, w2c)):
                  w2c_ego = torch.tensor(w2c_ego).float().to(img.device)
                  img_wh = (512, 512)#
                  rendered = self.model.render(
                      t, w2c_ego[None], K[None], img_wh, return_depth=True, return_mask=True
                      )
                  #print(torch.cat([img, rendered["img"][0]], dim=1).cpu().shape)
                  combined_img =rendered["img"][0].cpu()#  torch.cat([img, rendered["img"][0]], dim=1).cpu()
                  video.append(combined_img)                
                  ref_pred_depth = rendered["depth"][0].cpu()
                  ref_pred_depths.append(ref_pred_depth)
                  depth_min = min(depth_min, ref_pred_depth.min().item())
                  depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
                  if rendered["mask"] is not None:
                      masks.append(rendered["mask"][0].cpu().squeeze(-1))

                  image_path = osp.join(image_dir, f"frame_{batch_idx:04d}.png")
                  iio.imwrite(
                      image_path,
                      (combined_img.numpy() * 255).astype(np.uint8)
                  )

                  feat_path = osp.join(feat_dir, f"frame_{batch_idx:04d}.png")

                  depth_colormap = apply_depth_colormap(
                      ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                  )
                  depth_image_path = osp.join(image_dir, f"depth_{batch_idx:04d}.png")
                  iio.imwrite(
                      depth_image_path,
                      (depth_colormap.numpy() * 255).astype(np.uint8)
                  )
                video = torch.stack(video, dim=0)
                iio.mimwrite(
                    osp.join(video_dir, "rgbs.mp4"),
                    make_video_divisble((video.numpy() * 255).astype(np.uint8)),
                    fps=30,
                )

            
            else:


              w2c = torch.tensor(w2c).float().to(img.device)
              img_wh = img.shape[-2::-1]
              rendered = self.model.render(
                  t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
              )
              combined_img = torch.cat([img, rendered["img"][0]], dim=1).cpu()

              video.append(combined_img)

              ref_pred_depth = rendered["depth"][0].cpu()
              ref_pred_depths.append(ref_pred_depth)
              depth_min = min(depth_min, ref_pred_depth.min().item())
              depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
              if rendered["mask"] is not None:
                  masks.append(rendered["mask"][0].cpu().squeeze(-1))

              # Save individual images
              image_path = osp.join(image_dir, f"frame_{batch_idx:04d}.png")
              iio.imwrite(
                  image_path,
                  (combined_img.numpy() * 255).astype(np.uint8)
              )


              # Save depth images
              depth_colormap = apply_depth_colormap(
                  ref_pred_depth, near_plane=depth_min, far_plane=depth_max
              )
              depth_image_path = osp.join(image_dir, f"depth_{batch_idx:04d}.png")
              iio.imwrite(
                  depth_image_path,
                  (depth_colormap.numpy() * 255).astype(np.uint8)
              )

              # Save mask images if available
              if len(masks) > 0:
                  mask_image = masks[-1]
                  mask_image_path = osp.join(image_dir, f"mask_{batch_idx:04d}.png")
                  iio.imwrite(
                      mask_image_path,
                      (mask_image.numpy() * 255).astype(np.uint8)
                  )
    @torch.no_grad()
    def save_nts_images(self, epoch: int):
        if self.train_loader is None:
            return
        # Directories for videos
        video_dir = osp.join(self.save_dir, f"nts_videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)

        # Directories for images
        image_dir = osp.join(self.save_dir, f"nts_images", f"epoch_{epoch:04d}")
        os.makedirs(image_dir, exist_ok=True)


        print(f"Saving videos to {video_dir} and images to {image_dir}")

        fps = 15.0
        # Render video.




        video = []
        video_dino = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]

            w2c = torch.tensor(w2c).float().to(img.device)
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            combined_img = torch.cat([img, rendered["img"][0]], dim=1).cpu()

            video.append(combined_img)






            ref_pred_depth = rendered["depth"][0].cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
            if rendered["mask"] is not None:
                masks.append(rendered["mask"][0].cpu().squeeze(-1))

            # Save individual images
            image_path = osp.join(image_dir, f"frame_{batch_idx:04d}.png")
            iio.imwrite(
                image_path,
                (combined_img.numpy() * 255).astype(np.uint8)
            )



            # Save depth images
            depth_colormap = apply_depth_colormap(
                ref_pred_depth, near_plane=depth_min, far_plane=depth_max
            )
            depth_image_path = osp.join(image_dir, f"depth_{batch_idx:04d}.png")
            iio.imwrite(
                depth_image_path,
                (depth_colormap.numpy() * 255).astype(np.uint8)
            )

            # Save mask images if available
            if len(masks) > 0:
                mask_image = masks[-1]
                mask_image_path = osp.join(image_dir, f"mask_{batch_idx:04d}.png")
                iio.imwrite(
                    mask_image_path,
                    (mask_image.numpy() * 255).astype(np.uint8)
                )