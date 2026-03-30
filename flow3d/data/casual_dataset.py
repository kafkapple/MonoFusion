# no-split: upstream MonoFusion framework file — splitting breaks framework import structure
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, cast
from torch.utils.data import Dataset
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm
from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    UINT16_MAX,
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import json 
@dataclass
class DavisDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = "480p"
    image_type: str = "JPEGImages"
    mask_type: str = "Annotations"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False


@dataclass
class CustomDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: str = "modest"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    video_name: str = ''
    super_fast: bool = False



class CasualDataset(BaseDataset):
    def __init__(
        self,
        seq_name: str,
        root_dir: str,
        start: int = 0,
        end: int = -1,
        res: str = "480p",
        image_type: str = "JPEGImages",
        mask_type: str = "Annotations",
        depth_type: str = "aligned_depth_anything",
        camera_type: Literal["droid_recon"] = "droid_recon",
        track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir",
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        video_name: str = '_bike',
        super_fast: bool=False,
        **_,
    ):
        super().__init__()

        self.seq_name = seq_name
        self.root_dir = root_dir
        self.res = res
        self.depth_type = depth_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius

        self.root_path = Path(root_dir)
        self.tgt_name = seq_name.split('_')[0]
        self.video_name = video_name

        def _candidate_seq_paths(modality_names: list[str]) -> list[Path]:
            candidates: list[Path] = []
            for modality in modality_names:
                if not modality:
                    continue
                base = self.root_path / modality
                candidates.append(base / res / seq_name)
                candidates.append(base / seq_name)
            return candidates

        def _resolve_seq_path(modality_names: list[str], desc: str) -> Path:
            for candidate in _candidate_seq_paths(modality_names):
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"Could not locate {desc} for sequence '{seq_name}' under '{self.root_path}'."
            )

        self.img_dir = _resolve_seq_path([image_type, "images"], "image directory")
        image_files = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in {self.img_dir}.")
        self.img_ext = image_files[0].suffix

        self.feat_dir = _resolve_seq_path(
            ["dinov2_features", "dinov2features", "features", "feature"],
            "feature directory",
        )
        self.feat_ext = ".npy"

        mask_candidates = [mask_type, "sam_v2_dyn_mask", "masks"]
        self.mask_dir = _resolve_seq_path(mask_candidates, "dynamic mask directory")

        track_candidates: list[str] = []
        for name in ("tapir", track_2d_type, "bootstapir"):
            if name and name not in track_candidates:
                track_candidates.append(name)
        self.tracks_dir = _resolve_seq_path(track_candidates, "track directory")

        def _resolve_depth_dir() -> Path:
            aligned_root = self.root_path / "aligned_moge_depth"
            if not aligned_root.exists():
                return aligned_root / seq_name / "depth"

            depth_candidates: list[Path] = []
            if video_name:
                stripped = video_name.lstrip("_")
                if stripped:
                    depth_candidates.append(aligned_root / stripped / seq_name / "depth")
                    parts = [part for part in stripped.split("_") if part]
                    if len(parts) > 1:
                        depth_candidates.append(
                            aligned_root / "_".join(parts[1:]) / seq_name / "depth"
                        )
            depth_candidates.append(aligned_root / seq_name / "depth")
            depth_candidates.append(aligned_root / seq_name)

            for candidate in depth_candidates:
                if candidate.exists():
                    return candidate
            # Fall back to the first candidate to give a meaningful error later if absent
            return depth_candidates[0]

        self.depth_dir = _resolve_depth_dir()
        self.cache_dir = self.root_path / "flow3d_preprocessed" / seq_name

        numeric_frame_ids = []
        for img_path in image_files:
            try:
                numeric_frame_ids.append(int(img_path.stem))
            except ValueError:
                continue
        if numeric_frame_ids:
            self.min_ = min(numeric_frame_ids)
            self.max_ = max(numeric_frame_ids)
        else:
            self.min_, self.max_ = 0, max(len(image_files) - 1, 0)


        # self.camera_path
        self.hard_indx_dict = {
          '_bike': [49, 349, 3], 
          '_dance': [1477, 1778, 3],
          '': [273, 294, 1],
        }

        print(self.min_, self.max_)
        self.glb_first_indx = self.min_ #self.hard_indx_dict[self.video_name][0]
        self.glb_last_indx = self.max_ #self.hard_indx_dict[self.video_name][1]
        self.glb_step = 1 if 'm5t2' in (video_name or '') else 3

        frame_names = [p.stem for p in image_files]
        if super_fast:
          frame_names=frame_names[-22:]
        #print(frame_names)
        #print(self.video_name)

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        if 'm5t2' in (self.video_name or ''):
          self.frame_names = frame_names[start:end:self.glb_step]
        elif self.video_name=='_bike':
          self.frame_names = frame_names[start:end:self.glb_step][:-1]
        elif self.video_name=='_dance':
          self.frame_names = frame_names[start:end:self.glb_step]#[:-1]
        elif self.video_name=='' or 'soccer' in self.video_name or 'bike' in self.video_name or 'dance' in self.video_name:
          self.frame_names = frame_names[start:end:self.glb_step][:-1]
        else:
          self.frame_names = frame_names[start:end:self.glb_step]

        frame_names=self.frame_names
        # print(self.start, self.end)

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.feats: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]



        self.debug=False


        def load_known_cameras_panoptic(
            path: str, H: int, W: int, noise: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          import json
          import numpy as np
          size = W
          
          seq = seq_name.split('_')[0]
          json_path = f'data/{seq}/train_meta.json'

          with open(json_path, 'r') as f:
              df = json.load(f)
          # 0 10 15 30
          cam_ids = [3, 21, 23, 25]
          for i, kv in enumerate(cam_ids):
              if kv > 30:
                  k = 4
              elif kv > 15:
                  k = 3
              elif kv > 10:
                  k = 2
              else:
                  k = 1

              cam_ids[i] = kv - k 

          intrinsics = np.array([df['k'][0][i] for i in cam_ids])
          w2cs = np.array([df['w2c'][0][i] for i in cam_ids])
          # print(np.array(intrinsics).shape, np.array(w2cs).shape)


          def k_to_intrinsics(k_matrix, image_width, image_height):
              """
              Convert a 3x3 K matrix to [image_width, image_height, fx, fy, cx, cy].
              """
              fx = k_matrix[0, 0]
              fy = k_matrix[1, 1]
              cx = k_matrix[0, 2]
              cy = k_matrix[1, 2]
              
              return [image_width, image_height, fx, fy, cx, cy]

          intrinsics_list = [k_to_intrinsics(intrinsics[i], 512, 288) 
                            for i in range(len(intrinsics))]
          #poses = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam', 'qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam',  ]].values.tolist()
              # pose: [tx, ty, tz, qw, qx, qy, qz]
          # 3.  [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
          # intrinsics = df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist()
          #       3.         [1764.426025, 1764.426025, 1920.0, 1080.0],

          poses=(w2cs[:]) # 
          intrinsics= intrinsics_list[:]

          def convert_to_matrix(pose):
              tx, ty, tz, qw, qx, qy, qz = pose
              rotation = R.from_quat([qx, qy, qz, qw])
              rotation_matrix = rotation.as_matrix()
              #(x, y, z, w)
              
              transformation_matrix = np.eye(4)
              transformation_matrix[:3, :3] = rotation_matrix
              transformation_matrix[:3, 3] = [tx, ty, tz]
              
              return transformation_matrix
          pose_matrices = poses# [(convert_to_matrix(pose)) for pose in poses]
          # print(w2cs.shape)

          def convert_intrinsics_to_matrix(intrinsics, size):
              _, _, fx, fy, cx, cy = intrinsics
              ratio = size/640
              fx *= ratio
              fy *= ratio
              cx *= ratio
              cy *= ratio
              intrinsics_matrix = np.array([
                  [fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]
              ])
              
              return intrinsics_matrix

          intrinsics_matrices = [convert_intrinsics_to_matrix(intrinsics, size) for intrinsics in intrinsics]

          pose_matrices = np.array(pose_matrices)
          intrinsics_matrices = np.array(intrinsics_matrices)

          cam_iddddds = {
            '3': 0,
            '21': 1, 
            '23': 2,
            '25': 3,
          }
          c = int(self.seq_name[-2:])
          posessss, intrinsicsss = [], []
          for t in range(self.glb_first_indx, self.glb_last_indx, self.glb_step):
            pose_matrice, intrinsics_matrice = pose_matrices[cam_iddddds[str(c)]], intrinsics_matrices[cam_iddddds[str(c)]]
            posessss.append(pose_matrice)
            intrinsicsss.append(intrinsics_matrice)


          pose_matrices = torch.tensor(posessss).float()
          intrinsics_matrices = torch.tensor(intrinsicsss).float()


        
          return pose_matrices, intrinsics_matrices, None 

        def load_known_cameras(
            path: str, H: int, W: int, noise: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert os.path.exists(path), f"Camera file {path} does not exist."
            md = json.load(open(path, 'r'))
            c2ws = []
            # print(md['hw'].shape)
            print(np.array(md['k']).shape)
            print(np.array(md['w2c']).shape)

            #for c in range(4, 5):
            c = int(self.seq_name[-1])
            #for t in range(self.glb_first_indx, self.glb_last_indx, self.glb_step):
            T = len(md['k'])
            for t in range(0, T, self.glb_step):
              h, w = md['hw'][c]
              k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
              if noise:
                R = w2c[:3, :3]
                t = w2c[:3, 3]

                # Define the maximum deviation in degrees and convert to radians
                max_deviation_deg = 5
                max_deviation_rad = np.deg2rad(max_deviation_deg)

                # Generate random rotation angles within ±5 degrees for each axis
                noise_angles = np.random.uniform(-max_deviation_rad, max_deviation_rad, size=3)

                # Create rotation matrices around x, y, and z axes
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(noise_angles[0]), -np.sin(noise_angles[0])],
                    [0, np.sin(noise_angles[0]),  np.cos(noise_angles[0])]
                ])

                Ry = np.array([
                    [ np.cos(noise_angles[1]), 0, np.sin(noise_angles[1])],
                    [0, 1, 0],
                    [-np.sin(noise_angles[1]), 0, np.cos(noise_angles[1])]
                ])

                Rz = np.array([
                    [np.cos(noise_angles[2]), -np.sin(noise_angles[2]), 0],
                    [np.sin(noise_angles[2]),  np.cos(noise_angles[2]), 0],
                    [0, 0, 1]
                ])

                # Combine the rotation matrices
                R_noise = Rz @ Ry @ Rx

                # Apply the rotation noise to the original rotation
                R_new = R_noise @ R

                # Construct the new w2c matrix with the noisy rotation and original translation
                w2c_new = np.eye(4)
                w2c_new[:3, :3] = R_new
                w2c_new[:3, 3] = t

                # Update w2c with the new matrix
                w2c = w2c_new

              c2ws.append(w2c[None, ...])

            traj_c2w = np.concatenate(c2ws)
            sy, sx = H / h, W / w
            fx, fy, cx, cy = k[0][0],  k[1][1], k[0][2], k[1][2], # (4,)

            K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
            Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)


            kf_tstamps=None
            return (
                torch.from_numpy(traj_c2w).float(),
                torch.from_numpy(Ks).float(),
                kf_tstamps,
            )
        

        if camera_type == "droid_recon":
            img = self.get_image(0)
            H, W = img.shape[:2]
            
            # Use absolute path via self.root_path (CWD-independent)
            path = str(self.root_path / '_raw_data' / self.video_name[1:] / 'trajectory' / 'Dy_train_meta.json')
            if self.debug:
              w2cs, Ks, tstamps = load_cameras(
                  f"{root_dir}/{camera_type}/{seq_name}.npy", H, W
              )
            else:
              #try:
              
              w2cs, Ks, tstamps = load_known_cameras(
                path, H, W, noise=False ##############################
              )
              #except:
              #  w2cs, Ks, tstamps = load_known_cameras_panoptic(
              #  path, H, W, noise=False ##############################
              #  )     

        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        assert (
            len(frame_names) == len(w2cs) == len(Ks)
        ), f"{len(frame_names)}, {len(w2cs)}, {len(Ks)}"


        self.w2cs = w2cs[start:end]
        self.Ks = Ks[start:end]
        if tstamps is None:
          self._keyframe_idcs=None
        else:
          tmask = (tstamps >= start) & (tstamps < end)
          self._keyframe_idcs = tstamps[tmask] - start
          
        self.scale = 1

        if scene_norm_dict is None:
            cached_scene_norm_dict_path = Path(self.cache_dir) / "scene_norm_dict.pth"
            if cached_scene_norm_dict_path.exists() and self.load_from_cache:
                guru.info("loading cached scene norm dict...")
                scene_norm_dict = torch.load(cached_scene_norm_dict_path)
            else:
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return self._keyframe_idcs

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_images(self):
        imgs = [cast(torch.Tensor, self.load_image(index)) for index in range(len(self.frame_names))]
        return imgs

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )
        num_query_frames = self.num_frames // stride
        query_endpts = torch.linspace(start, end, num_query_frames + 1)
        query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            feat = self.get_feat(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            bool_mask = (bg_mask * (depth > 0)).to(torch.bool)

            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # get the bounding box of previous points that reproject into frame
            # inefficient but works for now
            bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
            for p3d, _, _, _, _ in bg_geometry:
                if len(p3d) < 1:
                    continue
                # reproject into current frame
                p2d = torch.einsum(
                    "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                )
                p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                bmin_x = min(bmin_x, int(xmin))
                bmin_y = min(bmin_y, int(ymin))
                bmax_x = max(bmax_x, int(xmax))
                bmax_y = max(bmax_y, int(ymax))

            # don't include points that are covered by previous points
            bmin_x = max(0, bmin_x)
            bmin_y = max(0, bmin_y)
            bmax_x = min(W, bmax_x)
            bmax_y = min(H, bmax_y)
            overlap_mask = torch.ones_like(bool_mask)
            overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            point_feats = feat[bool_mask]
            point_sizes = depth[bool_mask] / ((K[0][0] + K[0][1])/2)


            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            point_feats = point_feats[sel_idcs]
            point_sizes = point_sizes[sel_idcs]

            
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors, point_feats, point_sizes))

        bg_points, bg_normals, bg_colors, bg_feats, bg_sizes = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]
            bg_feats = bg_feats[sel_idcs]
            bg_sizes = bg_sizes[sel_idcs]

        return bg_points, bg_normals, bg_colors, bg_feats, bg_sizes

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, 1))  # targets always all frames

        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)

        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        depths_valid_mask = (depths > 0).bool()

        # Ensure fg_masks is also a boolean tensor
        fg_masks = (masks == 1).float()
        fg_masks = fg_masks.bool()
        final_masks = (fg_masks & depths_valid_mask).numpy()#.float()
        r = 2
        kernel = np.ones((r, r), np.uint8)
        # cv2.erode requires 2D — apply per-frame if 3D
        if final_masks.ndim == 3:
            final_masks = np.stack([
                cv2.erode(final_masks[t].astype(np.uint8), kernel, iterations=1)
                for t in range(final_masks.shape[0])
            ])
        else:
            final_masks = cv2.erode(final_masks.astype(np.uint8), kernel, iterations=1)
        final_masks = torch.from_numpy(final_masks).float()

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))

        cur_num = 0
        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )
            if num_sel < len(tracks_2d):
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            feat = self.get_feat(q_idx)


            tidx = target_idcs.index(q_idx)


            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, final_masks, inv_Ks, c2ws, feat
            )
  
            tracks_all_queries.append(tracks_tuple)


        tracks_3d, colors, feats, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )


        return tracks_3d, visibles, invisibles, confidences, colors, feats

    def _get_image_hw(self) -> tuple[int, int]:
        """Get image (H, W) from first image file, cached."""
        if not hasattr(self, '_img_hw'):
            from PIL import Image
            first_img = sorted(self.img_dir.glob("*.png")) or sorted(self.img_dir.glob("*.jpg"))
            img = Image.open(first_img[0])
            self._img_hw = (img.height, img.width)
        return self._img_hw

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img

    def get_feat(self, index) -> torch.Tensor:
        if self.feats[index] is None:
            self.feats[index] = self.load_feat(index)
        feat = cast(torch.Tensor, self.feats[index])
        return feat

    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        
        mask = cast(torch.Tensor, self.masks[index])
        return mask

    def get_masks(self):
        masks = [cast(torch.Tensor, self.load_mask(index)) for index in range(len(self.frame_names))]
        return masks

    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] #/ self.scale

    def get_depths(self):
        depths = [cast(torch.Tensor, self.load_depth(index)) for index in range(len(self.frame_names))]
        return depths

    def load_image(self, index) -> torch.Tensor:
        path = self.img_dir / f"{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(str(path))).float() / 255.0

    def load_feat(self, index) -> torch.Tensor:
        feature_path = self.feat_dir / f"{self.frame_names[index]}{self.feat_ext}"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file at {feature_path}")
        dinov2_feature = np.load(feature_path).astype(np.float32)
        feat = torch.from_numpy(dinov2_feature)
        # Upsample to image resolution if needed (e.g., 37×37 → H×W)
        # Cap at 512×512 to avoid RAM explosion (1024×1152×384 = 1.7GB/frame)
        img_h, img_w = self._get_image_hw()
        tgt_h = min(img_h, 512)
        tgt_w = min(img_w, 512)
        if feat.shape[0] != tgt_h or feat.shape[1] != tgt_w:
            feat = F.interpolate(
                feat.permute(2, 0, 1).unsqueeze(0),
                size=(tgt_h, tgt_w), mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        return feat

    def load_mask(self, index) -> torch.Tensor:
        r = self.mask_erosion_radius
        base_name = self.frame_names[index]
        candidate_paths = []
        if self.debug:
            candidate_paths.append(self.mask_dir / f"{base_name}.png")
        candidate_paths.append(self.mask_dir / f"dyn_mask_{base_name}.npz")
        candidate_paths.append(self.mask_dir / f"{base_name}.npz")
        candidate_paths.append(self.mask_dir / f"{base_name}.png")

        mask_path = next((p for p in candidate_paths if p.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"No dynamic-mask file found for frame {base_name} in {self.mask_dir}")

        if mask_path.suffix.lower() == ".npz":
            with np.load(mask_path) as data:
                if "dyn_mask" in data:
                    mask = data["dyn_mask"]
                else:
                    first_key = data.files[0]
                    mask = data[first_key]
            if mask.ndim == 4:
                mask = mask[0]
            if mask.ndim == 3:
                mask = mask[0]
            mask = np.repeat(mask[..., None], 3, axis=2)
        else:
            mask = imageio.imread(str(mask_path))

        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    def load_org_depth(self, index) -> torch.Tensor:
        path = Path(self.depth_dir) / f"{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            base_path = self.tracks_dir / f"{q_name}_{t_name}"
            track_path = None
            for ext in (".npy", ".npz"):
                candidate = base_path.with_suffix(ext)
                if candidate.exists():
                    track_path = candidate
                    break
            if track_path is None:
                raise FileNotFoundError(
                    f"Missing TAPIR track file for pair ({q_name}, {t_name}) in {self.tracks_dir}"
                )
            if track_path.suffix == ".npz":
                with np.load(track_path) as data:
                    if "tracks" in data:
                        tracks = data["tracks"]
                    else:
                        tracks = data[data.files[0]]
            else:
                tracks = np.load(track_path)
            tracks = tracks.astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def load_depth(self, index) -> torch.Tensor:
        #  load_da2_depth load_duster_depth load_org_depth
        if self.depth_type == 'modest':
           depth = self.load_modest_depth(index)
        elif self.depth_type == 'moge':
           depth = self.load_moge_depth(index)
        elif self.depth_type == 'algo':
           depth = self.load_algo_depth(index)

        #  depth = self.load_panoptic_gt(index)
        return depth

    def load_moge_depth(self, index) -> torch.Tensor:
        depth_dir = Path(self.depth_dir)
        filename = f"{self.frame_names[index]}.npy"
        depth_path = depth_dir / filename
        if not depth_path.exists():
            raise FileNotFoundError(f"Aligned MoGe depth not found at {depth_path}")
        depth_map = np.load(depth_path)
        depth = torch.from_numpy(depth_map).float()
        return depth


    def load_algo_depth(self, index) -> torch.Tensor:
        # Fallback to MoGe depth if algorithm-specific depth is unavailable.
        return self.load_moge_depth(index)



    def __getitem__(self, index: int, target_inds=None):
        #index = np.random.randint(0, self.num_frames)
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.get_image(index),
            "feats": self.get_feat(index),
            "depths": self.get_depth(index),
        }
        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        if target_inds is None:
          target_inds = torch.from_numpy(
              np.random.choice(
                  self.num_frames, (self.num_targets_per_frame,), replace=False
              )
          )
        else:
          target_inds = torch.from_numpy(
              target_inds
          )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)



        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]
        return data


class CasualDatasetVideoView(Dataset):
    """Return a dataset view of the video trajectory."""

    def __init__(self, dataset: CasualDataset):
        super().__init__()
        self.dataset = dataset
        self.fps = 15
        self.imgs = self.dataset.get_images()
        self.masks = self.dataset.get_masks()
        self.depths = self.dataset.get_depths()

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, index):

        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": index,
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.imgs[index],
            "depths": self.depths[index],
            "masks": self.masks[index],
        }
