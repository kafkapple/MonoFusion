

from PIL import Image
import numpy as np
import os
import imageio
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd

def interpolate_extrinsics(extrinsics1, extrinsics2, alpha):
    """
    Interpolate between two extrinsics matrices (4x4) by factor alpha.
    Returns the interpolated extrinsics matrix.
    """
    # import pytorch3d.transforms as transforms
    # Extract rotation matrices and translation vectors
    
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]

    # Convert rotation matrices to quaternions
    quat1 = transforms.matrix_to_quaternion(R1)
    quat2 = transforms.matrix_to_quaternion(R2)

    # Perform SLERP on quaternions
    quat_interp = slerp(quat1, quat2, alpha)
    quat_interp = quat_interp / quat_interp.norm()  
    R_interp = transforms.quaternion_to_matrix(quat_interp)

    # Interpolate translation vectors
    t_interp = (1 - alpha) * t1 + alpha * t2

    # Combine into extrinsics matrix
    extrinsics_interp = torch.eye(4, dtype=extrinsics1.dtype).to(extrinsics1.device)
    extrinsics_interp[:3, :3] = R_interp
    extrinsics_interp[:3, 3] = t_interp

    return extrinsics_interp


def get_preset_data(size, seq='_dance'):
    df = pd.read_csv(f'/data3/zihanwa3/Capstone-DSR/Processing{seq}/trajectory/gopro_calibs.csv')[:]
    # Define the poses
    poses = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam', 'qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam',]].values.tolist()
        # pose: [tx, ty, tz, qw, qx, qy, qz]
    # 3.  [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
    intrinsics = df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist()
    #       3.         [1764.426025, 1764.426025, 1920.0, 1080.0],

    poses=poses[:]
    intrinsics= intrinsics[:]
    # Convert pose data (tx, ty, tz, qx, qy, qz, qw) to 4x4 transformation matrices

    pose_matrices = poses

    # Convert the intrinsics to 3x3 matrix format
    def convert_intrinsics_to_matrix(intrinsics, size):
        _, _, fx, fy, cx, cy = intrinsics
        cx -= 0.5
        cy -= 0.5
        ratio = size/3840
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
    return pose_matrices, intrinsics_matrices