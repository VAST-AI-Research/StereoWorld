"""Inference-only subset; unused upstream utilities have been removed.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

def batch_np_inv(view_mats: np.ndarray) -> np.ndarray:
    """
    Invert a batch of homogeneous transform matrices (rigid [R|T]).
    Args:
        view_mats: Array of shape [..., 4, 4], assumed to be rigid transforms.
    
    Returns:
        inv_mats: Array of shape [..., 4, 4], the inverses.
    """
    assert view_mats.shape[-2:] == (4, 4), "Input must be [..., 4, 4]"

    R = view_mats[..., :3, :3]   # [..., 3, 3]
    T = view_mats[..., :3, 3:]   # [..., 3, 1]

    # R_inv = R.transpose(-1, -2)  # R^T
    R_inv = R.swapaxes(-1, -2)  # R^T
    T_inv = -R_inv @ T           # -R^T T

    inv_mats = np.zeros_like(view_mats)
    inv_mats[..., :3, :3] = R_inv
    inv_mats[..., :3, 3:] = T_inv
    inv_mats[..., 3, 3] = 1.0

    return inv_mats


def smooth_camera_trajectory_gaussian(c2w, sigma=2.0):
    """
    使用高斯滤波平滑相机轨迹
    """
    translations = c2w[:, :3, 3]
    smoothed_t = gaussian_filter1d(translations, sigma=sigma, axis=0, mode='nearest')

    rotations = R.from_matrix(c2w[:, :3, :3])
    quats = rotations.as_quat()
    smoothed_quats = gaussian_filter1d(quats, sigma=sigma, axis=0, mode='nearest')
    smoothed_quats /= np.linalg.norm(smoothed_quats, axis=1, keepdims=True)

    smoothed_R = R.from_quat(smoothed_quats).as_matrix()

    smoothed_c2w = np.eye(4)[None].repeat(len(c2w), axis=0)
    smoothed_c2w[:, :3, :3] = smoothed_R
    smoothed_c2w[:, :3, 3] = smoothed_t
    return smoothed_c2w
