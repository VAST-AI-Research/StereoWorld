"""Camera utilities for inference.

Implements action-to-camera conversion:
  action_seq + speed -> per-frame poses -> SLERP interpolation -> viewmats + K
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from einops import rearrange


# --- Action mapping ---
ACTION_DICT = {
    "w": "forward", "s": "backward",
    "a": "left", "d": "right",
    "j": "left_rot", "l": "right_rot",
    "i": "up_rot", "k": "down_rot",
}

TRANSLATION_BASE_UNIT = 1.0
ROTATION_BASE_UNIT = 10.0


# --- Camera class ---
class Camera:
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


# --- Motion computation ---
def _compute_translation_step(motion_type, current_pose, translation_value, duration):
    if motion_type in ['forward', 'backward']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        pitch_rad = np.radians(current_pose['rotation'][0])
        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        direction = 1 if motion_type == 'forward' else -1
        return forward_vec * translation_value * direction / duration
    elif motion_type in ['left', 'right']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        right_vec = np.array([math.cos(yaw_rad), 0, math.sin(yaw_rad)])
        direction = -1 if motion_type == 'left' else 1
        return right_vec * translation_value * direction / duration
    return np.zeros(3)


def _compute_rotation_step(motion_type, rotation_value, duration):
    if motion_type.endswith('rot'):
        axis = motion_type.split('_')[0]
        total_rotation = np.zeros(3)
        if axis == 'left':
            total_rotation[1] = rotation_value
        elif axis == 'right':
            total_rotation[1] = -rotation_value
        elif axis == 'up':
            total_rotation[0] = -rotation_value
        elif axis == 'down':
            total_rotation[0] = rotation_value
        return total_rotation / duration
    return np.zeros(3)


def euler_to_quaternion(angles):
    """Convert [pitch, yaw, roll] in degrees to quaternion [qw, qx, qy, qz]."""
    pitch, yaw, roll = np.radians(angles)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * sp * cr + sy * cp * sr
    qy = sy * cp * cr - cy * sp * sr
    qz = cy * cp * sr - sy * sp * cr
    return [qw, qx, qy, qz]


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])


# --- Trajectory generation ---
def generate_composite_motion_segment(current_pose, motion_types, translation_value, rotation_value, duration):
    if isinstance(motion_types, str):
        motion_types = [motion_types]

    translation_step = np.zeros(3)
    rotation_step = np.zeros(3)
    for mt in motion_types:
        translation_step += _compute_translation_step(mt, current_pose, translation_value, duration)
        rotation_step += _compute_rotation_step(mt, rotation_value, duration)

    positions, rotations = [], []
    for i in range(1, duration + 1):
        positions.append((current_pose['position'] + translation_step * i).copy())
        rotations.append((current_pose['rotation'] + rotation_step * i).copy())

    current_pose['position'] = positions[-1].copy()
    current_pose['rotation'] = rotations[-1].copy()
    return positions, rotations, current_pose


def action_to_poses(action_seq, action_speed_list, video_length):
    """Convert action sequence to per-frame pose strings.

    Args:
        action_seq: list of action strings, e.g. ["w", "dj"]
        action_speed_list: list of speed values per segment
        video_length: total number of video frames (e.g. 121)

    Returns:
        list of pose strings (length = video_length)
    """
    duration = math.ceil(video_length / len(action_seq))
    all_positions, all_rotations = [], []
    current_pose = {'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.array([0.0, 0.0, 0.0])}
    intrinsic = [0.8, 0.5, 0.5, 0.5]

    for idx, action_id in enumerate(action_seq):
        keys = list(action_id)
        motion_types = [ACTION_DICT[key] for key in keys]
        speed = action_speed_list[idx]
        positions, rotations, current_pose = generate_composite_motion_segment(
            current_pose, motion_types,
            translation_value=speed * TRANSLATION_BASE_UNIT,
            rotation_value=speed * ROTATION_BASE_UNIT,
            duration=duration,
        )
        all_positions.extend(positions)
        all_rotations.extend(rotations)

    # Frame 0: identity
    pose_list = []
    row = [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pose_list.append(" ".join(map(str, row)))

    for i, (pos, rot) in enumerate(zip(all_positions, all_rotations)):
        quat = euler_to_quaternion(rot)
        R = quaternion_to_rotation_matrix(quat)
        t = -R @ pos
        extrinsic = np.hstack([R, t.reshape(3, 1)])
        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))

    return pose_list[:video_length]


# --- SLERP interpolation ---
def interpolate_camera_poses(cam_params, src_indices, tgt_indices):
    """Interpolate camera poses using SLERP (rotation) + LERP (translation)."""
    src_indices = np.asarray(src_indices, dtype=np.float64)
    tgt_indices = np.asarray(tgt_indices, dtype=np.float64)

    src_rot_mat = np.array([cam.w2c_mat[:3, :3] for cam in cam_params])
    src_trans_vec = np.array([cam.w2c_mat[:3, 3] for cam in cam_params])

    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat

    interp_func_trans = interp1d(src_indices, src_trans_vec, axis=0, kind='linear',
                                  bounds_error=False, fill_value="extrapolate")
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_mat = slerp_func_rot(tgt_indices).as_matrix()

    if flip_handedness:
        interpolated_rot_mat = interpolated_rot_mat @ flip_mat

    ref_cam = cam_params[0]
    result = []
    for i in range(len(tgt_indices)):
        w2c_3x4 = np.hstack([interpolated_rot_mat[i], interpolated_trans_vec[i].reshape(3, 1)])
        entry = np.zeros(19, dtype=np.float32)
        entry[1:5] = [ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy]
        entry[7:] = w2c_3x4.reshape(12)
        result.append(Camera(entry))
    return result


# --- Relative pose ---
def get_relative_pose(cam_params):
    abs_w2cs = [cam.w2c_mat for cam in cam_params]
    abs_c2ws = [cam.c2w_mat for cam in cam_params]
    target_cam_c2w = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ c2w for c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def _invert_SE3(transforms):
    """Invert batch of 4x4 SE(3) matrices."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# --- Main entry point ---
def build_control_camera_from_action(action_seq, action_speed_list, video_length):
    """Generate control_camera_video from action sequence.

    Args:
        action_seq: list of action strings, e.g. ["w", "dj"]
        action_speed_list: list of speed values per segment
        video_length: total video frames (e.g. 121)

    Returns:
        dict with "viewmats" [T_latent, 4, 4] and "K" [T_latent, 3, 3]
    """
    # Step 1: action → per-frame pose strings
    pose_list = action_to_poses(action_seq, action_speed_list, video_length)

    # Step 2: parse pose strings → Camera objects
    poses_parsed = [[float(x) for x in pose.split(' ')] for pose in pose_list]
    cam_params = [Camera(p) for p in poses_parsed]

    # Step 3: SLERP interpolation to T_latent frames
    n_frames = len(cam_params)
    T_latent = 1 + (n_frames - 1) // 4
    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, T_latent)
    cam_params = interpolate_camera_poses(cam_params, src_indices, tgt_indices)

    # Step 4: relative pose (frame 0 = identity)
    c2w_poses = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2w_poses, dtype=torch.float32)

    # Step 5: c2w → w2c (viewmats)
    viewmats = _invert_SE3(c2ws)

    # Step 6: fixed normalized intrinsics
    fx_norm = 969.6969696969696 / (960.0 * 2)  # ≈ 0.5051
    fy_norm = 969.6969696969696 / (540.0 * 2)  # ≈ 0.8979
    K = torch.zeros(T_latent, 3, 3)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 2, 2] = 1.0

    # Step 7: latent frame indices
    timestep = torch.arange(T_latent, dtype=torch.long)

    return {"viewmats": viewmats, "K": K, "timestep": timestep}


def build_control_stereo_camera_from_action(action_seq, action_speed_list, video_length, baseline=0.065):
    """Generate stereo control_camera_video from action sequence.

    Left eye (view 1) follows the action trajectory. Right eye (view 2) is
    offset by ``baseline`` along the camera-local +X axis (right direction in
    OpenCV convention).

    Args:
        action_seq: list of action strings, e.g. ["w", "dj"]
        action_speed_list: list of speed values per segment
        video_length: total video frames (e.g. 121)
        baseline: stereo baseline distance (right eye offset in +X)

    Returns:
        dict with keys:
            viewmats1: [T_latent, 4, 4] — left eye w2c
            viewmats2: [T_latent, 4, 4] — right eye w2c
            K1: [T_latent, 3, 3] — left eye normalized intrinsics
            K2: [T_latent, 3, 3] — right eye normalized intrinsics
            timestep1: [T_latent] — latent frame indices for left eye
            timestep2: [T_latent] — latent frame indices for right eye
    """
    # Build left eye (same as mono)
    pose_list = action_to_poses(action_seq, action_speed_list, video_length)
    poses_parsed = [[float(x) for x in pose.split(' ')] for pose in pose_list]
    cam_params = [Camera(p) for p in poses_parsed]

    n_frames = len(cam_params)
    T_latent = 1 + (n_frames - 1) // 4
    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, T_latent)
    cam_params = interpolate_camera_poses(cam_params, src_indices, tgt_indices)

    # Relative c2w (frame 0 = identity)
    c2w_left = get_relative_pose(cam_params)
    c2ws_left = torch.as_tensor(c2w_left, dtype=torch.float32)  # [T_latent, 4, 4]

    # Right eye: offset along camera-local +X by baseline
    # In c2w, the camera X-axis is the first column of the rotation matrix
    # right_c2w = left_c2w @ T, where T is a translation of [baseline, 0, 0] in camera space
    T_offset = torch.eye(4, dtype=torch.float32)
    T_offset[0, 3] = baseline  # +X offset in camera local frame
    c2ws_right = c2ws_left @ T_offset  # [T_latent, 4, 4]

    # c2w → w2c
    viewmats1 = _invert_SE3(c2ws_left)
    viewmats2 = _invert_SE3(c2ws_right)

    # Fixed normalized intrinsics (same for both eyes)
    fx_norm = 969.6969696969696 / (960.0 * 2)
    fy_norm = 969.6969696969696 / (540.0 * 2)
    K = torch.zeros(T_latent, 3, 3)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 2, 2] = 1.0

    # Latent frame indices
    timestep = torch.arange(T_latent, dtype=torch.long)

    return {
        "viewmats1": viewmats1,
        "viewmats2": viewmats2,
        "K1": K.clone(),
        "K2": K.clone(),
        "timestep1": timestep.clone(),
        "timestep2": timestep.clone(),
    }


# --- Raymap ---

def _batch_rigid_inv(view_mats):
    """Invert batch of [*, 4, 4] rigid transforms."""
    R = view_mats[..., :3, :3]
    T = view_mats[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    T_inv = -R_inv @ T
    inv = torch.zeros_like(view_mats)
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3:] = T_inv
    inv[..., 3, 3] = 1.0
    return inv


def extrinsic_to_raymap(
    extrinsic, intrinsic,
    ray_o_scale_factor=10.0, dmax=1.0,
    H=480, W=720, vae_downsample=8, align_corners=False,
):
    """Convert w2c extrinsics + normalized intrinsics to raymap [N, 6, H_down, W_down]."""
    is_numpy = isinstance(extrinsic, np.ndarray)
    if is_numpy:
        extrinsic = torch.from_numpy(extrinsic).float()
        intrinsic = torch.from_numpy(intrinsic).float()

    camera_pose = _batch_rigid_inv(extrinsic)

    # Build per-pixel ray directions from intrinsics
    fu = intrinsic[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fv = intrinsic[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cu = intrinsic[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cv = intrinsic[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    u = (u / W).unsqueeze(0).expand(intrinsic.shape[0], -1, -1).to(intrinsic.device)
    v = (v / H).unsqueeze(0).expand(intrinsic.shape[0], -1, -1).to(intrinsic.device)

    z_cam = torch.ones_like(u)
    x_cam = (u - cu) / fu
    y_cam = (v - cv) / fv
    ones = torch.ones_like(u)
    raymap_cam = torch.stack((x_cam, y_cam, z_cam, ones), dim=-1)  # [N, H, W, 4]

    T, rh, rw, _ = raymap_cam.shape
    raymap_cam = rearrange(raymap_cam, "t h w c -> t c (h w)")

    _camera_pose = camera_pose.clone()
    _camera_pose[:, :3, 3] = 0.0
    raymap_world = torch.bmm(_camera_pose, raymap_cam)
    raymap_world = rearrange(raymap_world, "t c (h w) -> t c h w", h=rh, w=rw)

    if vae_downsample != 1:
        raymap_world = F.interpolate(
            raymap_world, scale_factor=1 / vae_downsample,
            mode="bilinear", align_corners=align_corners,
        )

    ray_d = raymap_world[:, :3]
    ray_o = camera_pose[:, :3, 3].unsqueeze(-1).unsqueeze(-1).expand_as(ray_d)
    raymap = torch.cat([ray_d, ray_o], dim=1)  # [N, 6, H_down, W_down]

    if is_numpy:
        raymap = raymap.cpu().numpy()
    return raymap

