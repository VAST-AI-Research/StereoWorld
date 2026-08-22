"""Flexible stereo camera utilities for inference.

4 modes for generating the right camera trajectory:
  1. horizontal_offset — R offset along camera-local +X (with yaw vergence)
  2. depth_offset      — R offset along camera-local +Z (ahead/behind, with yaw)
  3. height_offset     — R offset along camera-local +Y (above/below, with pitch)
  4. converging        — starts separated, gradually converges (baseline shrinks + vergence grows)

All modes return: {viewmats1, viewmats2, K1, K2, timestep1, timestep2}
"""
import math
import numpy as np
import torch
from typing import List, Optional

from camera_utils import (
    action_to_poses,
    Camera,
    interpolate_camera_poses,
    get_relative_pose,
    _invert_SE3,
    extrinsic_to_raymap,
)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _build_left_trajectory(action_seq, speed_list, video_length):
    """Build left camera trajectory → c2ws [T_latent, 4, 4], T_latent."""
    pose_list = action_to_poses(action_seq, speed_list, video_length)
    poses_parsed = [[float(x) for x in pose.split(' ')] for pose in pose_list]
    cam_params = [Camera(p) for p in poses_parsed]

    n_frames = len(cam_params)
    T_latent = 1 + (n_frames - 1) // 4
    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, T_latent)
    cam_params = interpolate_camera_poses(cam_params, src_indices, tgt_indices)

    c2w_left = get_relative_pose(cam_params)
    return torch.as_tensor(c2w_left, dtype=torch.float32), T_latent


def _make_output(c2ws_left, c2ws_right, T_latent):
    """Package c2ws into the standard output dict."""
    viewmats1 = _invert_SE3(c2ws_left)
    viewmats2 = _invert_SE3(c2ws_right)

    fx_norm = 969.6969696969696 / (960.0 * 2)
    fy_norm = 969.6969696969696 / (540.0 * 2)
    K = torch.zeros(T_latent, 3, 3)
    K[:, 0, 0] = fx_norm
    K[:, 1, 1] = fy_norm
    K[:, 0, 2] = 0.5
    K[:, 1, 2] = 0.5
    K[:, 2, 2] = 1.0

    timestep = torch.arange(T_latent, dtype=torch.long)

    return {
        "viewmats1": viewmats1,
        "viewmats2": viewmats2,
        "K1": K.clone(),
        "K2": K.clone(),
        "timestep1": timestep.clone(),
        "timestep2": timestep.clone(),
    }


def _yaw_matrix(yaw_rad):
    """4×4 rotation matrix around camera Y axis (yaw)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = torch.eye(4, dtype=torch.float32)
    R[0, 0] = c;  R[0, 2] = s
    R[2, 0] = -s; R[2, 2] = c
    return R


def _pitch_matrix(pitch_rad):
    """4×4 rotation matrix around camera X axis (pitch)."""
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    R = torch.eye(4, dtype=torch.float32)
    R[1, 1] = c;  R[1, 2] = -s
    R[2, 1] = s;  R[2, 2] = c
    return R


def _translate_matrix(x=0.0, y=0.0, z=0.0):
    """4×4 translation matrix."""
    T = torch.eye(4, dtype=torch.float32)
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


# ─── Mode 1: Horizontal Offset (+X with yaw vergence) ────────────────────

def _build_horizontal_offset(action_seq_left, speed_left, action_seq_right, speed_right,
                              video_length, **kwargs):
    """Right = left + fixed offset along camera-local +X, with yaw inward.

    Classic stereo baseline: R is to the right of L, both looking slightly inward.
    """
    c2ws_left, T_latent = _build_left_trajectory(action_seq_left, speed_left, video_length)

    baseline = kwargs.get("baseline", 4.0)
    vergence_deg = kwargs.get("vergence_deg", 20.0)

    T_offset = _translate_matrix(x=baseline)
    R_yaw = _yaw_matrix(math.radians(-vergence_deg))
    combined = T_offset @ R_yaw

    c2ws_right = c2ws_left @ combined.unsqueeze(0)
    return _make_output(c2ws_left, c2ws_right, T_latent)


# ─── Mode 2: Depth Offset (+Z with optional yaw) ─────────────────────────

def _build_depth_offset(action_seq_left, speed_left, action_seq_right, speed_right,
                         video_length, **kwargs):
    """Right = left + fixed offset along camera-local +Z (in front or behind).

    R is ahead of L (positive depth_offset) or behind (negative).
    An optional small yaw gives a slight angular difference.
    """
    c2ws_left, T_latent = _build_left_trajectory(action_seq_left, speed_left, video_length)

    depth_offset = kwargs.get("depth_offset", 4.0)
    vergence_deg = kwargs.get("vergence_deg", 15.0)

    T_offset = _translate_matrix(z=depth_offset)
    R_yaw = _yaw_matrix(math.radians(-vergence_deg))
    combined = T_offset @ R_yaw

    c2ws_right = c2ws_left @ combined.unsqueeze(0)
    return _make_output(c2ws_left, c2ws_right, T_latent)


# ─── Mode 3: Height Offset (+Y with pitch) ───────────────────────────────

def _build_height_offset(action_seq_left, speed_left, action_seq_right, speed_right,
                          video_length, **kwargs):
    """Right = left + fixed offset along camera-local +Y (above or below), with pitch.

    R is above L (positive) and looks slightly downward, or below L and looks up.
    """
    c2ws_left, T_latent = _build_left_trajectory(action_seq_left, speed_left, video_length)

    height_offset = kwargs.get("height_offset", 4.0)
    pitch_deg = kwargs.get("pitch_deg", -10.0)

    T_offset = _translate_matrix(y=height_offset)
    R_pitch = _pitch_matrix(math.radians(pitch_deg))
    combined = T_offset @ R_pitch

    c2ws_right = c2ws_left @ combined.unsqueeze(0)
    return _make_output(c2ws_left, c2ws_right, T_latent)


# ─── Mode 4: Converging (starts apart, ends close + looking inward) ──────

def _build_converging(action_seq_left, speed_left, action_seq_right, speed_right,
                      video_length, **kwargs):
    """Two cameras start separated, gradually converge in both distance and gaze.

    Baseline: starts at start_baseline, shrinks to end_baseline.
    Vergence: starts at start_vergence_deg, grows to end_vergence_deg.
    This creates a natural "zooming in on the same point" effect.
    """
    c2ws_left, T_latent = _build_left_trajectory(action_seq_left, speed_left, video_length)

    start_baseline = kwargs.get("start_baseline", 5.0)
    end_baseline = kwargs.get("end_baseline", 1.5)
    start_vergence_deg = kwargs.get("start_vergence_deg", 10.0)
    end_vergence_deg = kwargs.get("end_vergence_deg", 35.0)

    t = torch.linspace(0, 1, T_latent)
    # Smooth interpolation (cosine ease)
    ease = 0.5 * (1 - torch.cos(math.pi * t))
    baseline_t = start_baseline + (end_baseline - start_baseline) * ease
    vergence_t = start_vergence_deg + (end_vergence_deg - start_vergence_deg) * ease

    c2ws_right = c2ws_left.clone()
    for i in range(T_latent):
        T_offset = _translate_matrix(x=baseline_t[i].item())
        R_yaw = _yaw_matrix(math.radians(-vergence_t[i].item()))
        c2ws_right[i] = c2ws_left[i] @ T_offset @ R_yaw

    return _make_output(c2ws_left, c2ws_right, T_latent)


# ─── Registry ────────────────────────────────────────────────────────────

MODE_REGISTRY = {
    "horizontal_offset": _build_horizontal_offset,
    "depth_offset": _build_depth_offset,
    "height_offset": _build_height_offset,
    "converging": _build_converging,
}


def build_flex_stereo_camera(action_seq_left, speed_left,
                             action_seq_right, speed_right,
                             video_length, mode="horizontal_offset", **kwargs):
    """Unified entry point for flexible stereo camera generation.

    Args:
        action_seq_left:  list of action strings for left eye
        speed_left:       list of speed values per left segment
        action_seq_right: (unused for current modes)
        speed_right:      (unused for current modes)
        video_length:     total video frames (e.g. 81, 121)
        mode:             one of: horizontal_offset, depth_offset, height_offset, converging
        **kwargs:         mode-specific parameters

    Returns:
        dict with keys: viewmats1, viewmats2, K1, K2, timestep1, timestep2
    """
    if mode not in MODE_REGISTRY:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(MODE_REGISTRY.keys())}")

    return MODE_REGISTRY[mode](action_seq_left, speed_left, action_seq_right, speed_right,
                               video_length, **kwargs)


# ─── GT Camera (from camera_sample.json) ──────────────────────────────────

def build_gt_stereo_camera(left_c2w_600, right_c2w_600, K_left, K_right,
                           image_size, video_length=81):
    """Build stereo camera directly from GT training trajectories.

    Args:
        left_c2w_600:  (600, 4, 4) ndarray
        right_c2w_600: (600, 4, 4) ndarray
        K_left:        (3, 3) ndarray — left intrinsics (pixel space)
        K_right:       (3, 3) ndarray — right intrinsics (pixel space)
        image_size:    [W, H]
        video_length:  target video frames (default 81)

    Returns:
        dict with keys: viewmats1, viewmats2, K1, K2, timestep1, timestep2
    """
    indices = np.linspace(0, len(left_c2w_600) - 1, video_length).astype(int)
    lc = left_c2w_600[indices]
    rc = right_c2w_600[indices]

    T_latent = 1 + (video_length - 1) // 4
    latent_indices = np.linspace(0, video_length - 1, T_latent).astype(int)
    lc = lc[latent_indices]
    rc = rc[latent_indices]

    c2ws_left = torch.as_tensor(lc, dtype=torch.float32)
    c2ws_right = torch.as_tensor(rc, dtype=torch.float32)

    viewmats1 = _invert_SE3(c2ws_left)
    viewmats2 = _invert_SE3(c2ws_right)

    W, H = image_size
    K_left = np.array(K_left)
    K_right = np.array(K_right)

    K1 = torch.zeros(T_latent, 3, 3)
    K1[:, 0, 0] = K_left[0, 0] / W
    K1[:, 1, 1] = K_left[1, 1] / H
    K1[:, 0, 2] = K_left[0, 2] / W
    K1[:, 1, 2] = K_left[1, 2] / H
    K1[:, 2, 2] = 1.0

    K2 = torch.zeros(T_latent, 3, 3)
    K2[:, 0, 0] = K_right[0, 0] / W
    K2[:, 1, 1] = K_right[1, 1] / H
    K2[:, 0, 2] = K_right[0, 2] / W
    K2[:, 1, 2] = K_right[1, 2] / H
    K2[:, 2, 2] = 1.0

    timestep = torch.arange(T_latent, dtype=torch.long)

    return {
        "viewmats1": viewmats1,
        "viewmats2": viewmats2,
        "K1": K1,
        "K2": K2,
        "timestep1": timestep.clone(),
        "timestep2": timestep.clone(),
    }
