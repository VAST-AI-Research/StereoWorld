"""Inference-only subset; unused upstream utilities have been removed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

def batch_torch_inv(view_mats: torch.Tensor) -> torch.Tensor:
    """
    Invert a batch of homogeneous transform matrices (rigid [R|T]).
    
    Args:
        view_mats: Tensor of shape [..., 4, 4], assumed to be rigid transforms.
    
    Returns:
        inv_mats: Tensor of shape [..., 4, 4], the inverses.
    """
    assert view_mats.shape[-2:] == (4, 4), "Input must be [..., 4, 4]"

    R = view_mats[..., :3, :3]   # [..., 3, 3]
    T = view_mats[..., :3, 3:]   # [..., 3, 1]

    R_inv = R.transpose(-1, -2)  # R^T
    T_inv = -R_inv @ T           # -R^T T

    inv_mats = torch.zeros_like(view_mats)
    inv_mats[..., :3, :3] = R_inv
    inv_mats[..., :3, 3:] = T_inv
    inv_mats[..., 3, 3] = 1.0

    return inv_mats


def normalize_batch_intrinsics(intrinsics, H, W):
    """
    This function assumes that extrinsics are in the form of [B, N, 3, 3] where N is the number of cameras.
    """
    normed_intrinsics = intrinsics.clone()  # [B, N, 3, 3]
    normed_intrinsics[..., 0, :] /= W
    normed_intrinsics[..., 1, :] /= H
    return normed_intrinsics


def extrinsic_to_raymap(
    extrinsic,
    intrinsic,
    ray_o_scale_factor: float = 10.0,
    dmax: float = 1.0,
    H: int = 480,
    W: int = 720,
    vae_downsample: int = 8,
    align_corners: bool = False,
) -> np.ndarray:
    """
    Convert camera pose to raymap.

    Args:
        camera_pose: (N, 4, 4) camera poses
        intrinsic: (N, 3, 3) intrinsics
        ray_o_scale_factor: A constant scale factor for ray_o to avoid too large translation values.
            Default to 10.0. If you use pre-trained AetherV1 model, you should always set it to 10.0.
        dmax: A constant scale factor for ray_d to avoid too large translation values.
            It should be equal to the maximum disparity value (before sqrt) of the sequence
            if you have ground truth disparity. Default to 1.0.
    Returns:
        (N, 6, H, W) raymap
    """
    is_numpy = isinstance(extrinsic, np.ndarray)
    if is_numpy:
        extrinsic = torch.from_numpy(extrinsic).float()
        intrinsic = torch.from_numpy(intrinsic).float()
    # scale_factor = 1.0 / dmax
    # camera_pose[:, :3, 3] = signed_log1p(
    #     camera_pose[:, :3, 3] / scale_factor * ray_o_scale_factor
    # )

    camera_pose = batch_torch_inv(extrinsic)
    raymap = get_raymap_from_camera_parameters(
        intrinsic,
        camera_pose,
        H,
        W,
        vae_downsample,
        align_corners,
    )
    if is_numpy:
        raymap = raymap.cpu().numpy()
    return raymap


def get_raymap_from_camera_parameters(
    intrinsic,
    camera_pose,
    H,
    W,
    vae_downsample=8,
    align_corners=True,
):
    def get_raymap_from_trans2d(intrinsic, H, W):
        fu = intrinsic[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        fv = intrinsic[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cu = intrinsic[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cv = intrinsic[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

        u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        # NOTE fu/fv/cu/cv are all in normalized image coordinates
        u = u / W  # align with func: normalize_batch_intrinsics
        v = v / H

        u = u.unsqueeze(0).repeat(intrinsic.shape[0], 1, 1).to(intrinsic.device)
        v = v.unsqueeze(0).repeat(intrinsic.shape[0], 1, 1).to(intrinsic.device)

        z_cam = torch.ones_like(u).to(intrinsic.device)
        x_cam = (u - cu) / fu
        y_cam = (v - cv) / fv
        addition_dim = torch.ones_like(u).to(intrinsic.device)
        return torch.stack((x_cam, y_cam, z_cam, addition_dim), dim=-1)

    raymap_cam = get_raymap_from_trans2d(intrinsic, H, W).to(camera_pose.device)

    T, raymap_cam_h, raymap_cam_w, _ = raymap_cam.shape
    raymap_cam = rearrange(raymap_cam, "t h w c -> t c (h w)")

    _camera_pose = camera_pose.clone()
    _camera_pose[:, :3, 3] = 0.0
    raymap_world = torch.bmm(_camera_pose, raymap_cam)
    raymap_world = rearrange(
        raymap_world, "t c (h w) -> t c h w", h=raymap_cam_h, w=raymap_cam_w
    )

    if vae_downsample != 1:
        raymap_world = F.interpolate(
            raymap_world,
            scale_factor=1 / vae_downsample,
            mode="bilinear",
            align_corners=align_corners,
        )
    raymap_world = raymap_world[:, :3]
    ray_o = torch.ones_like(raymap_world).to(raymap_world.device) * camera_pose[
        :, :3, 3
    ].unsqueeze(-1).unsqueeze(-1)

    raymap_world = torch.cat([raymap_world, ray_o], dim=1)
    return raymap_world
