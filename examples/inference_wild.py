import os
import sys
from pathlib import Path

# Use the source bundled with this checkout, even if other copies are installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "Diff-Factory" / "src"),
                str(REPO_ROOT / "UniDataset" / "src")]
import json

import numpy as np
import torch
from PIL import Image as PILImage

from diffactory.pipelines.video_synthesis.pipeline_video_i2v_prope import VideoPipeline
from diffactory.models.transformers.wan_prope_transformer import WanTransformer3DPropeModel

from diffusers.utils import export_to_video
from process_utils import prepare_condition_info, postprocess, get_output_visualizations

from UniDataset.utils.geometry_utils import crop_and_resize
from UniDataset.utils.camera_utils_np import batch_np_inv, smooth_camera_trajectory_gaussian

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Paper checkpoint inference from calibrated stereo images and camera trajectories.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to a single data folder containing left.png, right.png, camera.json, caption.txt, trajectory.json")
    parser.add_argument("--ckpt_dir", type=str, default="pipeline")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--use_prope", action="store_true")
    parser.add_argument("--use_rowattn", action="store_true")
    parser.add_argument("--expand_raymap", action="store_true")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    parser.add_argument("--scale_ratio", type=float, default=10.0)
    parser.add_argument("--num_interpolation_frames", type=int, default=49)
    return parser.parse_args()


def load_images(folder_path):
    left_image = np.array(PILImage.open(os.path.join(folder_path, "left.png")).convert("RGB")).astype(np.float32)
    right_image = np.array(PILImage.open(os.path.join(folder_path, "right.png")).convert("RGB")).astype(np.float32)
    return left_image.transpose(2, 0, 1), right_image.transpose(2, 0, 1)  # [3, H, W]


def get_camera_info(folder_path):
    with open(os.path.join(folder_path, "camera.json"), "r") as f:
        data = json.load(f)
    left_extrinsics = np.array(data["left_extrinsics"])   # [4, 4]
    right_extrinsics = np.array(data["right_extrinsics"])  # [4, 4]
    left_intrinsics = np.array(data["left_intrinsics"])    # [3, 3]
    right_intrinsics = np.array(data["right_intrinsics"])  # [3, 3]
    return left_extrinsics, right_extrinsics, left_intrinsics, right_intrinsics


def get_trajectory(folder_path, left_init_extrinsics=None, right_init_extrinsics=None):
    with open(os.path.join(folder_path, "trajectory.json"), "r") as f:
        data = json.load(f)
    if "left" in data and "right" in data:
        left_extrinsics = np.array(data["left"])   # [N, 4, 4]
        right_extrinsics = np.array(data["right"])  # [N, 4, 4]
        action = data.get("action", "default")
    else:
        raise NotImplementedError("trajectory.json must contain 'left' and 'right' keys")
    return left_extrinsics, right_extrinsics, action


def load_batch(folder_path, target_h, target_w, scale_ratio, num_interpolation_frames):
    left_image, right_image = load_images(folder_path)
    H, W = left_image.shape[1], left_image.shape[2]

    with open(os.path.join(folder_path, "caption.txt"), "r") as f:
        caption = f.read().strip()

    left_init_ext, right_init_ext, left_K, right_K = get_camera_info(folder_path)
    left_extrinsics, right_extrinsics, action = get_trajectory(folder_path, left_init_ext, right_init_ext)

    # crop and resize images, adjust intrinsics
    resized_left_image, new_left_K = crop_and_resize(left_K, left_image, H, W, target_h, target_w)
    resized_right_image, new_right_K = crop_and_resize(right_K, right_image, H, W, target_h, target_w)

    # convert w2c extrinsics to c2w, scale translation
    left_c2ws = batch_np_inv(left_extrinsics)   # [Nf, 4, 4]
    right_c2ws = batch_np_inv(right_extrinsics)  # [Nf, 4, 4]

    left_c2ws[:, :3, 3] /= scale_ratio
    right_c2ws[:, :3, 3] /= scale_ratio

    # select first N frames
    left_c2ws = left_c2ws[:num_interpolation_frames]
    right_c2ws = right_c2ws[:num_interpolation_frames]

    # smooth camera trajectory
    left_c2ws = smooth_camera_trajectory_gaussian(left_c2ws)
    right_c2ws = smooth_camera_trajectory_gaussian(right_c2ws)

    # convert back to w2c
    normed_left_extrinsics = batch_np_inv(left_c2ws)   # [Nf, 4, 4]
    normed_right_extrinsics = batch_np_inv(right_c2ws)  # [Nf, 4, 4]

    # normalize to first frame (make first frame the reference)
    ref_c2w = batch_np_inv(normed_left_extrinsics[:1])  # [1, 4, 4]
    normed_left_extrinsics = normed_left_extrinsics @ ref_c2w
    normed_right_extrinsics = normed_right_extrinsics @ ref_c2w

    Nf = len(normed_left_extrinsics)
    new_left_K = new_left_K[None].repeat(Nf, axis=0)    # [Nf, 3, 3]
    new_right_K = new_right_K[None].repeat(Nf, axis=0)  # [Nf, 3, 3]

    scene_name = os.path.basename(os.path.normpath(folder_path))

    return {
        "left_image": resized_left_image.astype(np.float32),
        "right_image": resized_right_image.astype(np.float32),
        "left_extrinsics": normed_left_extrinsics.astype(np.float32),
        "right_extrinsics": normed_right_extrinsics.astype(np.float32),
        "left_intrinsics": new_left_K.astype(np.float32),
        "right_intrinsics": new_right_K.astype(np.float32),
        "scene_name": scene_name,
        "caption": caption,
        "action": action,
    }


def main():
    args = parse_args()
    if args.num_views not in (1, 2):
        raise ValueError("--num_views must be 1 or 2")
    if args.use_rowattn and (args.num_views != 2 or not args.use_prope):
        raise ValueError("--use_rowattn requires --num_views 2 and --use_prope")
    if args.H <= 0 or args.W <= 0 or args.H % 32 or args.W % 32:
        raise ValueError("--H and --W must be positive multiples of 32")
    if args.num_interpolation_frames < 1 or (args.num_interpolation_frames - 1) % 4:
        raise ValueError("--num_interpolation_frames must be 1 + 4*k")
    if args.scale_ratio <= 0:
        raise ValueError("--scale_ratio must be positive")
    for name in ("left.png", "right.png", "camera.json", "caption.txt", "trajectory.json"):
        if not (Path(args.input_dir) / name).is_file():
            raise FileNotFoundError(Path(args.input_dir) / name)
    left, right, _ = get_trajectory(args.input_dir)
    if len(left) != len(right) or len(left) < args.num_interpolation_frames:
        raise ValueError("Left/right trajectories must have equal length and cover the requested frames")
    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    target_h, target_w = args.H, args.W

    # Prepare the pipeline
    device = "cuda"
    dtype = torch.bfloat16

    transformer = WanTransformer3DPropeModel.from_pretrained(
        ckpt_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        enable_init_custom=True,
        use_prope=args.use_prope,
        use_causal=False,
        use_rowattn=args.use_rowattn,
        expand_raymap=args.expand_raymap,
    )

    pipe: VideoPipeline = VideoPipeline.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
        transformer=transformer,
    ).to(device)

    eval_kwargs = {
        "height": target_h,
        "width": target_w,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "negative_prompt": (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        ),
    }

    # Load and process a single data folder
    batch = load_batch(
        folder_path=args.input_dir,
        target_h=target_h,
        target_w=target_w,
        scale_ratio=args.scale_ratio,
        num_interpolation_frames=args.num_interpolation_frames,
    )
    batch = torch.utils.data.default_collate([batch])
    batch = {k: v.to(device).to(dtype) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(device).to(dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(device).to(dtype)

    processed_batch = prepare_condition_info(
        batch, pipe.vae, latents_mean, latents_std,
        stage="test", use_raymap=args.expand_raymap, num_views=args.num_views,
    )

    output = pipe(
        image_left=processed_batch["context_latent"],
        image_right=None,
        prompt=processed_batch["prompts"],
        left_extrinsics=processed_batch["extrinsics"],
        right_extrinsics=None,
        left_intrinsics=processed_batch["intrinsics"],
        right_intrinsics=None,
        raymap=processed_batch["raymap"],
        frame_ids=processed_batch["frame_ids"],
        condition_latents=None,
        num_frames=len(batch["left_extrinsics"][0]),
        num_views=processed_batch["num_views"],
        generator=torch.Generator(device=device).manual_seed(42),
        output_type="latent",
        return_dict=False,
        **eval_kwargs,
    )[0]  # BCFHW

    outputs = postprocess(output.to(dtype), processed_batch, pipe, latents_mean, latents_std)
    export_to_video(
        get_output_visualizations(batch, device=device, dtype=dtype, outputs=outputs, with_target=False),
        f'{output_dir}/{batch["scene_name"][0]}.mp4',
        fps=16,
    )
    print(f"saved video to {output_dir}/{batch['scene_name'][0]}.mp4")


if __name__ == "__main__":
    main()
