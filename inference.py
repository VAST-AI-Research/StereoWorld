"""Stereo video generation with action-driven camera control.

Generates stereo (left + right) videos from a single image + text prompt,
with camera trajectories defined by WASD-style action sequences.

Usage:
    # Single example:
    python inference.py \
        --pipeline_dir /path/to/pipeline \
        --input_dir /path/to/data \
        --action_seq wk --action_speed 6 \
        --baseline 0.2

    # Batch inference from eval.json:
    python inference.py \
        --pipeline_dir /path/to/pipeline \
        --eval_json /path/to/eval.json \
        --baseline 0.2

    # Multi-GPU (sequence parallel):
    torchrun --nproc_per_node=4 inference.py \
        --pipeline_dir /path/to/pipeline \
        --eval_json /path/to/eval.json \
        --baseline 0.2 \
        --ulysses_degree 4 --ring_degree 1

    Action keys: w(forward) s(backward) a(left) d(right)
                 j(yaw left) l(yaw right) k(pitch down) i(pitch up)
    Actions can be combined: "wk" = forward + pitch down
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import torch
from PIL import Image as PILImage

from models.pipelines.pipeline_stereoworld import StereoWorldPipeline
from models.transformers.wan_transformer_3d import Wan2_2Transformer3DModel
from models.wan_vae import AutoencoderKLWan3_8
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from camera_utils import build_control_stereo_camera_from_action, extrinsic_to_raymap


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default=None,
                    help="Path to single data folder containing left.png, caption.txt")
parser.add_argument("--eval_json", type=str, default=None,
                    help="Path to eval.json for batch inference")
parser.add_argument("--pipeline_dir", type=str, required=True,
                    help="Path to pipeline directory (contains transformer/, vae/, tokenizer/, text_encoder/, scheduler/)")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--H", type=int, default=704)
parser.add_argument("--W", type=int, default=1280)
parser.add_argument("--num_frames", type=int, default=121,
                    help="Number of output frames per view, must satisfy 1+4k (e.g. 81, 121)")
parser.add_argument("--fps", type=int, default=24, help="FPS for output video")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--guidance_scale", type=float, default=3.0)
parser.add_argument("--shift", type=float, default=3.0)
parser.add_argument("--boundary", type=float, default=0.875)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--action_seq", type=str, nargs="+", default=["w"],
                    help="Action sequence for single input, e.g. --action_seq w dj")
parser.add_argument("--action_speed", type=float, nargs="+", default=[6],
                    help="Speed per action segment, e.g. --action_speed 4 6")
parser.add_argument("--baseline", type=float, default=0.2,
                    help="Stereo baseline distance")
parser.add_argument("--use_raymap", action="store_true",
                    help="Enable raymap conditioning (requires checkpoint trained with raymap)")
parser.add_argument("--ulysses_degree", type=int, default=1,
                    help="Ulysses sequence parallel degree (head parallel)")
parser.add_argument("--ring_degree", type=int, default=1,
                    help="Ring sequence parallel degree (sequence split)")
args, extras = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------- Multi-GPU setup ----------

if args.ulysses_degree > 1 or args.ring_degree > 1:
    from dist import set_multi_gpus_devices
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    print(f"Multi-GPU enabled: ulysses={args.ulysses_degree}, ring={args.ring_degree}, device={device}")
else:
    device = "cuda"


# ---------- Pipeline setup ----------

dtype = torch.bfloat16
pipeline_dir = args.pipeline_dir

transformer_additional_kwargs = {
    "cam_method": "prope",
    "add_control_adapter": True,
    "boundary": args.boundary,
}

if args.use_raymap:
    transformer_additional_kwargs["in_dim"] = 48 + 6

print("Loading transformer...")
transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(pipeline_dir, "transformer"),
    transformer_additional_kwargs=transformer_additional_kwargs,
    torch_dtype=dtype,
)

if args.ulysses_degree > 1 or args.ring_degree > 1:
    transformer.enable_multi_gpus_inference()

print("Loading VAE...")
vae = AutoencoderKLWan3_8.from_pretrained(
    os.path.join(pipeline_dir, "vae")).to(dtype)

print("Loading tokenizer and text encoder...")
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(pipeline_dir, "tokenizer"))
text_encoder = UMT5EncoderModel.from_pretrained(
    os.path.join(pipeline_dir, "text_encoder"),
    torch_dtype=dtype,
).eval()

print("Loading scheduler...")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    os.path.join(pipeline_dir, "scheduler"))

print("Assembling pipeline...")
pipeline = StereoWorldPipeline(
    transformer=transformer,
    transformer_2=None,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
).to(device)

print("Pipeline ready.\n")

# ---------- Inference ----------

negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def to_uint8_video(frames):
    """Convert decoded frames to uint8 numpy array [F, H, W, 3]."""
    if isinstance(frames, torch.Tensor):
        frames = frames.float().cpu().clamp(0, 1).mul(255).byte().numpy()
    elif frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    frames = np.squeeze(frames)
    if frames.ndim == 4 and frames.shape[0] == 3:
        frames = np.transpose(frames, (1, 2, 3, 0))
    elif frames.ndim == 4 and frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    return frames


def run_single(image_path, caption, action_seq, action_speed_list, scene_name, pipeline, args):
    """Run stereo inference on a single example."""
    target_h, target_w = args.H, args.W

    # Load and center-crop image to target aspect ratio
    img = np.array(PILImage.open(image_path).convert("RGB"))
    H, W = img.shape[0], img.shape[1]
    target_aspect = target_w / target_h
    src_aspect = W / H
    if src_aspect > target_aspect:
        new_w = int(H * target_aspect)
        x0 = (W - new_w) // 2
        img = img[:, x0:x0 + new_w]
    elif src_aspect < target_aspect:
        new_h = int(W / target_aspect)
        y0 = (H - new_h) // 2
        img = img[y0:y0 + new_h, :]
    img = np.array(PILImage.fromarray(img).resize((target_w, target_h), PILImage.LANCZOS))
    img = img.astype(np.float32).transpose(2, 0, 1)  # [3, H, W]

    # Expand action speed if needed
    if len(action_speed_list) == 1 and len(action_seq) > 1:
        action_speed_list = action_speed_list * len(action_seq)

    # Build stereo camera control
    stereo_cam = build_control_stereo_camera_from_action(
        action_seq, action_speed_list, args.num_frames, baseline=args.baseline)

    T_latent = stereo_cam["viewmats1"].shape[0]

    # Interleave left/right: [L0, R0, L1, R1, ...]
    viewmats = torch.stack([stereo_cam["viewmats1"], stereo_cam["viewmats2"]], dim=1).reshape(-1, 4, 4)
    K = torch.stack([stereo_cam["K1"], stereo_cam["K2"]], dim=1).reshape(-1, 3, 3)
    timestep = torch.stack([stereo_cam["timestep1"], stereo_cam["timestep2"]], dim=1).reshape(-1)

    control_camera_video = {"viewmats": viewmats, "K": K, "timestep": timestep}

    # Pipeline sees 2x latent frames
    pipeline_num_frames = 1 + (2 * T_latent - 1) * 4

    start_image = torch.from_numpy(img).float() / 255.0
    start_image = start_image.unsqueeze(0).unsqueeze(2).expand(-1, -1, pipeline_num_frames, -1, -1)

    print(f"  Resolution: {target_h}x{target_w}, stereo frames: {pipeline_num_frames} (2x{T_latent} latent)")

    # Build raymap if enabled (before dtype cast, needs float32)
    raymap = None
    if args.use_raymap:
        vae_downsample = vae.config.spatial_compression_ratio
        raymap = extrinsic_to_raymap(
            control_camera_video["viewmats"].float(),
            control_camera_video["K"].float(),
            H=target_h, W=target_w,
            vae_downsample=vae_downsample,
        )  # [T, 6, H_down, W_down]
        raymap = torch.as_tensor(raymap).permute(1, 0, 2, 3).to(device, dtype)  # [6, T, H_down, W_down]
        print(f"  Raymap: {raymap.shape}")

    # Move to device
    control_camera_video = {k: v.to(device, dtype) if v.is_floating_point() else v.to(device)
                            for k, v in control_camera_video.items()}
    start_image = start_image.to(device)

    print(f"  Action: {action_seq}, speed: {action_speed_list}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Caption: {caption[:80]}...")

    output = pipeline(
        prompt=caption,
        negative_prompt=negative_prompt,
        height=target_h,
        width=target_w,
        num_frames=pipeline_num_frames,
        start_image=start_image,
        control_camera_video=control_camera_video,
        raymap=raymap,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        boundary=args.boundary,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        output_type="latent",
        return_dict=True,
    )

    # Decode: split interleaved latents
    latents = output.videos
    left_latents = latents[:, :, ::2]
    right_latents = latents[:, :, 1::2]

    left_video = to_uint8_video(pipeline.decode_latents(left_latents))
    right_video = to_uint8_video(pipeline.decode_latents(right_latents))

    # Save stereo video
    out_stereo = os.path.join(args.output_dir, f"{scene_name}.mp4")

    stereo_frames = [PILImage.fromarray(np.concatenate([l, r], axis=1))
                     for l, r in zip(left_video, right_video)]
    export_to_video(stereo_frames, out_stereo, fps=args.fps)

    # Save metadata
    left_c2w = torch.linalg.inv(stereo_cam["viewmats1"].float()).numpy()
    meta = {
        "caption": caption,
        "scene_name": scene_name,
        "action_seq": action_seq,
        "action_speed_list": action_speed_list,
        "baseline": args.baseline,
        "num_frames": pipeline_num_frames,
        "T_latent": T_latent,
        "left_intrinsics": stereo_cam["K1"][0].numpy().tolist(),
        "left_c2w": left_c2w.tolist(),
    }
    out_json = os.path.join(args.output_dir, f"{scene_name}.json")
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {out_stereo}, {out_json}")


# Build job list
jobs = []
if args.eval_json:
    json_dir = os.path.dirname(os.path.abspath(args.eval_json))
    with open(args.eval_json) as f:
        entries = json.load(f)
    for entry in entries:
        image_path = entry["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(json_dir, image_path)
        if "scene_name" in entry:
            scene_name = entry["scene_name"]
        else:
            scene_name = os.path.splitext(os.path.basename(image_path))[0]
        jobs.append({
            "image_path": image_path,
            "caption": entry["caption"],
            "action_seq": entry["action_seq"],
            "action_speed_list": entry["action_speed_list"],
            "scene_name": scene_name,
        })
elif args.input_dir:
    img_path = os.path.join(args.input_dir, "left.png")
    caption_path = os.path.join(args.input_dir, "caption.txt")
    caption = open(caption_path).read().strip() if os.path.exists(caption_path) else ""
    scene_name = os.path.basename(os.path.normpath(args.input_dir))
    jobs.append({
        "image_path": img_path,
        "caption": caption,
        "action_seq": args.action_seq,
        "action_speed_list": args.action_speed,
        "scene_name": scene_name,
    })
else:
    raise ValueError("Must specify --input_dir or --eval_json")

print(f"Total jobs: {len(jobs)}\n")

for i, job in enumerate(jobs):
    print(f"=== [{i+1}/{len(jobs)}] {job['scene_name']} ===")
    try:
        run_single(job["image_path"], job["caption"], job["action_seq"],
                   job["action_speed_list"], job["scene_name"], pipeline, args)
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\nAll done.")
