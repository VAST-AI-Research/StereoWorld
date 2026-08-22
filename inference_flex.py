"""Flexible stereo video generation with diverse right-camera trajectories.

Generates stereo (left + right) videos where the right camera can follow
various trajectory modes (independent, orbital, dynamic baseline, etc.)

Usage:
    # Single example:
    python inference_flex.py \
        --pipeline_dir /path/to/pipeline \
        --input_dir /path/to/data \
        --action_seq w wl --action_speed 4 6 \
        --right_mode orbital --right_angle_deg 20

    # Batch inference from eval.json:
    python inference_flex.py \
        --pipeline_dir /path/to/pipeline \
        --eval_json /path/to/eval.json

    # Multi-GPU (sequence parallel):
    torchrun --nproc_per_node=4 inference_flex.py \
        --pipeline_dir /path/to/pipeline \
        --eval_json /path/to/eval.json \
        --ulysses_degree 4 --ring_degree 1

    Right camera modes:
      independent      — fully independent right trajectory
      dynamic_baseline — breathing baseline (oscillates min↔max)
      orbital          — right orbits around left's gaze
      lead_follow      — right follows left's path with delay
      converging       — cameras converge then diverge
      height_offset    — fixed vertical separation + pitch tilt
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

from camera_utils_flex import build_flex_stereo_camera, build_gt_stereo_camera


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default=None,
                    help="Path to single data folder containing left.png, caption.txt")
parser.add_argument("--eval_json", type=str, default=None,
                    help="Path to eval.json for batch inference")
parser.add_argument("--pipeline_dir", type=str, required=True,
                    help="Path to pipeline directory")
parser.add_argument("--output_dir", type=str, default="output_flex")
parser.add_argument("--H", type=int, default=480)
parser.add_argument("--W", type=int, default=832)
parser.add_argument("--num_frames", type=int, default=81,
                    help="Number of output frames per view, must satisfy 1+4k")
parser.add_argument("--fps", type=int, default=16, help="FPS for output video")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--guidance_scale", type=float, default=3.0)
parser.add_argument(
    "--shift",
    type=float,
    default=3.0,
    help="Flow-matching scheduler shift used for sampling",
)
parser.add_argument("--boundary", type=float, default=0.875)
parser.add_argument("--seed", type=int, default=42)

# Left camera
parser.add_argument("--action_seq", type=str, nargs="+", default=["w"],
                    help="Action sequence for left eye")
parser.add_argument("--action_speed", type=float, nargs="+", default=[6],
                    help="Speed per left action segment")

# Right camera mode
parser.add_argument("--right_mode", type=str, default="horizontal_offset",
                    choices=["horizontal_offset", "depth_offset",
                             "height_offset", "converging", "gt_camera"],
                    help="Right camera trajectory mode")
parser.add_argument("--gt_camera_json", type=str, default=None,
                    help="Path to camera_sample.json for gt_camera mode")
parser.add_argument("--action_seq_right", type=str, nargs="+", default=None,
                    help="Action sequence for right eye (independent mode)")
parser.add_argument("--action_speed_right", type=float, nargs="+", default=None,
                    help="Speed per right action segment (independent mode)")

# Mode-specific parameters
parser.add_argument("--baseline", type=float, default=4.0)
parser.add_argument("--vergence_deg", type=float, default=20.0)
parser.add_argument("--depth_offset", type=float, default=4.0)
parser.add_argument("--height_offset", type=float, default=4.0)
parser.add_argument("--pitch_deg", type=float, default=-10.0)
parser.add_argument("--start_baseline", type=float, default=5.0)
parser.add_argument("--end_baseline", type=float, default=1.5)
parser.add_argument("--start_vergence_deg", type=float, default=10.0)
parser.add_argument("--end_vergence_deg", type=float, default=35.0)

# Multi-GPU
parser.add_argument("--ulysses_degree", type=int, default=1)
parser.add_argument("--ring_degree", type=int, default=1)
args, extras = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------- Multi-GPU setup ----------

if args.ulysses_degree > 1 or args.ring_degree > 1:
    from dist import set_multi_gpus_devices
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    print(f"Multi-GPU: ulysses={args.ulysses_degree}, ring={args.ring_degree}, device={device}")
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
print(f"Scheduler flow shift requested for sampling: {args.shift}")

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


def _get_mode_kwargs(job, args):
    """Extract mode-specific kwargs from job dict or args."""
    mode = job.get("right_mode", args.right_mode)
    kwargs = {}

    if mode == "horizontal_offset":
        kwargs["baseline"] = job.get("baseline", args.baseline)
        kwargs["vergence_deg"] = job.get("vergence_deg", args.vergence_deg)
    elif mode == "depth_offset":
        kwargs["depth_offset"] = job.get("depth_offset", args.depth_offset)
        kwargs["vergence_deg"] = job.get("vergence_deg", args.vergence_deg)
    elif mode == "height_offset":
        kwargs["height_offset"] = job.get("height_offset", args.height_offset)
        kwargs["pitch_deg"] = job.get("pitch_deg", args.pitch_deg)
    elif mode == "converging":
        kwargs["start_baseline"] = job.get("start_baseline", args.start_baseline)
        kwargs["end_baseline"] = job.get("end_baseline", args.end_baseline)
        kwargs["start_vergence_deg"] = job.get("start_vergence_deg", args.start_vergence_deg)
        kwargs["end_vergence_deg"] = job.get("end_vergence_deg", args.end_vergence_deg)
    elif mode == "gt_camera":
        kwargs["gt_cam_data"] = job.get("_gt_cam_data")

    return mode, kwargs


def run_single(image_path, caption, action_seq_left, speed_left,
               action_seq_right, speed_right, right_mode, mode_kwargs,
               scene_name, pipeline, args):
    """Run flex stereo inference on a single example."""
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

    # Expand speed lists if needed
    if len(speed_left) == 1 and len(action_seq_left) > 1:
        speed_left = speed_left * len(action_seq_left)
    if action_seq_right and len(speed_right) == 1 and len(action_seq_right) > 1:
        speed_right = speed_right * len(action_seq_right)

    # Build flex stereo camera
    if right_mode == "gt_camera":
        gt_cam_data = mode_kwargs.get("gt_cam_data")
        if gt_cam_data is None:
            raise ValueError("gt_camera mode requires gt_cam_data in mode_kwargs")
        import numpy as _np
        stereo_cam = build_gt_stereo_camera(
            left_c2w_600=_np.array(gt_cam_data["left_c2w"]),
            right_c2w_600=_np.array(gt_cam_data["right_c2w"]),
            K_left=gt_cam_data["left_intrinsics"],
            K_right=gt_cam_data["right_intrinsics"],
            image_size=gt_cam_data["image_size"],
            video_length=args.num_frames,
        )
    else:
        stereo_cam = build_flex_stereo_camera(
            action_seq_left, speed_left,
            action_seq_right, speed_right,
            args.num_frames, mode=right_mode, **mode_kwargs
        )

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
    print(f"  Mode: {right_mode}, kwargs: {mode_kwargs}")
    print(f"  Left action: {action_seq_left}, speed: {speed_left}")
    if action_seq_right:
        print(f"  Right action: {action_seq_right}, speed: {speed_right}")
    print(f"  Caption: {caption[:80]}...")

    # Move to device
    control_camera_video = {k: v.to(device, dtype) if v.is_floating_point() else v.to(device)
                            for k, v in control_camera_video.items()}
    start_image = start_image.to(device)

    output = pipeline(
        prompt=caption,
        negative_prompt=negative_prompt,
        height=target_h,
        width=target_w,
        num_frames=pipeline_num_frames,
        start_image=start_image,
        control_camera_video=control_camera_video,
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

    # Save stereo video (left-right concatenated)
    out_stereo = os.path.join(args.output_dir, f"{scene_name}.mp4")
    stereo_frames = [PILImage.fromarray(np.concatenate([l, r], axis=1))
                     for l, r in zip(left_video, right_video)]
    export_to_video(stereo_frames, out_stereo, fps=args.fps)

    # Save metadata
    left_c2w = torch.linalg.inv(stereo_cam["viewmats1"].float()).numpy()
    right_c2w = torch.linalg.inv(stereo_cam["viewmats2"].float()).numpy()
    # Don't serialize gt_cam_data (too large)
    meta_kwargs = {k: v for k, v in mode_kwargs.items() if k != "gt_cam_data"}
    meta = {
        "caption": caption,
        "scene_name": scene_name,
        "right_mode": right_mode,
        "mode_kwargs": meta_kwargs,
        "action_seq_left": action_seq_left,
        "action_speed_left": speed_left,
        "action_seq_right": action_seq_right,
        "action_speed_right": speed_right,
        "num_frames": pipeline_num_frames,
        "T_latent": T_latent,
        "left_intrinsics": stereo_cam["K1"][0].numpy().tolist(),
        "left_c2w": left_c2w.tolist(),
        "right_c2w": right_c2w.tolist(),
    }
    out_json = os.path.join(args.output_dir, f"{scene_name}.json")
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {out_stereo}, {out_json}")


# ---------- Build job list ----------

# Pre-load GT camera data if needed
_gt_camera_samples = None
def _load_gt_cameras():
    global _gt_camera_samples
    if _gt_camera_samples is not None:
        return _gt_camera_samples
    gt_path = args.gt_camera_json
    if gt_path is None:
        # Default path
        gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "Exp", "camera_sample.json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            _gt_camera_samples = json.load(f)
        print(f"Loaded {len(_gt_camera_samples)} GT camera samples from {gt_path}")
    else:
        raise FileNotFoundError(f"GT camera file not found: {gt_path}")
    return _gt_camera_samples

jobs = []
if args.eval_json:
    json_dir = os.path.dirname(os.path.abspath(args.eval_json))
    with open(args.eval_json) as f:
        entries = json.load(f)
    for entry in entries:
        image_path = entry["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(json_dir, image_path)
        scene_name = entry.get("scene_name",
                               os.path.splitext(os.path.basename(image_path))[0])
        job = {
            "image_path": image_path,
            "caption": entry["caption"],
            "action_seq_left": entry.get("action_seq_left", entry.get("action_seq", ["w"])),
            "action_speed_left": entry.get("action_speed_left", entry.get("action_speed_list", [6])),
            "action_seq_right": entry.get("action_seq_right", None),
            "action_speed_right": entry.get("action_speed_right", None),
            "right_mode": entry.get("right_mode", args.right_mode),
            "scene_name": scene_name,
            # Mode-specific overrides from JSON
            **{k: entry[k] for k in [
                "baseline", "vergence_deg", "depth_offset",
                "height_offset", "pitch_deg",
                "start_baseline", "end_baseline",
                "start_vergence_deg", "end_vergence_deg",
            ] if k in entry},
        }
        # Load GT camera data if needed
        if job["right_mode"] == "gt_camera":
            gt_idx = entry.get("gt_camera_index", 0)
            gt_samples = _load_gt_cameras()
            job["_gt_cam_data"] = gt_samples[gt_idx % len(gt_samples)]
        jobs.append(job)
elif args.input_dir:
    img_path = os.path.join(args.input_dir, "left.png")
    caption_path = os.path.join(args.input_dir, "caption.txt")
    caption = open(caption_path).read().strip() if os.path.exists(caption_path) else ""
    scene_name = os.path.basename(os.path.normpath(args.input_dir))
    jobs.append({
        "image_path": img_path,
        "caption": caption,
        "action_seq_left": args.action_seq,
        "action_speed_left": args.action_speed,
        "action_seq_right": args.action_seq_right,
        "action_speed_right": args.action_speed_right,
        "right_mode": args.right_mode,
        "scene_name": scene_name,
    })
else:
    raise ValueError("Must specify --input_dir or --eval_json")

print(f"Total jobs: {len(jobs)}\n")

for i, job in enumerate(jobs):
    print(f"=== [{i+1}/{len(jobs)}] {job['scene_name']} ({job.get('right_mode', args.right_mode)}) ===")
    try:
        mode, mode_kwargs = _get_mode_kwargs(job, args)
        action_seq_right = job.get("action_seq_right") or args.action_seq_right or job.get("action_seq_left", ["w"])
        speed_right = job.get("action_speed_right") or args.action_speed_right or job.get("action_speed_left", [6])

        run_single(
            job["image_path"], job["caption"],
            job["action_seq_left"], job["action_speed_left"],
            action_seq_right, speed_right,
            mode, mode_kwargs,
            job["scene_name"], pipeline, args,
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\nAll done.")
