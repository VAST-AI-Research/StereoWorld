"""Generate a right view while keeping an input left video fixed in latent space.

The input video is encoded with the StereoWorld VAE. Its latent frames occupy
the even positions of the model sequence and are restored after every denoising
step. Only the odd (right-view) latent frames are sampled.

Without source-camera extrinsics, the input video is conditioned as a static
identity left camera. The default right camera uses one fixed pose relative to
that assumed left camera.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from einops import rearrange
from PIL import Image as PILImage
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel

if TYPE_CHECKING:
    from models.pipelines.pipeline_stereoworld import StereoWorldPipeline


DTYPE = torch.bfloat16
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, deformed, disfigured, "
    "misshapen limbs, fused fingers, messy background"
)

CAMERA_PARAMETER_NAMES = (
    "baseline",
    "vergence_deg",
    "depth_offset",
    "height_offset",
    "pitch_deg",
    "start_baseline",
    "end_baseline",
    "start_vergence_deg",
    "end_vergence_deg",
)
CAMERA_DEFAULTS = {
    "baseline": 4.0,
    "vergence_deg": 15.0,
    "depth_offset": 4.0,
    "height_offset": 2.0,
    "pitch_deg": -10.0,
    "start_baseline": 5.0,
    "end_baseline": 1.5,
    "start_vergence_deg": 10.0,
    "end_vergence_deg": 35.0,
}
MODE_NAMES = (
    "horizontal_offset",
    "depth_offset",
    "height_offset",
    "converging",
)


def _validate_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise ValueError(f"{field} must be a {qualifier}string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain a NUL byte")
    return value


def _validate_scene_name(value: Any, field: str) -> str:
    value = _validate_string(value, field)
    if value != Path(value).name or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty file-name component")
    return value


def _validate_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline_dir", "--pipeline-dir", dest="pipeline_dir", type=Path, required=True)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--input_video", "--input-video", dest="input_video", type=Path)
    input_source.add_argument("--eval_json", "--eval-json", dest="eval_json", type=Path)
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir",
        type=Path, default=Path("output_view_inpaint")
    )
    parser.add_argument("--scene_name", "--scene-name", dest="scene_name", type=str, default=None)
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument(
        "--negative_prompt", "--negative-prompt", dest="negative_prompt",
        type=str, default=DEFAULT_NEGATIVE_PROMPT
    )
    parser.add_argument("--height", "--H", dest="height", type=int, default=None)
    parser.add_argument("--width", "--W", dest="width", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None, help="Defaults to the source video FPS")
    parser.add_argument(
        "--num_inference_steps", "--num-inference-steps", "--steps",
        dest="num_inference_steps", type=int, default=50
    )
    parser.add_argument(
        "--guidance_scale", "--guidance-scale", dest="guidance_scale",
        type=float, default=3.0
    )
    parser.add_argument(
        "--flow_shift", "--flow-shift", "--shift", dest="flow_shift",
        type=float, default=3.0
    )
    parser.add_argument("--boundary", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--right_mode", "--right-mode", dest="right_mode",
        choices=MODE_NAMES,
        default="horizontal_offset",
    )
    parser.add_argument("--baseline", type=float, default=CAMERA_DEFAULTS["baseline"])
    parser.add_argument(
        "--vergence_deg", "--vergence-deg", dest="vergence_deg",
        type=float, default=CAMERA_DEFAULTS["vergence_deg"]
    )
    parser.add_argument(
        "--depth_offset", "--depth-offset", dest="depth_offset",
        type=float, default=CAMERA_DEFAULTS["depth_offset"]
    )
    parser.add_argument(
        "--height_offset", "--height-offset", dest="height_offset",
        type=float, default=CAMERA_DEFAULTS["height_offset"]
    )
    parser.add_argument(
        "--pitch_deg", "--pitch-deg", dest="pitch_deg",
        type=float, default=CAMERA_DEFAULTS["pitch_deg"]
    )
    parser.add_argument(
        "--start_baseline", "--start-baseline", dest="start_baseline",
        type=float, default=CAMERA_DEFAULTS["start_baseline"]
    )
    parser.add_argument(
        "--end_baseline", "--end-baseline", dest="end_baseline",
        type=float, default=CAMERA_DEFAULTS["end_baseline"]
    )
    parser.add_argument(
        "--start_vergence_deg", "--start-vergence-deg", dest="start_vergence_deg",
        type=float, default=CAMERA_DEFAULTS["start_vergence_deg"]
    )
    parser.add_argument(
        "--end_vergence_deg", "--end-vergence-deg", dest="end_vergence_deg",
        type=float, default=CAMERA_DEFAULTS["end_vergence_deg"]
    )
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--save_left_reconstruction", action="store_true")
    return parser.parse_args(argv)


def validate_job_args(args: argparse.Namespace) -> None:
    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input_video}")
    args.scene_name = _validate_scene_name(args.scene_name or args.input_video.stem, "scene_name")
    if args.right_mode not in MODE_NAMES:
        raise ValueError(f"right_mode must be one of {', '.join(MODE_NAMES)}")
    for name in CAMERA_PARAMETER_NAMES:
        setattr(args, name, _validate_finite_number(getattr(args, name), f"--{name}"))


def validate_args(args: argparse.Namespace) -> None:
    args.pipeline_dir = args.pipeline_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.input_video is not None:
        args.input_video = args.input_video.expanduser().resolve()
    if args.eval_json is not None:
        args.eval_json = args.eval_json.expanduser().resolve()

    if (args.input_video is None) == (args.eval_json is None):
        raise ValueError("Specify exactly one of --input_video or --eval_json")
    if args.input_video is not None:
        validate_job_args(args)
    else:
        if not args.eval_json.is_file():
            raise FileNotFoundError(f"Eval JSON does not exist: {args.eval_json}")
        if args.scene_name is not None:
            raise ValueError("--scene_name is only valid with --input_video")
    required_components = ("transformer", "vae", "tokenizer", "text_encoder", "scheduler")
    missing = [name for name in required_components if not (args.pipeline_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Pipeline directory is missing required components: {', '.join(missing)}"
        )
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.height is not None and (args.height <= 0 or args.width <= 0):
        raise ValueError("--height and --width must be positive")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.num_inference_steps <= 0:
        raise ValueError("--num_inference_steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 0:
        raise ValueError("--guidance_scale must be a finite positive number")
    if not math.isfinite(args.flow_shift) or args.flow_shift <= 0:
        raise ValueError("--flow_shift must be a finite positive number")
    if not math.isfinite(args.boundary):
        raise ValueError("--boundary must be finite")


def parse_eval_jobs(eval_json: Path, output_root: Path) -> list[argparse.Namespace]:
    """Parse flat view-inpainting eval entries into validated per-video jobs.

    Relative ``input_video`` paths are resolved from the eval JSON's directory.
    The output directory intentionally groups all right-camera modes of the
    same video under ``<output_root>/<input_video_stem>/``.
    """
    eval_json = Path(eval_json).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    try:
        with eval_json.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Eval JSON is invalid: {eval_json}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("Eval JSON must contain a non-empty top-level list")

    jobs: list[argparse.Namespace] = []
    seen_scene_names: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"Eval entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{prefix} must be an object")

        input_value = _validate_string(entry.get("input_video"), f"{prefix}.input_video")
        input_video = Path(input_value).expanduser()
        if not input_video.is_absolute():
            input_video = eval_json.parent / input_video
        input_video = input_video.resolve()
        if not input_video.is_file():
            raise FileNotFoundError(f"{prefix}.input_video does not exist: {input_video}")

        scene_name = _validate_scene_name(
            entry.get("scene_name", input_video.stem), f"{prefix}.scene_name"
        )
        if scene_name in seen_scene_names:
            raise ValueError(f"{prefix}.scene_name duplicates an earlier entry: {scene_name}")
        seen_scene_names.add(scene_name)

        right_mode = _validate_string(
            entry.get("right_mode", "horizontal_offset"), f"{prefix}.right_mode"
        )
        if right_mode not in MODE_NAMES:
            raise ValueError(
                f"{prefix}.right_mode must be one of {', '.join(MODE_NAMES)}, "
                f"got {right_mode!r}"
            )

        job_data: dict[str, Any] = {
            "task_id": entry.get("task_id"),
            "input_video": input_video,
            "caption": _validate_string(entry.get("caption", ""), f"{prefix}.caption", allow_empty=True),
            "negative_prompt": entry.get("negative_prompt"),
            "scene_name": scene_name,
            "right_mode": right_mode,
            "output_dir": output_root / input_video.stem,
        }
        if job_data["negative_prompt"] is not None:
            job_data["negative_prompt"] = _validate_string(
                job_data["negative_prompt"], f"{prefix}.negative_prompt", allow_empty=True
            )
        required_camera_fields = {
            "horizontal_offset": ("baseline", "vergence_deg"),
            "depth_offset": ("depth_offset", "vergence_deg"),
            "height_offset": ("height_offset", "pitch_deg"),
            "converging": (
                "start_baseline",
                "end_baseline",
                "start_vergence_deg",
                "end_vergence_deg",
            ),
        }[right_mode]
        missing_camera_fields = [name for name in required_camera_fields if name not in entry]
        if missing_camera_fields:
            raise ValueError(
                f"{prefix} ({right_mode}) is missing camera fields: "
                f"{', '.join(missing_camera_fields)}"
            )
        for name in CAMERA_PARAMETER_NAMES:
            if name in entry:
                job_data[name] = _validate_finite_number(entry[name], f"{prefix}.{name}")
        jobs.append(argparse.Namespace(**job_data))
    return jobs


def apply_eval_job(args: argparse.Namespace, job: argparse.Namespace) -> argparse.Namespace:
    """Apply one eval entry over CLI defaults without mutating the base args."""
    job_args = argparse.Namespace(**vars(args))
    for name, value in vars(job).items():
        if value is not None:
            setattr(job_args, name, value)
    job_args.output_dir = Path(job_args.output_dir).expanduser().resolve()
    validate_job_args(job_args)
    return job_args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path) -> tuple[np.ndarray, float]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"Input video has no decodable frames: {path}")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"Input video contains inconsistent frame sizes: {sorted(shapes)}")
    return np.stack(frames), source_fps


def transformer_weight_input_channels(transformer_dir: Path) -> int | None:
    """Read Conv3d input channels without materializing the transformer weights."""
    weight_name = "patch_embedding.weight"
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard = index.get("weight_map", {}).get(weight_name)
        if shard is None:
            raise ValueError(f"Transformer index does not contain {weight_name}")
        weight_path = transformer_dir / shard
    elif not weight_path.is_file():
        return None

    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        if weight_name not in handle.keys():
            raise ValueError(f"Transformer weights do not contain {weight_name}")
        shape = handle.get_slice(weight_name).get_shape()
    if len(shape) != 5:
        raise ValueError(f"Expected a Conv3d patch embedding, got shape {shape}")
    return int(shape[1])


def center_crop_resize(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = frames.shape[1:3]
    if (source_h, source_w) == (height, width):
        return frames
    target_aspect = width / height
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = int(source_h * target_aspect)
        x0 = (source_w - crop_w) // 2
        frames = frames[:, :, x0 : x0 + crop_w]
    elif source_aspect < target_aspect:
        crop_h = int(source_w / target_aspect)
        y0 = (source_h - crop_h) // 2
        frames = frames[:, y0 : y0 + crop_h]
    resized = [
        np.asarray(PILImage.fromarray(frame).resize((width, height), PILImage.LANCZOS))
        for frame in frames
    ]
    return np.stack(resized)


def validate_video_shape(
    frames: np.ndarray,
    temporal_compression_ratio: int,
    height_alignment: int,
    width_alignment: int,
) -> None:
    num_frames, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB video frames, got {channels} channels")
    if num_frames < 1 or (num_frames - 1) % temporal_compression_ratio:
        raise ValueError(
            "Input video frame count must be 1 + k * temporal_compression_ratio "
            f"(ratio={temporal_compression_ratio}), got {num_frames}"
        )
    if height % height_alignment or width % width_alignment:
        raise ValueError(
            "Video resolution must align with the VAE and transformer patches; "
            f"height multiple={height_alignment}, width multiple={width_alignment}, "
            f"got {height}x{width}"
        )


def load_pipeline(args: argparse.Namespace, device: torch.device) -> StereoWorldPipeline:
    from models.pipelines.pipeline_stereoworld import StereoWorldPipeline
    from models.transformers.wan_transformer_3d import Wan2_2Transformer3DModel
    from models.wan_vae import AutoencoderKLWan3_8

    print("Loading transformer...", flush=True)
    transformer_dir = args.pipeline_dir / "transformer"
    transformer_kwargs: dict[str, Any] = {
        "cam_method": "prope",
        "add_control_adapter": True,
        "boundary": args.boundary,
    }
    weight_input_channels = transformer_weight_input_channels(transformer_dir)
    if weight_input_channels is not None:
        with (transformer_dir / "config.json").open("r", encoding="utf-8") as handle:
            configured_input_channels = int(json.load(handle)["in_dim"])
        if configured_input_channels != weight_input_channels:
            print(
                "Correcting stale transformer input-channel config: "
                f"config={configured_input_channels}, weights={weight_input_channels}",
                flush=True,
            )
        transformer_kwargs.update(
            in_dim=weight_input_channels,
            in_channels=weight_input_channels,
        )
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        transformer_dir,
        transformer_additional_kwargs=transformer_kwargs,
        torch_dtype=DTYPE,
    ).eval()
    print("Loading VAE, tokenizer, text encoder, and scheduler...", flush=True)
    pretrained_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pipeline_dir / "scheduler"
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pretrained_scheduler.config, shift=args.flow_shift
    )
    pipeline = StereoWorldPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=AutoencoderKLWan3_8.from_pretrained(args.pipeline_dir / "vae").to(DTYPE).eval(),
        tokenizer=AutoTokenizer.from_pretrained(args.pipeline_dir / "tokenizer"),
        text_encoder=UMT5EncoderModel.from_pretrained(
            args.pipeline_dir / "text_encoder", torch_dtype=DTYPE
        ).eval(),
        scheduler=scheduler,
    ).to(device)
    return pipeline


def mode_kwargs(args: argparse.Namespace) -> dict[str, float]:
    if args.right_mode == "horizontal_offset":
        return {"baseline": args.baseline, "vergence_deg": args.vergence_deg}
    if args.right_mode == "depth_offset":
        return {"depth_offset": args.depth_offset, "vergence_deg": args.vergence_deg}
    if args.right_mode == "height_offset":
        return {"height_offset": args.height_offset, "pitch_deg": args.pitch_deg}
    if args.right_mode == "converging":
        return {
            "start_baseline": args.start_baseline,
            "end_baseline": args.end_baseline,
            "start_vergence_deg": args.start_vergence_deg,
            "end_vergence_deg": args.end_vergence_deg,
        }
    raise ValueError(f"Unsupported right camera mode: {args.right_mode}")


def build_camera(
    num_frames: int, args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    from camera_utils_flex import build_flex_stereo_camera

    kwargs = mode_kwargs(args)
    stereo = build_flex_stereo_camera(
        ["w"], [0.0], ["w"], [0.0], num_frames, mode=args.right_mode, **kwargs
    )
    interleaved = {
        "viewmats": torch.stack([stereo["viewmats1"], stereo["viewmats2"]], dim=1)
        .reshape(-1, 4, 4),
        "K": torch.stack([stereo["K1"], stereo["K2"]], dim=1).reshape(-1, 3, 3),
        "timestep": torch.stack([stereo["timestep1"], stereo["timestep2"]], dim=1)
        .reshape(-1),
    }
    # Preserve float32 camera matrices until ray construction. The pipeline
    # casts PRoPE inputs to the transformer dtype immediately before use.
    interleaved = {key: value.to(device=device) for key, value in interleaved.items()}
    return stereo, interleaved, kwargs


@torch.no_grad()
def encode_left_video(
    pipeline: "StereoWorldPipeline",
    frames: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    height, width = frames.shape[1:3]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    processed = pipeline.image_processor.preprocess(
        rearrange(video, "b f c h w -> (b f) c h w"), height=height, width=width
    ).float()
    processed = rearrange(processed, "(b f) c h w -> b c f h w", b=1)
    _, left_latents = pipeline.prepare_control_latents(
        None, processed, 1, height, width, DTYPE, device, None, False
    )
    if left_latents is None:
        raise RuntimeError("StereoWorld VAE did not return left-video latents")
    return left_latents


def make_interleaved_condition(
    left_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left_latents.ndim != 5 or left_latents.shape[0] != 1:
        raise ValueError("Left latents must have shape [1, C, T, H, W]")
    clean_latents = left_latents.new_zeros(
        left_latents.shape[0],
        left_latents.shape[1],
        left_latents.shape[2] * 2,
        left_latents.shape[3],
        left_latents.shape[4],
    )
    clean_latents[:, :, 0::2] = left_latents
    denoise_mask = torch.ones(
        left_latents.shape[0],
        1,
        left_latents.shape[2] * 2,
        1,
        1,
        dtype=torch.bool,
        device=left_latents.device,
    )
    denoise_mask[:, :, 0::2] = False
    return clean_latents, denoise_mask


def make_interleaved_raymap(
    camera: dict[str, torch.Tensor],
    height: int,
    width: int,
    vae_downsample: int,
    expected_latent_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build `[6, T, H_latent, W_latent]` rays from pair-relative cameras."""
    from camera_utils import extrinsic_to_raymap

    raymap = extrinsic_to_raymap(
        camera["viewmats"].to(device=device, dtype=torch.float32),
        camera["K"].to(device=device, dtype=torch.float32),
        H=height,
        W=width,
        vae_downsample=vae_downsample,
    ).permute(1, 0, 2, 3)
    if tuple(raymap.shape[1:]) != expected_latent_shape:
        raise RuntimeError(
            "Raymap and latent sequence shapes differ: "
            f"{tuple(raymap.shape[1:])} vs {expected_latent_shape}"
        )
    return raymap.to(device=device, dtype=dtype)


def to_uint8_video(frames: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        frames = frames.float().cpu().clamp(0, 1).mul(255).byte().numpy()
    elif frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError(f"Expected one decoded video, got batch size {frames.shape[0]}")
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor after removing batch, got {frames.shape}")
    if frames.shape[0] == 3:
        frames = np.transpose(frames, (1, 2, 3, 0))
    elif frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    elif frames.shape[-1] != 3:
        raise ValueError(f"Could not identify the RGB channel dimension in {frames.shape}")
    return frames


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    export_to_video([PILImage.fromarray(frame) for frame in frames], str(path), fps=fps)


@torch.no_grad()
def run_single(
    args: argparse.Namespace,
    pipeline: "StereoWorldPipeline",
    device: torch.device,
) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_frames, source_fps = read_video(args.input_video)
    source_height, source_width = source_frames.shape[1:3]
    target_height = args.height or source_height
    target_width = args.width or source_width
    frames = center_crop_resize(source_frames, target_height, target_width)
    if args.fps is None and source_fps <= 0:
        raise ValueError("Input video does not provide a valid frame rate; pass --fps explicitly")
    output_fps = args.fps or int(round(source_fps))

    spatial = int(pipeline.vae.config.spatial_compression_ratio)
    temporal = int(pipeline.vae.config.temporal_compression_ratio)
    patch = tuple(int(value) for value in pipeline.transformer.config.patch_size)
    if temporal != 4:
        raise NotImplementedError(
            "camera_utils_flex currently requires VAE temporal compression ratio 4, "
            f"got {temporal}"
        )
    validate_video_shape(frames, temporal, spatial * patch[1], spatial * patch[2])

    print(
        f"Encoding fixed left video: {len(frames)} frames, {target_height}x{target_width}",
        flush=True,
    )
    left_latents = encode_left_video(pipeline, frames, device)
    clean_latents, denoise_mask = make_interleaved_condition(left_latents)
    stereo_camera, camera, camera_kwargs = build_camera(len(frames), args, device)
    if camera["viewmats"].shape[0] != clean_latents.shape[2]:
        raise RuntimeError(
            "Camera and latent sequence lengths differ: "
            f"{camera['viewmats'].shape[0]} vs {clean_latents.shape[2]}"
        )
    pipeline_num_frames = 1 + (clean_latents.shape[2] - 1) * temporal
    latent_channels = clean_latents.shape[1]
    transformer_in_channels = pipeline.transformer.patch_embedding.in_channels
    raymap_channels = transformer_in_channels - latent_channels
    if raymap_channels == 6:
        raymap = make_interleaved_raymap(
            camera,
            target_height,
            target_width,
            spatial,
            tuple(clean_latents.shape[2:]),
            device,
            DTYPE,
        )
    elif raymap_channels == 0:
        raymap = None
    else:
        raise RuntimeError(
            "Unsupported transformer input layout: "
            f"{transformer_in_channels} input channels for {latent_channels} latent channels"
        )
    print(
        f"Sampling {clean_latents.shape[2] // 2} right latent frames "
        f"with mode={args.right_mode}, kwargs={camera_kwargs}, "
        f"raymap_channels={raymap_channels}",
        flush=True,
    )
    output = pipeline(
        prompt=args.caption,
        negative_prompt=args.negative_prompt,
        height=target_height,
        width=target_width,
        num_frames=pipeline_num_frames,
        control_camera_video=camera,
        raymap=raymap,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.flow_shift,
        boundary=args.boundary,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        clean_latents=clean_latents,
        denoise_mask=denoise_mask,
        output_type="latent",
        return_dict=True,
    )
    output_latents = output.videos
    expected_left = left_latents.to(device=output_latents.device, dtype=output_latents.dtype)
    actual_left = output_latents[:, :, 0::2]
    max_left_error = float((actual_left.float() - expected_left.float()).abs().max().cpu())
    if not torch.equal(actual_left, expected_left):
        raise RuntimeError(
            f"Fixed left latents changed during denoising (max abs error {max_left_error})"
        )

    right_latents = output_latents[:, :, 1::2]
    right_frames = to_uint8_video(pipeline.decode_latents(right_latents))
    if right_frames.shape != frames.shape:
        raise RuntimeError(
            f"Decoded right video shape {right_frames.shape} does not match left {frames.shape}"
        )

    right_path = args.output_dir / f"{args.scene_name}_right.mp4"
    stereo_path = args.output_dir / f"{args.scene_name}_stereo.mp4"
    metadata_path = args.output_dir / f"{args.scene_name}.json"
    write_video(right_frames, right_path, output_fps)
    stereo_frames = np.concatenate([frames, right_frames], axis=2)
    write_video(stereo_frames, stereo_path, output_fps)

    left_reconstruction_path = None
    if args.save_left_reconstruction:
        left_reconstruction_path = args.output_dir / f"{args.scene_name}_left_reconstruction.mp4"
        reconstructed_left = to_uint8_video(pipeline.decode_latents(actual_left))
        write_video(reconstructed_left, left_reconstruction_path, output_fps)

    latent_path = None
    if args.save_latents:
        latent_path = args.output_dir / f"{args.scene_name}_latents.safetensors"
        save_file(
            {
                "left_clean": expected_left.detach().cpu().contiguous(),
                "right_generated": right_latents.detach().cpu().contiguous(),
            },
            str(latent_path),
        )

    left_c2w = torch.linalg.inv(stereo_camera["viewmats1"].float()).tolist()
    right_c2w = torch.linalg.inv(stereo_camera["viewmats2"].float()).tolist()
    metadata: dict[str, Any] = {
        "schema_version": "stereoworld_view_inpainting.v1",
        "input_video": str(args.input_video),
        "input_sha256": sha256(args.input_video),
        "pipeline_dir": str(args.pipeline_dir),
        "caption": args.caption,
        "scene_name": args.scene_name,
        "source": {
            "num_frames": int(source_frames.shape[0]),
            "height": int(source_height),
            "width": int(source_width),
            "fps": source_fps,
        },
        "output": {
            "num_frames": int(frames.shape[0]),
            "height_per_eye": int(target_height),
            "width_per_eye": int(target_width),
            "fps": output_fps,
            "right_video": str(right_path),
            "stereo_video": str(stereo_path),
            "left_reconstruction": str(left_reconstruction_path) if left_reconstruction_path else None,
            "latents": str(latent_path) if latent_path else None,
        },
        "latent_inpainting": {
            "left_shape": list(expected_left.shape),
            "interleaved_shape": list(output_latents.shape),
            "mask_pattern": "fixed_left_even_generated_right_odd",
            "left_latents_exactly_locked": True,
            "left_latent_max_abs_error": max_left_error,
            "dtype": str(output_latents.dtype),
        },
        "sampling": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "boundary": args.boundary,
            "seed": args.seed,
        },
        "camera": {
            "left_motion_assumption": "static_identity_without_source_extrinsics",
            "right_mode": args.right_mode,
            "relative_pose_is_fixed": args.right_mode != "converging",
            "raymap_channels": raymap_channels,
            "mode_kwargs": camera_kwargs,
            "left_intrinsics": stereo_camera["K1"][0].tolist(),
            "right_intrinsics": stereo_camera["K2"][0].tolist(),
            "left_c2w": left_c2w,
            "right_c2w": right_c2w,
        },
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved right view: {right_path}")
    print(f"Saved stereo video: {stereo_path}")
    print(f"Left latent lock verified exactly (max abs error: {max_left_error})")


def run(args: argparse.Namespace) -> None:
    """Run a single video, retaining the legacy programmatic entry point."""
    if args.input_video is None:
        raise ValueError("run() only supports a single --input_video job; use main() for --eval_json")
    device = torch.device("cuda")
    pipeline = load_pipeline(args, device)
    run_single(args, pipeline, device)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("View-inpainting inference requires a CUDA device")
    try:
        jobs = parse_eval_jobs(args.eval_json, args.output_dir) if args.eval_json is not None else None
        device = torch.device("cuda")
        pipeline = load_pipeline(args, device)
        if jobs is not None:
            print(f"Total jobs: {len(jobs)}", flush=True)
            for index, job in enumerate(jobs, start=1):
                job_args = apply_eval_job(args, job)
                print(
                    f"=== [{index}/{len(jobs)}] {job_args.scene_name} "
                    f"({job_args.right_mode}) ===",
                    flush=True,
                )
                run_single(job_args, pipeline, device)
                gc.collect()
                torch.cuda.empty_cache()
        else:
            run_single(args, pipeline, device)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
