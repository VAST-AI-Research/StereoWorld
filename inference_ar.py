"""Autoregressive long-video inference for a few-step StereoWorld student.

The default protocol uses four denoising steps, six interleaved latent frames
per chunk, 42 frames of KV history, and a bounded camera extension that
preserves the original 81-frame training prefix.

Usage:
    python inference_ar.py \
        --pipeline_dir weights/StereoWorldModel \
        --checkpoint weights/student_stage3.generator.pt \
        --eval_json ExpData/demo_custom_eval.json \
        --output_dir output_ar
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from einops import rearrange
from PIL import Image as PILImage
from transformers import AutoTokenizer, UMT5EncoderModel

from camera_utils import build_control_stereo_camera_from_action
from models.pipelines.pipeline_stereoworld import StereoWorldPipeline
from models.transformers.wan_transformer_3d import Wan2_2Transformer3DModel
from models.wan_vae import AutoencoderKLWan3_8


NOMINAL_TIMESTEPS = (1000, 750, 500, 250)
NOISE_LEVELS = (1.0, 0.75, 0.5, 0.25)
TIMESTEP_SHIFT = 5.0
CHUNK_FRAMES = 6
HISTORY_FRAMES = 42
BASE_NUM_FRAMES = 81
EXTENSION_ACTIONS = ("j", "l")
EXTENSION_SPEEDS = (1.0, 1.0)
MAX_EXTENSION_TRANSLATION = 0.0
MAX_EXTENSION_ROTATION = 15.0
DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("output_ar"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--height", "--H", dest="height", type=int, default=480)
    parser.add_argument("--width", "--W", dest="width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=153, help="Decoded frames per eye")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--baseline", type=float, default=0.2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("pipeline_dir", "checkpoint", "eval_json"):
        path = getattr(args, name).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--start must be non-negative and --limit must be positive")
    if args.seed == 0:
        raise ValueError("--seed must be an explicit non-zero integer")
    if args.height <= 0 or args.width <= 0 or args.fps <= 0:
        raise ValueError("--height, --width, and --fps must be positive")
    if not math.isfinite(args.baseline) or args.baseline < 0:
        raise ValueError("--baseline must be a finite non-negative number")
    if args.num_frames < BASE_NUM_FRAMES or (args.num_frames - BASE_NUM_FRAMES) % 4:
        raise ValueError("--num_frames must be >= 81 and preserve the 1+4k training prefix")
    latent_frames = 2 * (1 + (args.num_frames - 1) // 4)
    if latent_frames % CHUNK_FRAMES:
        raise ValueError(f"Interleaved latent frames ({latent_frames}) must divide into 6-frame chunks")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jobs(path: Path, start: int, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("--eval_json must contain a non-empty JSON list")
    rows = rows[start : None if limit is None else start + limit]
    if not rows:
        raise ValueError(f"No eval entries selected from index {start}")

    jobs = []
    task_ids = set()
    for index, row in enumerate(rows, start=start):
        required = ("image_path", "caption", "action_seq", "action_speed_list")
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(f"eval entry {index} is missing required fields")
        source_image_path = Path(str(row["image_path"])).expanduser()
        image_path = source_image_path
        image_path = image_path if image_path.is_absolute() else path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        actions = [str(value) for value in row["action_seq"]]
        speeds = [float(value) for value in row["action_speed_list"]]
        if len(speeds) == 1 and len(actions) > 1:
            speeds *= len(actions)
        if not actions or len(actions) != len(speeds):
            raise ValueError(f"eval entry {index} has mismatched actions and speeds")
        if any(not math.isfinite(speed) or speed < 0 for speed in speeds):
            raise ValueError(f"eval entry {index} has an invalid action speed")
        task_id = str(row.get("task_id") or row.get("scene_name") or image_path.stem)
        if not task_id or Path(task_id).name != task_id:
            raise ValueError(f"eval entry {index} has an invalid task_id")
        if task_id in task_ids:
            raise ValueError(f"eval entry {index} repeats task_id {task_id!r}")
        task_ids.add(task_id)
        jobs.append({
            "task_id": task_id,
            "image_path": image_path,
            "image_reference": (
                source_image_path.as_posix()
                if not source_image_path.is_absolute()
                else source_image_path.name
            ),
            "caption": str(row["caption"]),
            "action_seq": actions,
            "action_speed_list": speeds,
            "eval_index": index,
        })
    return jobs


def torch_load(path: Path) -> dict[str, Any]:
    for kwargs in (
        {"map_location": "cpu", "mmap": True, "weights_only": False},
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    ):
        try:
            payload = torch.load(path, **kwargs)
            if not isinstance(payload, dict):
                raise ValueError("Student checkpoint must be a dictionary")
            return payload
        except (TypeError, RuntimeError):
            pass
    raise RuntimeError(f"Unable to load checkpoint: {path}")


def load_student_weights(transformer: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch_load(checkpoint)
    state = payload.get("state_dict", payload)
    weights = {
        key.removeprefix("transformer."): value
        for key, value in state.items()
        if key.startswith("transformer.")
    }
    if not weights:
        model_keys = set(transformer.state_dict())
        weights = {key: value for key, value in state.items() if key in model_keys}
    if not weights:
        raise ValueError("Student checkpoint does not contain transformer weights")
    result = transformer.load_state_dict(weights, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)}"
        )
    del payload, state, weights
    gc.collect()


def load_pipeline(args: argparse.Namespace) -> StereoWorldPipeline:
    print("Loading transformer...")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        args.pipeline_dir / "transformer",
        transformer_additional_kwargs={"cam_method": "prope", "add_control_adapter": True},
        torch_dtype=DTYPE,
    )
    load_student_weights(transformer, args.checkpoint)
    print("Loading VAE, tokenizer, text encoder, and scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pipeline_dir / "scheduler", shift=TIMESTEP_SHIFT
    )
    loaded_shift = float(getattr(scheduler, "shift", scheduler.config.shift))
    if not math.isclose(loaded_shift, TIMESTEP_SHIFT):
        raise RuntimeError(
            f"FlowMatch scheduler shift must be {TIMESTEP_SHIFT}, got {loaded_shift}"
        )
    pipeline = StereoWorldPipeline(
        transformer=transformer.eval(),
        transformer_2=None,
        vae=AutoencoderKLWan3_8.from_pretrained(args.pipeline_dir / "vae").to(DTYPE).eval(),
        tokenizer=AutoTokenizer.from_pretrained(args.pipeline_dir / "tokenizer"),
        text_encoder=UMT5EncoderModel.from_pretrained(
            args.pipeline_dir / "text_encoder", torch_dtype=DTYPE
        ).eval(),
        scheduler=scheduler,
    ).to(torch.device("cuda"))

    spatial = int(pipeline.vae.config.spatial_compression_ratio)
    patch = tuple(int(value) for value in pipeline.transformer.config.patch_size)
    if spatial != 16:
        raise ValueError("The AR student requires the spatial-compression-16 StereoWorld VAE")
    if args.height % (spatial * patch[1]) or args.width % (spatial * patch[2]):
        raise ValueError("Resolution must align with the VAE and transformer patch sizes")
    return pipeline


def load_image(path: Path, height: int, width: int) -> torch.Tensor:
    image = np.array(PILImage.open(path).convert("RGB"))
    source_h, source_w = image.shape[:2]
    target_aspect = width / height
    if source_w / source_h > target_aspect:
        crop_w = int(source_h * target_aspect)
        x0 = (source_w - crop_w) // 2
        image = image[:, x0 : x0 + crop_w]
    elif source_w / source_h < target_aspect:
        crop_h = int(source_w / target_aspect)
        y0 = (source_h - crop_h) // 2
        image = image[y0 : y0 + crop_h]
    image = np.array(PILImage.fromarray(image).resize((width, height), PILImage.LANCZOS))
    return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)) / 255.0


def invert_se3(value: torch.Tensor) -> torch.Tensor:
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    output = torch.zeros_like(value)
    output[..., :3, :3] = rotation.transpose(-1, -2)
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", output[..., :3, :3], translation
    )
    output[..., 3, 3] = 1.0
    return output


def rotation_from_start(c2w: torch.Tensor) -> float:
    relative = c2w[0, :3, :3].T @ c2w[:, :3, :3]
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)).max())


def build_bounded_camera(
    actions: list[str], speeds: list[float], num_frames: int, baseline: float
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = build_control_stereo_camera_from_action(
        actions, speeds, BASE_NUM_FRAMES, baseline=baseline
    )
    base_latent_frames = int(base["viewmats1"].shape[0])
    extra_frames = num_frames - BASE_NUM_FRAMES
    extension_actions: list[str] = []
    extension_speeds: list[float] = []
    translation = rotation = 0.0
    camera = {key: value.clone() for key, value in base.items()}

    if extra_frames:
        segment_frames = math.ceil(BASE_NUM_FRAMES / len(actions))
        extension_count = math.ceil(extra_frames / segment_frames)
        extension_actions = [EXTENSION_ACTIONS[i % 2] for i in range(extension_count)]
        extension_speeds = [EXTENSION_SPEEDS[i % 2] for i in range(extension_count)]
        extension = build_control_stereo_camera_from_action(
            extension_actions, extension_speeds, extra_frames + 1, baseline=baseline
        )
        ext_left = invert_se3(extension["viewmats1"].float())
        ext_right = invert_se3(extension["viewmats2"].float())
        translation = float(
            torch.linalg.vector_norm(ext_left[:, :3, 3] - ext_left[:1, :3, 3], dim=-1).max()
        )
        rotation = rotation_from_start(ext_left)
        if translation > MAX_EXTENSION_TRANSLATION + 1e-5 or rotation > MAX_EXTENSION_ROTATION + 1e-4:
            raise RuntimeError("Camera extension exceeds the bounded motion budget")

        base_left_end = invert_se3(base["viewmats1"][-1:].float())
        base_right_end = invert_se3(base["viewmats2"][-1:].float())
        if not torch.allclose(base_left_end @ ext_right[:1], base_right_end, atol=1e-5, rtol=0):
            raise RuntimeError("Right camera is discontinuous at the extension splice")
        append_left = base_left_end @ ext_left[1:]
        append_right = base_left_end @ ext_right[1:]
        total = base_latent_frames + int(append_left.shape[0])
        camera = {
            "viewmats1": torch.cat([base["viewmats1"], invert_se3(append_left)]),
            "viewmats2": torch.cat([base["viewmats2"], invert_se3(append_right)]),
            "K1": torch.cat([base["K1"], extension["K1"][1:]]),
            "K2": torch.cat([base["K2"], extension["K2"][1:]]),
            "timestep1": torch.arange(total, dtype=base["timestep1"].dtype),
            "timestep2": torch.arange(total, dtype=base["timestep2"].dtype),
        }

    expected = 1 + (num_frames - 1) // 4
    if int(camera["viewmats1"].shape[0]) != expected:
        raise RuntimeError("Resolved camera length does not match --num_frames")
    for key in ("viewmats1", "viewmats2", "K1", "K2", "timestep1", "timestep2"):
        if not torch.equal(camera[key][:base_latent_frames], base[key]):
            raise RuntimeError(f"Camera extension changed the training prefix: {key}")

    left_c2w = invert_se3(camera["viewmats1"].float())
    right_c2w = invert_se3(camera["viewmats2"].float())
    rig = invert_se3(left_c2w) @ right_c2w
    expected_rig = torch.eye(4).expand_as(rig).clone()
    expected_rig[:, 0, 3] = baseline
    rig_error = float((rig - expected_rig).abs().max())
    if rig_error > 5e-5:
        raise RuntimeError(f"Stereo rig error is too large: {rig_error:.6g}")

    program = {
        "policy": "training_prefix_bounded",
        "replay_protocol": "build_base_and_extension_separately_then_splice_at_base_left_end",
        "base_num_frames": BASE_NUM_FRAMES,
        "requested_num_frames": num_frames,
        "source_action_seq": actions,
        "source_action_speed_list": speeds,
        "extension_action_seq": extension_actions,
        "extension_action_speed_list": extension_speeds,
        "extension_max_translation": translation,
        "extension_max_rotation_degrees": rotation,
        "rig_max_abs_error": rig_error,
    }
    return camera, program


def interleave_camera(
    stereo: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    values = {
        "viewmats": torch.stack([stereo["viewmats1"], stereo["viewmats2"]], dim=1).reshape(-1, 4, 4),
        "K": torch.stack([stereo["K1"], stereo["K2"]], dim=1).reshape(-1, 3, 3),
        "timestep": torch.stack([stereo["timestep1"], stereo["timestep2"]], dim=1).reshape(-1),
    }
    return {
        key: value.unsqueeze(0).to(device=device, dtype=DTYPE)
        if value.is_floating_point()
        else value.unsqueeze(0).to(device)
        for key, value in values.items()
    }


def prepare_i2v(
    pipeline: StereoWorldPipeline,
    image: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    channels = getattr(pipeline.vae.config, "z_dim", None) or pipeline.vae.config.latent_channels
    latents = pipeline.prepare_latents(
        1, channels, num_frames, height, width, DTYPE, device, generator
    )
    mask = torch.ones(
        (1, 1, latents.shape[2], latents.shape[3], latents.shape[4]),
        device=device,
        dtype=DTYPE,
    )
    processed = pipeline.image_processor.preprocess(
        rearrange(image, "b c f h w -> (b f) c h w"), height=height, width=width
    ).float()
    processed = rearrange(processed, "(b f) c h w -> b c f h w", f=1)
    _, first_latent = pipeline.prepare_control_latents(
        None, processed, 1, height, width, DTYPE, device, generator, False
    )
    if first_latent is None:
        raise RuntimeError("Failed to encode the I2V start image")
    start_latents = torch.zeros_like(latents)
    start_latents[:, :, :1] = first_latent.to(device=device, dtype=DTYPE)
    mask[:, :, :1] = 0
    return (1 - mask) * start_latents + mask * latents, start_latents, mask


def set_schedule(
    scheduler: FlowMatchEulerDiscreteScheduler, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model_timesteps = []
    for value in NOMINAL_TIMESTEPS:
        sigma = value / 1000
        denominator = 1 + (TIMESTEP_SHIFT - 1) * sigma
        model_timesteps.append(TIMESTEP_SHIFT * sigma / denominator * 1000)
    params = set(inspect.signature(scheduler.set_timesteps).parameters)
    if not {"timesteps", "sigmas", "mu"}.issubset(params):
        raise RuntimeError("FlowMatch scheduler must support paired timesteps and sigmas")
    scheduler.set_timesteps(
        timesteps=model_timesteps,
        sigmas=list(NOISE_LEVELS),
        device=device,
        mu=TIMESTEP_SHIFT,
    )
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = scheduler.sigmas[: len(timesteps)].to(device=device, dtype=torch.float32)
    if len(timesteps) != 4:
        raise RuntimeError("FlowMatch scheduler returned an invalid four-step schedule")
    return timesteps, sigmas


def seq_len(pipeline: StereoWorldPipeline, chunk: torch.Tensor) -> int:
    patch = tuple(int(value) for value in pipeline.transformer.config.patch_size)
    return int(chunk.shape[2] * (chunk.shape[3] // patch[1]) * (chunk.shape[4] // patch[2]))


def token_timestep(
    pipeline: StereoWorldPipeline, timestep: torch.Tensor, mask: torch.Tensor, length: int
) -> torch.Tensor:
    patch = tuple(int(value) for value in pipeline.transformer.config.patch_size)
    token_mask = mask[0, 0, :, :: patch[1], :: patch[2]].flatten()
    if token_mask.numel() != length:
        raise RuntimeError("I2V mask and transformer token count do not match")
    return (token_mask * timestep.to(dtype=token_mask.dtype)).unsqueeze(0)


def flowmatch_add_noise(
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = timestep.to(device=clean_latent.device, dtype=torch.float32) / 1000.0
    while sigma.ndim < clean_latent.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (clean_latent.ndim - sigma.ndim)))
    clean = clean_latent.to(dtype=torch.float32)
    eps = noise.to(device=clean_latent.device, dtype=torch.float32)
    return ((1.0 - sigma) * clean + sigma * eps).to(dtype=clean_latent.dtype)


def init_kv_cache(
    transformer: torch.nn.Module,
) -> list[dict[str, dict[str, torch.Tensor | None]]]:
    return [
        {"self": {"k": None, "v": None}, "prope": {"k": None, "v": None}}
        for _ in transformer.blocks
    ]


@torch.no_grad()
def predict_flow(
    pipeline: StereoWorldPipeline,
    chunk: torch.Tensor,
    prompt_embeds: list[torch.Tensor],
    camera: dict[str, torch.Tensor],
    mask: torch.Tensor,
    timestep: torch.Tensor,
    length: int,
    kv_cache: list[dict[str, dict[str, torch.Tensor | None]]],
    update_cache: bool,
) -> torch.Tensor:
    model_input = chunk
    if hasattr(pipeline.scheduler, "scale_model_input"):
        model_input = pipeline.scheduler.scale_model_input(model_input, timestep)
    return pipeline.transformer(
        x=model_input.to(DTYPE),
        context=prompt_embeds,
        t=token_timestep(pipeline, timestep, mask, length),
        seq_len=length,
        y=None,
        y_camera=camera,
        kv_cache=kv_cache,
        kv_cache_update=update_cache,
        kv_cache_history_window_frames=HISTORY_FRAMES,
    )


@torch.no_grad()
def generate_ar(
    pipeline: StereoWorldPipeline,
    latents: torch.Tensor,
    start_latents: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: list[torch.Tensor],
    camera: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = latents.device
    num_chunks = latents.shape[2] // CHUNK_FRAMES
    timesteps, flow_sigmas = set_schedule(pipeline.scheduler, device)
    timestep_values = [float(value.detach().cpu()) for value in timesteps]
    next_timesteps = [
        torch.full((int(latents.shape[0]),), value, device=device, dtype=torch.float32)
        for value in timestep_values[1:]
    ]
    length = seq_len(pipeline, latents[:, :, :CHUNK_FRAMES])
    kv_cache = init_kv_cache(pipeline.transformer)
    noise_shape = tuple(latents[:, :, :CHUNK_FRAMES].shape)
    noises = [
        torch.randn(noise_shape, generator=generator, dtype=torch.float32).to(device, DTYPE)
        for _chunk in range(num_chunks)
        for _step in range(3)
    ]

    for chunk_index in range(num_chunks):
        start = chunk_index * CHUNK_FRAMES
        end = start + CHUNK_FRAMES
        mask_chunk = mask[:, :, start:end].contiguous()
        start_chunk = start_latents[:, :, start:end].contiguous()
        camera_chunk = {key: value[:, start:end].contiguous() for key, value in camera.items()}

        for step, timestep in enumerate(timesteps):
            current = latents[:, :, start:end].contiguous()
            flow = predict_flow(
                pipeline, current, prompt_embeds, camera_chunk, mask_chunk,
                timestep, length, kv_cache, update_cache=False,
            )
            sigma = flow_sigmas[step].view(*([1] * current.ndim))
            x0 = (current.float() - sigma * flow.float()).to(DTYPE)
            if step < 3:
                output = flowmatch_add_noise(
                    x0,
                    noises[chunk_index * 3 + step],
                    next_timesteps[step],
                )
            else:
                output = x0
            latents[:, :, start:end] = (1 - mask_chunk) * start_chunk + mask_chunk * output

        if chunk_index < num_chunks - 1:
            predict_flow(
                pipeline,
                latents[:, :, start:end].contiguous(),
                prompt_embeds,
                camera_chunk,
                mask_chunk,
                torch.tensor(0.0, device=device),
                length,
                kv_cache,
                update_cache=True,
            )
        print(f"  chunk {chunk_index + 1:02d}/{num_chunks:02d} complete", flush=True)

    refresh_calls = max(num_chunks - 1, 0)
    profile = {
        "num_chunks": int(num_chunks),
        "chunk_latent_frames": CHUNK_FRAMES,
        "history_latent_frames": HISTORY_FRAMES,
        "denoise_calls": int(num_chunks * 4),
        "clean_refresh_calls": int(refresh_calls),
        "transformer_calls": int(num_chunks * 4 + refresh_calls),
        "model_timesteps": [float(value) for value in timesteps.cpu()],
        "flow_sigmas": [float(value) for value in flow_sigmas.cpu()],
        "update_rule": "flow_to_x0_then_renoise",
    }
    return latents, profile


def to_uint8_video(frames: np.ndarray | torch.Tensor) -> np.ndarray:
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


@torch.no_grad()
def write_video(
    pipeline: StereoWorldPipeline, latents: torch.Tensor, path: Path, fps: int
) -> int:
    left_latents, right_latents = latents[:, :, ::2], latents[:, :, 1::2]
    left = to_uint8_video(pipeline.decode_latents(left_latents))
    right = to_uint8_video(pipeline.decode_latents(right_latents))
    frames = [PILImage.fromarray(np.concatenate([l, r], axis=1)) for l, r in zip(left, right)]
    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)
    return len(frames)


@torch.no_grad()
def run_single(
    pipeline: StereoWorldPipeline,
    job: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_hash: str,
) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    stereo_camera, camera_program = build_bounded_camera(
        job["action_seq"], job["action_speed_list"], args.num_frames, args.baseline
    )
    camera = interleave_camera(stereo_camera, device)
    pipeline_num_frames = 1 + (camera["viewmats"].shape[1] - 1) * 4
    image = load_image(job["image_path"], args.height, args.width)
    image = image.unsqueeze(0).unsqueeze(2).to(device)
    latents, start_latents, mask = prepare_i2v(
        pipeline, image, pipeline_num_frames, args.height, args.width, generator, device
    )
    prompt_embeds, _ = pipeline.encode_prompt(
        job["caption"], do_classifier_free_guidance=False, device=device, dtype=DTYPE
    )
    latents, profile = generate_ar(
        pipeline, latents, start_latents, mask, prompt_embeds, camera, generator
    )

    video_path = args.output_dir / f"{job['task_id']}.mp4"
    json_path = args.output_dir / f"{job['task_id']}.json"
    decoded_frames = write_video(pipeline, latents, video_path, args.fps)
    if decoded_frames != args.num_frames:
        raise RuntimeError(f"Decoded {decoded_frames} frames, expected {args.num_frames}")

    metadata = {
        "schema_version": "stereoworld_ar_inference.v1",
        "task_id": job["task_id"],
        "eval_index": job["eval_index"],
        "image_path": job["image_reference"],
        "caption": job["caption"],
        "action_seq": job["action_seq"],
        "action_speed_list": job["action_speed_list"],
        "camera_program": camera_program,
        "baseline": args.baseline,
        "seed": args.seed,
        "pipeline": args.pipeline_dir.name,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "height": args.height,
        "width": args.width,
        "num_frames_per_eye": args.num_frames,
        "fps": args.fps,
        "schedule": {
            "nominal_timesteps": list(NOMINAL_TIMESTEPS),
            "noise_levels": list(NOISE_LEVELS),
            "timestep_shift": TIMESTEP_SHIFT,
            "guidance_scale": 1.0,
            "prediction_target": "flow",
        },
        "ar_profile": profile,
        "left_intrinsics": stereo_camera["K1"][0].tolist(),
        "left_c2w": invert_se3(stereo_camera["viewmats1"].float()).tolist(),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"  saved {video_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Autoregressive inference requires a CUDA device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.eval_json, args.start, args.limit)
    checkpoint_hash = sha256(args.checkpoint)
    pipeline = load_pipeline(args)
    print(f"Pipeline ready. Processing {len(jobs)} job(s).")

    failures = []
    for index, job in enumerate(jobs, start=1):
        print(f"\n=== [{index}/{len(jobs)}] {job['task_id']} ===", flush=True)
        try:
            run_single(pipeline, job, args, checkpoint_hash)
        except Exception as error:
            failures.append((job["task_id"], str(error)))
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        details = ", ".join(f"{name}: {error}" for name, error in failures)
        raise RuntimeError(f"{len(failures)} job(s) failed: {details}")
    print("\nAll jobs completed.")


if __name__ == "__main__":
    main()
