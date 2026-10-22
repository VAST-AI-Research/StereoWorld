import torch
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from torch import Tensor
from diffactory.utils.camera_utils import normalize_batch_intrinsics, extrinsic_to_raymap
from einops import rearrange
from PIL import Image
import numpy as np


def prepare_condition_info(batch, vae, latents_mean, latents_std, stage="test", use_raymap=False, num_views=2):
    # preset params
    batch['left_images'] = batch['left_image'].unsqueeze(1)
    batch['right_images'] = batch['right_image'].unsqueeze(1)  # B,1,C,H,W
    B, Nf, _, _ = batch['left_extrinsics'].shape
    device = batch['left_extrinsics'].device
    batch['frame_ids'] = torch.arange(Nf, device=device).unsqueeze(0).repeat(B, 1)  # B, Nf
    batch['num_views'] = num_views


    B, Nf, _, H, W = batch["left_images"].shape
    # ----- prepare prompts -----
    if "caption" not in batch:
        batch["prompts"] = [""] * B
    else:
        batch["prompts"] = batch["caption"]

    # >>> drop captions with probability
    # if self.cfg.drop_prob > 0 and stage == "train":
    #     drop_indices = torch.rand(B) < self.cfg.drop_prob
    #     for i in range(B):
    #         if drop_indices[i]:
    #             batch["prompts"][i] = ""

    num_views = batch["num_views"]
    if num_views == 1:
        processed_batch = prepare_condition_info_single_view(batch, vae, latents_mean, latents_std, stage=stage, num_views=num_views)
    elif num_views == 2:
        processed_batch = prepare_condition_info_two_view(batch, vae, latents_mean, latents_std, stage=stage, num_views=num_views, time_p=4)
    else:
        raise ValueError(f"num_views must be 1 or 2, but got {num_views}")
    
    processed_batch["prompts"] = batch["prompts"]
    ##### add raymap here
    if use_raymap:
        batch_raymap = []
        for i in range(B):
            raymap = extrinsic_to_raymap(processed_batch['extrinsics'][i].float(), processed_batch['intrinsics'][i].float(), H=H, W=W, vae_downsample=16)
            batch_raymap.append(raymap)
        #  B, Nf, 6, H, W -> B, 6, Nf, H, W
        batch_raymap = torch.stack(batch_raymap, dim=0).permute(0,2,1,3,4).to(processed_batch['extrinsics'].dtype)
    else:
        batch_raymap = None
    processed_batch["raymap"] = batch_raymap
    return processed_batch


def compress_temporal_cond(cond):
    p = 4
    padded_cond = cond
    return padded_cond[:,::p]  # 81 -> 21


@torch.no_grad()
def encode_video(
    video: Tensor, vae, latents_mean: Tensor, latents_std: Tensor
):
    video = video.permute(0, 2, 1, 3, 4).contiguous()  # BCFHW
    latents = (vae.encode(video).latent_dist.sample() - latents_mean) * latents_std
    return latents

@torch.no_grad()
def decode_video(
    latents: Tensor, vae, latents_mean: Tensor, latents_std: Tensor
):
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]  # BCFHW
    return video


def prepare_condition_info_single_view(batch, vae, latents_mean, latents_std, stage="train", num_views=1, time_p=4):
    B, Nf, _, H, W = batch["left_images"].shape

    # ----- process camera -----
    batch["left_images"] = batch["left_images"] / 255.
    normed_left_extrinsics = batch["left_extrinsics"]
    normed_left_intrinsics = normalize_batch_intrinsics(batch["left_intrinsics"], H, W)  # [B, Nf, 3, 3]
    
    normed_left_intrinsics = compress_temporal_cond(normed_left_intrinsics)
    normed_left_extrinsics = compress_temporal_cond(normed_left_extrinsics)

    frame_ids = compress_temporal_cond(batch['frame_ids']) // time_p

    # ----- process reference -----
    # reference_image = batch["ref_frames"] / 255.  # B,1,H,W
    # relative_ref_frame_ids = batch["ref_frame_ids"] // time_p  # B, 1
    # reference_extrinsics = batch["ref_extrinsics"]
    # reference_intrinsics = normalize_batch_intrinsics(batch["ref_intrinsics"], H, W)

    # ----- unified for single-view and two-view -----
    input_data = {}
    # condition_images = torch.cat([reference_image, batch["left_images"][:,:1]], dim=1)  # B,2,H,W
    condition_images = batch["left_images"][:,:1] # B,1,3,H,W
    condition_images = rearrange(condition_images, "B N C H W -> (B N) C H W")  # (B 1+1) C H W
    context_latent = encode_video(condition_images[:,None]*2-1, vae, latents_mean, latents_std)  # (B 1+1) 1 C H W -> (B 1+1) C 1 H W
    context_latent = rearrange(context_latent, "(B N) C 1 H W -> B C N H W", B=B)  # B, C, 1+1, H, W

    input_data['context_latent'] = context_latent
        
    # TODO check reference frame
    input_data['extrinsics'] = normed_left_extrinsics # B,1+Nf,4,4
    input_data['intrinsics'] = normed_left_intrinsics  # B,1+Nf,3,3
    input_data['frame_ids'] = frame_ids  # B,1+Nf

    input_data['num_views'] = num_views

    return input_data



def prepare_condition_info_two_view(batch, vae, latents_mean, latents_std, stage="train", num_views=2, time_p=4):
    B, Nf, _, H, W = batch["left_images"].shape

    # ----- process images -----
    batch["left_images"] = batch["left_images"] / 255.
    batch["right_images"] = batch["right_images"] / 255.

    normed_left_extrinsics = batch["left_extrinsics"]
    normed_right_extrinsics = batch["right_extrinsics"]

    normed_left_intrinsics = normalize_batch_intrinsics(batch["left_intrinsics"], H, W)  # [B, Nf, 3, 3]
    normed_right_intrinsics = normalize_batch_intrinsics(batch["right_intrinsics"], H, W)
    
    normed_left_intrinsics = compress_temporal_cond(normed_left_intrinsics)
    normed_right_intrinsics = compress_temporal_cond(normed_right_intrinsics)
    normed_left_extrinsics = compress_temporal_cond(normed_left_extrinsics)
    normed_right_extrinsics = compress_temporal_cond(normed_right_extrinsics)

    frame_ids = compress_temporal_cond(batch['frame_ids']) // time_p

    # ----- unified for single-view and two-view -----
    input_data = {}
    condition_images = torch.cat(
                [batch["left_images"][:,:1], batch["right_images"][:,:1]], dim=0
            )  # 2*B, 1, C, H, W
    condition_images = rearrange(condition_images, "B N C H W -> (B N) C H W")  # 2B C H W

    context_latent = encode_video(condition_images[:,None]*2-1, vae, latents_mean, latents_std)  # 2B 1 C H W -> 2B C 1 H W
    context_latent = rearrange(context_latent, "(nv B) C 1 H W -> B C nv H W", B=B, nv=num_views)  # 2*B, C, 1, H, W -> B, C, 2, H, W
    input_data['context_latent'] = context_latent

    input_data['extrinsics'] = torch.stack([normed_left_extrinsics, normed_right_extrinsics], dim=2).reshape(B,-1,4,4)  # B,Nf,2,4,4 -> B,Nf*2,4,4
    input_data['intrinsics'] = torch.stack([normed_left_intrinsics, normed_right_intrinsics], dim=2).reshape(B,-1,3,3)  # B,Nf,2,3,3 -> B,Nf*2,3,3
    input_data['frame_ids'] = torch.stack([frame_ids, frame_ids], dim=2).reshape(B,-1)  # B,Nf*2

    input_data['num_views'] = num_views
    return input_data


@torch.no_grad()
def postprocess(output, processed_batch, pipeline, latents_mean, latents_std):
    if processed_batch["num_views"] == 1:
        # TODO check reference frame
        # output_latents = output[:, :, 1:]
        output_latents = output
    else:
        left_xs_pred = output[:, :, ::2]
        right_xs_pred = output[:, :, 1::2]
        output_latents = torch.cat([left_xs_pred, right_xs_pred], dim=0)  # 2*B,C,Nf,H,W

    # output_latents = output_latents.to(self.my_dtype)
    output_latents = output_latents / latents_std + latents_mean
    video = pipeline.vae.decode(output_latents, return_dict=False)[0]  # BCFHW

    if processed_batch["num_views"] > 1:
        # stitch along w dimension
        left_video, right_video = torch.chunk(video, 2, dim=0)
        video = torch.cat([left_video, right_video], dim=-1)

    video = pipeline.video_processor.postprocess_video(video, output_type='np')

    outputs = WanPipelineOutput(frames=video)
    return outputs

def get_output_visualizations(batch, outputs, device, dtype, with_target=True, num_views=1):

    # num_frames = self.cfg.num_frames
    num_frames = batch['left_extrinsics'].shape[1] if 'left_extrinsics' in batch else batch['extrinsics'].shape[1]

    def _parse_output(output_video):
        pred_video = []
        for v in output_video:
            pred_video.append(torch.stack([torch.from_numpy(f) for f in v]))
        pred_video = torch.stack(pred_video).to(
            device, dtype
        )
        pred_video = rearrange(pred_video, "B F H W C -> F H (B W) C")[:num_frames]
        return pred_video
    
    pred_video = _parse_output(outputs.frames)

    if with_target:
        if num_views == 1:
            target_video = batch['left_images']  # one camera
        else:
            target_video = torch.cat([batch['left_images'], batch['right_images']], dim=-1)
        target_video = rearrange(
            target_video, "B F C H W -> F H (B W) C"
        )
        visual_video = torch.cat([target_video, pred_video], dim=1)
    else:
        visual_video = pred_video

    visual_pils = [
        Image.fromarray((255 * f.float().cpu().numpy()).astype(np.uint8))
        for f in visual_video
    ]
    return visual_pils