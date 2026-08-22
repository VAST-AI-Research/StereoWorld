# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from torch import nn, view_as_complex

from .attention_utils import attention
from .prope_utils import prope_qkv



def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs, temporal_indices=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))

        # temporal freqs: use indices if provided, otherwise take first f
        if temporal_indices is not None:
            t_indices = temporal_indices[i] if temporal_indices.dim() > 1 else temporal_indices
            t_indices = t_indices[:f].long()
            freqs_t = freqs[0][t_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_t = freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)

        freqs_i = torch.cat([
            freqs_t,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_qk(q, k, grid_sizes, freqs, temporal_indices=None):
    q = rope_apply(q, grid_sizes, freqs, temporal_indices)
    k = rope_apply(k, grid_sizes, freqs, temporal_indices)
    return q, k


def _apply_kv_cache(
    cache,
    k,
    v,
    *,
    seq_dim,
    update=False,
    history_window_frames=None,
    frame_seqlen=None,
):
    """Read or append clean-history KV; append each clean chunk exactly once."""
    if cache is None:
        return k, v

    cached_k = cache.get("k")
    cached_v = cache.get("v")
    if (cached_k is None) != (cached_v is None):
        raise ValueError("KV cache must contain both k and v")

    if update:
        max_tokens = None
        if history_window_frames is not None:
            if frame_seqlen is None or int(frame_seqlen) <= 0:
                raise ValueError("frame_seqlen is required for a bounded KV cache")
            max_tokens = int(history_window_frames) * int(frame_seqlen)
            if max_tokens <= 0:
                raise ValueError("history_window_frames must be positive")

        new_tokens = int(k.shape[seq_dim])
        if max_tokens is not None and new_tokens >= max_tokens:
            if new_tokens > max_tokens:
                slices = [slice(None)] * k.ndim
                slices[seq_dim] = slice(new_tokens - max_tokens, None)
                k = k[tuple(slices)].contiguous()
                v = v[tuple(slices)].contiguous()
            cached_k = cached_v = None
        elif cached_k is not None and max_tokens is not None:
            keep_tokens = max_tokens - new_tokens
            cached_tokens = int(cached_k.shape[seq_dim])
            if cached_tokens > keep_tokens:
                slices = [slice(None)] * cached_k.ndim
                slices[seq_dim] = slice(cached_tokens - keep_tokens, None)
                cached_k = cached_k[tuple(slices)]
                cached_v = cached_v[tuple(slices)]

        merged_k = k if cached_k is None else torch.cat([cached_k, k], dim=seq_dim)
        merged_v = v if cached_v is None else torch.cat([cached_v, v], dim=seq_dim)
        cache["k"] = merged_k.detach()
        cache["v"] = merged_v.detach()
        return merged_k, merged_v

    if cached_k is None:
        return k, v
    return torch.cat([cached_k, k], dim=seq_dim), torch.cat([cached_v, v], dim=seq_dim)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight.to(dtype=x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        dtype=torch.bfloat16,
        t=0,
        temporal_indices=None,
        kv_cache=None,
        cache_update=False,
        cache_history_window_frames=None,
        frame_seqlen=None,
        **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            temporal_indices(Tensor, optional): Shape [B, F] or [F], frame indices for RoPE
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs, temporal_indices)
        k, v = _apply_kv_cache(
            kv_cache,
            k.to(dtype),
            v.to(dtype),
            seq_dim=1,
            update=cache_update,
            history_window_frames=cache_history_window_frames,
            frame_seqlen=frame_seqlen,
        )

        x = attention(
            q.to(dtype),
            k,
            v=v,
            k_lens=None if kv_cache is not None else seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class PropeSelfAttention(nn.Module):
    """
    Lightweight parallel PRoPE self-attention module.
    
    """

    def __init__(self,
                 dim,
                 attn_dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        # independent q/k/v projections (dim -> attn_dim)
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        # zero-initialize out_proj for stable residual training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x,
        cam_emb,
        seq_lens=None,
        sp_group=None,
        kv_cache=None,
        cache_update=False,
        cache_history_window_frames=None,
        frame_seqlen=None,
    ):
        """
        Args:
            x (Tensor): Shape [B, S, D] — input tokens (after norm + modulation).
                        S may be padded to max_seq_len; actual lengths are in seq_lens.
            cam_emb (dict): Camera embedding dict with keys 'viewmats' and 'K'
            seq_lens (Tensor, optional): Shape [B], actual token count per sample
        Returns:
            Tensor: Shape [B, S, D] — prope attention output (zero-padded to match input)
        """
        batch_size, seq_len, dim = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim


        # q/k/v projections: [B, L, N, D]
        q = self.norm_q(self.q_proj(x)).view(batch_size, seq_len, num_heads, head_dim)
        k = self.norm_k(self.k_proj(x)).view(batch_size, seq_len, num_heads, head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, num_heads, head_dim)

        # transpose to [B, N, L, D] for prope_qkv
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply PRoPE positional encoding
        q, k, v, apply_fn_o = prope_qkv(
            q, k, v,
            viewmats=cam_emb['viewmats'],
            Ks=cam_emb['K'],
        )
        k, v = _apply_kv_cache(
            kv_cache,
            k,
            v,
            seq_dim=2,
            update=cache_update,
            history_window_frames=cache_history_window_frames,
            frame_seqlen=frame_seqlen,
        )

        # attention: expects [B, L, N, D]
        out = attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=None if kv_cache is not None else seq_lens,
        )

        # apply inverse PRoPE transform
        out = apply_fn_o(out.transpose(1, 2)).transpose(1, 2)

        # project back to dim: [B, L_valid, D]
        out = out.flatten(2)
        out = self.out_proj(out)


        return out

class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # compute attention
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.add_control_adapter = kwargs.get('add_control_adapter', False)
        self.cam_method = kwargs.get('cam_method', None)
        self.attn_compress = kwargs.get('attn_compress', 1)
        self.layer_idx = kwargs.get('layer_idx', None)
        cam_self_attn_layers = kwargs.get('cam_self_attn_layers', None)

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # parallel PRoPE self-attention branch 
        add_cam_attn = self.add_control_adapter and self.cam_method == 'prope'
        if add_cam_attn and cam_self_attn_layers is not None:
            add_cam_attn = self.layer_idx in cam_self_attn_layers
        if add_cam_attn:
            self.cam_self_attn = PropeSelfAttention(
                dim,
                dim // self.attn_compress,
                num_heads // self.attn_compress,
                qk_norm=qk_norm,
                eps=eps,
            )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
        group=None,
        cam_emb=None,
        temporal_indices=None,
        kv_cache=None,
        cache_update=False,
        cache_history_window_frames=None,
        frame_seqlen=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            temporal_indices(Tensor, optional): Shape [B, F] or [F], frame indices for RoPE
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention (RoPE branch)
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(
            temp_x,
            seq_lens,
            grid_sizes,
            freqs,
            dtype,
            temporal_indices=temporal_indices,
            kv_cache=None if kv_cache is None else kv_cache["self"],
            cache_update=cache_update,
            cache_history_window_frames=cache_history_window_frames,
            frame_seqlen=frame_seqlen,
        )

        # parallel PRoPE branch (v2 design)
        if hasattr(self, 'cam_self_attn') and cam_emb is not None:
            # print(f'temp_x: {temp_x.shape}')
            # print(f'cam_emb["viewmats"]: {cam_emb["viewmats"].shape}')
            # print(f'cam_emb["K"]: {cam_emb["K"].shape}')
            y = y + self.cam_self_attn(
                temp_x,
                cam_emb,
                seq_lens=seq_lens,
                kv_cache=None if kv_cache is None else kv_cache["prope"],
                cache_update=cache_update,
                cache_history_window_frames=cache_history_window_frames,
                frame_seqlen=frame_seqlen,
            )

        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        
        x = self.norm(x) * (1 + e[1]) + e[0]
        x = self.head(x.to(self.head.weight.dtype))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            WanLayerNorm(in_dim, elementwise_affine=True), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            WanLayerNorm(out_dim, elementwise_affine=True))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        cross_attn_type=None,
        add_control_adapter=False,
        cam_method='prope',
        attn_compress=1,
        cam_self_attn_layers=None,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            if model_type == "t2v":
                cross_attn_type = 't2v_cross_attn'
            else:
                cross_attn_type = 'i2v_cross_attn'
            # else:
                # cross_attn_type = "cross_attn"

        attn_class = WanAttentionBlock # by default use WanAttentionBlock
        
        self.control_adapter = None
        self.camera_projection = None

        self.blocks = nn.ModuleList([
            attn_class(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              add_control_adapter=add_control_adapter,
                              cam_method=cam_method,
                              attn_compress=attn_compress,
                              layer_idx=layer_idx,
                              cam_self_attn_layers=cam_self_attn_layers)
            for layer_idx in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.cam_method = cam_method

        
        self.gradient_checkpointing = False

        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.sp_group = None

        self.customize_initialize_weights()
    
    def customize_initialize_weights(self):
        self.init_weights()

        for blk in self.blocks:
            if hasattr(blk, 'cam_self_attn'):
                nn.init.zeros_(blk.cam_self_attn.out_proj.weight)
                nn.init.zeros_(blk.cam_self_attn.out_proj.bias)


    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")


    def init_expand_raymap(self, raymap_dim=6):
        """Expand patch_embedding to accept additional raymap input channels.

        New raymap channels are zero-initialized so pretrained weights are preserved.
        """
        _weight = self.patch_embedding.weight.clone()
        _bias = self.patch_embedding.bias.clone()
        _kernel_size = self.patch_embedding.kernel_size
        _stride = self.patch_embedding.stride
        in_channels = self.patch_embedding.in_channels
        out_channels = self.patch_embedding.out_channels

        self.patch_embedding = nn.Conv3d(
            raymap_dim + in_channels, out_channels,
            kernel_size=_kernel_size, stride=_stride)

        _weight_padded = torch.zeros_like(self.patch_embedding.weight)
        _weight_padded[:, :-raymap_dim] = _weight
        self.patch_embedding.weight = nn.Parameter(_weight_padded)
        self.patch_embedding.bias = nn.Parameter(_bias)
        self.config.in_channels = raymap_dim + in_channels
        print(f"=== Expand with {raymap_dim} Raymap. In Channels is : {self.config.in_channels} Now ===")

    def enable_multi_gpus_inference(self,):
        from dist import (get_sequence_parallel_rank,
                          get_sequence_parallel_world_size, get_sp_group,
                          usp_attn_forward, sp_prope_forward)
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        self.sp_group = None # None for inference

        # For RoPE self-attention.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

            # For PRoPE parallel branch (cam_self_attn).
            if hasattr(block, 'cam_self_attn'):
                block.cam_self_attn.forward = types.MethodType(
                    sp_prope_forward, block.cam_self_attn)

    # @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        dtype=torch.bfloat16,
        kv_cache=None,
        kv_cache_update=False,
        kv_cache_history_window_frames=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        device = self.patch_embedding.weight.device
        dtype = x[0].dtype
        if kv_cache is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("KV cache is inference-only; use torch.no_grad()")
            if self.sp_world_size > 1:
                raise NotImplementedError("KV-cache inference currently supports one GPU")
            if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) != len(self.blocks):
                raise ValueError("kv_cache must contain one entry per transformer block")
            for block_cache in kv_cache:
                if not isinstance(block_cache, dict) or not {"self", "prope"}.issubset(block_cache):
                    raise ValueError("Each block cache must contain self and prope entries")
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        frame_seqlen = None
        if kv_cache is not None:
            spatial_grids = grid_sizes[:, 1:]
            if not torch.equal(spatial_grids, spatial_grids[:1].expand_as(spatial_grids)):
                raise ValueError("KV-cache inference requires a common spatial token grid")
            frame_seqlen = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())

        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        if kv_cache is not None and not torch.all(seq_lens == int(seq_len)):
            raise ValueError("KV-cache inference does not support padded token sequences")
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # print(f"seq_len: {seq_len} t.shape: {t.shape}")
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:

            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

                
        # Unpack cam_emb dict into individual tensors for gradient checkpointing compatibility.
        # torch.utils.checkpoint does not correctly track tensors nested inside dicts,
        # which causes NCCL all-to-all desync during recomputation.
        if y_camera is not None and isinstance(y_camera, dict):
            cam_viewmats = y_camera['viewmats']
            cam_K = y_camera['K']
            cam_timestep = y_camera.get('timestep', None)
        else:
            cam_viewmats = None
            cam_K = None
            cam_timestep = None

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, sp_group):
                    def custom_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype, t, _cam_viewmats, _cam_K, _cam_timestep):
                        cam_emb = None
                        if _cam_viewmats is not None:
                            cam_emb = {'viewmats': _cam_viewmats, 'K': _cam_K}
                        return module(x, e=e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs,
                                      context=context, context_lens=context_lens, dtype=dtype, t=t,
                                      group=sp_group, cam_emb=cam_emb, temporal_indices=_cam_timestep)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, self.sp_group),
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    dtype,
                    t,
                    cam_viewmats,
                    cam_K,
                    cam_timestep,
                    **ckpt_kwargs,
                )
            else:
                cam_emb_dict = None
                if cam_viewmats is not None:
                    cam_emb_dict = {'viewmats': cam_viewmats, 'K': cam_K}
                kwargs = dict(
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.freqs,
                    context=context,
                    context_lens=context_lens,
                    dtype=dtype,
                    t=t,
                    group=self.sp_group,
                    cam_emb=cam_emb_dict,
                    temporal_indices=cam_timestep,
                    kv_cache=None if kv_cache is None else kv_cache[block_index],
                    cache_update=bool(kv_cache_update),
                    cache_history_window_frames=kv_cache_history_window_frames,
                    frame_seqlen=frame_seqlen,
                )
                x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        return x


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                else:
                    from diffusers.models.modeling_utils import \
                        load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
                    model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
                    model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']

                filtered_state_dict = {}
                for key in state_dict:
                    if key in model.state_dict() and model.state_dict()[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")
                        
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                def initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
                    initialized_dict = {}
                    
                    with torch.no_grad():
                        for key in missing_keys:
                            param_shape = model_state_dict[key].shape
                            param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                            if 'weight' in key:
                                if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                                elif 'embedding' in key or 'embed' in key:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                                elif 'head' in key or 'output' in key or 'proj_out' in key or 'project_out' in key or 'out_proj' in key:
                                    logging.info(f"Initialize {key} with zeros")
                                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                elif len(param_shape) >= 2:
                                    initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                                    nn.init.xavier_uniform_(initialized_dict[key])
                                else:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                            elif 'bias' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_mean' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_var' in key:
                                initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                            elif 'num_batches_tracked' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)

                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys, 
                        model.state_dict(), 
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)
                    # logging.info(f"Weights of Camera_adapter: {initialized_params}")

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            print("patch embeddings are not the same")
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model


class Wan2_2Transformer3DModel(WanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        cam_method='prope',
        attn_compress=1,
        cam_self_attn_layers=None,
    ):
        r"""
        Initialize the diffusion model backbone.
        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            cross_attn_type="cross_attn",
            cam_method=cam_method,
            attn_compress=attn_compress,
            cam_self_attn_layers=cam_self_attn_layers,
        )
        
        if hasattr(self, "img_emb"):
            del self.img_emb
