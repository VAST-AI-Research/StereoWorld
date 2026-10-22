import math
from typing import Callable, List, Optional, Union, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange
from torch import nn

from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections, WanTimeTextImageEmbedding, _get_added_kv_projections

# from ..transformers.wan_prope_transformer import WanTransformer3DPropeModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_npu_available():
    import torch_npu

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_wan_transformer_attn_processor(
    transformer,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_self_attn_module_names: Optional[list[str]] = None,
) -> None:
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        dim_head = transformer.config.attention_head_dim
        # num_heads = transformer.config.num_attention_heads  # this is wrong!
        num_heads = transformer.blocks[0].attn1.heads
        if name.endswith("attn1.processor"):
            attn_procs[name] = (
                set_self_attn_proc_func(name, dim_head, num_heads, attn_processor)
                if do_set_processor(name, set_self_attn_module_names)
                else attn_processor
            )
        elif name.endswith("attn2.processor"):
            attn_procs[name] = attn_processor

    transformer.set_attn_processor(attn_procs)


class WanAttnPropeCausalProcessor(torch.nn.Module):
    _attention_backend = None

    def __init__(
            self,
            dim_head: int,
            num_heads: int,
            use_prope: bool = True,
            use_causal: bool = True,
            name: Optional[str] = None,
            eps: float = 1e-5,
            ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        
        super().__init__()
        self.use_prope = use_prope
        self.use_causal = use_causal
        self.name = name

        dim = dim_head * num_heads
        inner_dim = dim // 4  # TODO set as arguments. Affect _prepare_apply_fns function.

        if self.use_prope:
            self.to_q_prope = nn.Linear(
                in_features=dim, out_features=inner_dim, bias=True
            )
            self.to_k_prope = nn.Linear(
                in_features=dim, out_features=inner_dim, bias=True
            )

            self.norm_q_prope = torch.nn.RMSNorm(inner_dim, eps=eps, elementwise_affine=True)
            self.norm_k_prope = torch.nn.RMSNorm(inner_dim, eps=eps, elementwise_affine=True)

    def _get_prope_qk_projections(self, hidden_states: torch.Tensor,):
        query = self.to_q_prope(hidden_states)
        key = self.to_k_prope(hidden_states)
        return query, key


    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        left_extrinsics: Optional[torch.Tensor] = None,
        right_extrinsics: Optional[torch.Tensor] = None,
        left_intrinsics: Optional[torch.Tensor] = None,
        right_intrinsics: Optional[torch.Tensor] = None,
        post_video_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            attn: The attention layer.
            hidden_states: The hidden states of the query.  [B, Seq, C]
            encoder_hidden_states: The hidden states of the key and value.  [B, S', C']
            attention_mask: The attention mask.
            rotary_emb: The rotary embedding.  [1, S, C''] x 2
            viewmats: The view matrices.
            Ks: The K matrices.
        """

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # --------------- 2D RoPE Part ---------------
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))  # (B, Seq, num_heads, head_dim)
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # --------------- PRoPE ---------------
        if rotary_emb is not None:
            if self.use_prope:
                query_prope, key_prope = self._get_prope_qk_projections(hidden_states)
                query_prope = self.norm_q_prope(query_prope)
                key_prope = self.norm_k_prope(key_prope)

                query_prope = query_prope.unflatten(2, (attn.heads, -1))  # (B, Seq, num_heads, head_dim)
                key_prope = key_prope.unflatten(2, (attn.heads, -1))  # (B, Seq, num_heads, head_dim)

                query = torch.cat((query, query_prope), dim=-1)
                key = torch.cat((key, key_prope), dim=-1)

            query = query.permute(0, 2, 1, 3)  # (B, num_heads, Seq, head_dim)
            key = key.permute(0, 2, 1, 3)  # (B, num_heads, Seq, head_dim)
            value = value.permute(0, 2, 1, 3)  # (B, num_heads, Seq, head_dim)

            (batch, _, _, head_dim) = query.shape
            cameras = left_extrinsics.shape[1]
            assert left_extrinsics.shape == (batch, cameras, 4, 4)
            assert left_intrinsics is None or left_intrinsics.shape == (batch, cameras, 3, 3)

            freqs_cos, freqs_sin = rotary_emb
            # (1, seqlen, 1, head_dim) -> (1, 1, seqlen, head_dim)
            cos = freqs_cos[..., 0::2].permute(0, 2, 1, 3)
            sin = freqs_sin[..., 1::2].permute(0, 2, 1, 3)

            apply_fn_q, apply_fn_kv, apply_fn_v_custom, apply_fn_o_custom = _prepare_apply_fns(
                head_dim=head_dim,
                viewmats=left_extrinsics,
                Ks=left_intrinsics,
                rope_coeffs=(cos, sin),
                use_prope=self.use_prope,
            )

            # # (B, num_heads, Seq, head_dim) -> (B, Seq, num_heads, head_dim)
            query = apply_fn_q(query).permute(0, 2, 1, 3)
            key = apply_fn_kv(key).permute(0, 2, 1, 3)

            value = apply_fn_v_custom(value)
            value = value.permute(0, 2, 1, 3)

            # -------------------------

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            assert NotImplementedError("I2V task is not supported yet.")

        hidden_states = dispatch_attention_fn(
            query.type_as(value),
            key.type_as(value),
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )  # (B, Seq, num_heads, head_dim)

        if rotary_emb is not None:
            # apply output
            hidden_states = apply_fn_o_custom(hidden_states.permute(0, 2, 1, 3))  # (B, num_heads, Seq, head_dim)
            hidden_states = hidden_states.permute(0, 2, 1, 3)  # (B, Seq, num_heads, head_dim)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(value)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# --------------------- utils ---------------------

def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    rope_coeffs: Tuple[torch.Tensor, torch.Tensor],
    use_prope: bool = True,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        assert Ks[0,0,0,2] < 1.0, "Ks[0,0,0,2] must be < 1.0"
        Ks_norm = Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    inner_dim = head_dim // 5 * 4 if use_prope else head_dim  # head_dim contains x y t c
    if use_prope:
        # TODO: add config interface
        transforms_q = [
            (partial(_rope_apply_coeffs, coeffs=rope_coeffs, inverse=True), inner_dim),
            (partial(_apply_tiled_projmat, matrix=P_T), head_dim-inner_dim),
        ]
        transforms_kv = [
            (partial(_rope_apply_coeffs, coeffs=rope_coeffs, inverse=True), inner_dim),
            (partial(_apply_tiled_projmat, matrix=P_inv), head_dim-inner_dim),
        ]
    else:
        transforms_q = [
            (partial(_rope_apply_coeffs, coeffs=rope_coeffs, inverse=True), head_dim),
        ]
        transforms_kv = [
            (partial(_rope_apply_coeffs, coeffs=rope_coeffs, inverse=True), head_dim),
        ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)

    # custom apply_fn_v
    apply_fn_v_custom = partial(_apply_block_diagonal, func_size_pairs=[
        (lambda x: x, inner_dim),
        # (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
    ])
    
    apply_fn_o_custom = partial(_apply_block_diagonal, func_size_pairs=[
        (lambda x: x, inner_dim),
        # (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
    ])

    return apply_fn_q, apply_fn_kv, apply_fn_v_custom, apply_fn_o_custom


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)

    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0, f"seqlen {seqlen} should be greater than cameras {cameras} and divisible by cameras"
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0


    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.

    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    # x_in = feats[..., : feats.shape[-1] // 2]
    # y_in = feats[..., feats.shape[-1] // 2 :]
    x_in, y_in = feats.unflatten(-1, (-1, 2)).unbind(-1)

    return torch.stack(
        [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
        if not inverse
        else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        , dim=-1
    ).flatten(-2)



def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out.to(Ks.dtype)


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out