# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2019-2023, Ross Wightman.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################
# Code is based on the PyTorch Image Models (timm) library
# (https://github.com/huggingface/pytorch-image-models)
###############################################################

import logging
import math
import os
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import _assert, nn
from torch.jit import Final
from torch.nn.init import _calculate_fan_in_and_fan_out

from ..data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from ..utils import to_2tuple
from .builder import build_model_with_cfg
from .helpers import named_apply
from .registry import generate_default_cfgs, register_model

_USE_FUSED_ATTN = bool(int(os.environ.get("USE_FUSED_ATTN", "1")))

__all__ = ["VisionTransformer"]  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


class _Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        H, W = x.shape[-2:]
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(self.img_size[0] == H, f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(self.img_size[1] == W, f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            else:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class _Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = _USE_FUSED_ATTN

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = _Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer.

    A PyTorch impl of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        class_token: bool = True,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        embed_layer: Callable = _PatchEmbed,
        norm_layer: Optional[Union[str, Callable, Type[torch.nn.Module]]] = None,
        act_layer: Optional[Union[str, Callable, Type[torch.nn.Module]]] = None,
        block_fn: Type[nn.Module] = _Block,
        mlp_layer: Type[nn.Module] = _Mlp,
    ) -> None:
        """
        Args:
        ----
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.

        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.has_class_token = class_token

        embed_args = {}
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        head_bias = -math.log(self.num_classes)
        nn.init.normal_(self.pos_embed, std=1 / math.sqrt(self.pos_embed.shape[-1]))
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(partial(_init_weights_vit_jax, head_bias=head_bias), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str) -> None:
        _load_weights(self, checkpoint_path)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))

        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat([*to_cat, x], dim=1)
        x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.

    Args:
    ----
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    """
    with torch.no_grad():
        torch.nn.init.trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


def _lecun_normal_(tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    variance = 1.0 / fan_in
    # constant is stddev of standard normal truncated to (-2, 2)
    _trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)


def _init_weights_vit_jax(module: nn.Module, name: str = "", head_bias: float = 0.0) -> None:
    """ViT weight initialization, matching JAX (Flax) impl."""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if "qkv" in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                nn.init.xavier_uniform_(module.weight, gain=math.sqrt((fan_in + fan_out) / (fan_in + (fan_out / 3))))
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if "mlp" in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        _lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str) -> None:
    sd = torch.load(checkpoint_path)["state_dict"]
    if sd["head.weight"].shape != model.state_dict()["head.weight"].shape:
        sd.pop("head.weight")
        sd.pop("head.bias", None)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    assert not unexpected_keys
    assert [k in ("head.weight", "head.bias") for k in missing_keys]
    _logger.info(f"Successfully loaded pretrained weights from {checkpoint_path}.")


def _cfg(**kwargs) -> Dict[str, Any]:
    return {
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "vit_small_patch32_224.augreg_in21k_ft_in1k": _cfg(custom_load=True),
        "vit_small_patch16_224.augreg_in21k_ft_in1k": _cfg(custom_load=True),
    },
)


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> VisionTransformer:
    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_strict=True,
        **kwargs,
    )


@register_model
def vit_small_patch32_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/32)."""
    model_args = {"patch_size": 32, "embed_dim": 384, "depth": 12, "num_heads": 6}
    return _create_vision_transformer("vit_small_patch32_224", pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/16)."""
    model_args = {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6}
    return _create_vision_transformer("vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs))
