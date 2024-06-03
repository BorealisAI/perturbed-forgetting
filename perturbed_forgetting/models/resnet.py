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

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .builder import build_model_with_cfg
from .registry import generate_default_cfgs, register_model

__all__ = ["ResNet"]  # model_registry will add each entrypoint fn to this


class _AvgPool2d(nn.Module):

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 1,
        pool_type: str = "avg",
        flatten: bool = False,
    ):
        super().__init__()
        self.pool_type = pool_type or ""
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        else:
            assert pool_type == "avg"
            self.pool_type = pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        """
        Args:
        ----
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.

        """
        super().__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def _get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _downsample_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    first_dilation: Optional[int] = None,
    norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = _get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=p,
                dilation=first_dilation,
                bias=False,
            ),
            norm_layer(out_channels),
        ],
    )


def _downsample_avg(
    in_channels: int,
    out_channels: int,
    _kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    _first_dilation: Optional[int] = None,
    norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        if avg_stride == 1 and dilation > 1:
            raise NotImplementedError
        pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[pool, nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), norm_layer(out_channels)],
    )


def _make_blocks(
    block_fn: _Bottleneck,
    channels: List[int],
    block_repeats: List[int],
    inplanes: int,
    reduce_first: int = 1,
    output_stride: int = 32,
    down_kernel_size: int = 1,
    avg_down: bool = False,
    **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stage_name = f"layer{stage_idx + 1}"
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = {
                "in_channels": inplanes,
                "out_channels": planes * block_fn.expansion,
                "kernel_size": down_kernel_size,
                "stride": stride,
                "dilation": dilation,
                "first_dilation": prev_dilation,
                "norm_layer": kwargs.get("norm_layer"),
            }
            downsample = _downsample_avg(**down_kwargs) if avg_down else _downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    **block_kwargs,
                ),
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append({"num_chs": inplanes, "reduction": net_stride, "module": stage_name})

    return stages, feature_info


def _create_classifier(
    num_features: int,
    num_classes: int,
    pool_type: str = "avg",
    drop_rate: Optional[float] = None,
):
    assert pool_type == "avg"
    global_pool = _AvgPool2d(pool_type=pool_type, flatten=True)
    fc = nn.Linear(num_features, num_classes, bias=True)
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ResNet(nn.Module):

    def __init__(
        self,
        block: _Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        global_pool: str = "avg",
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        block_reduce_first: int = 1,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        act_layer: Union[str, Callable, Type[torch.nn.Module]] = nn.ReLU,
        norm_layer: Union[str, Callable, Type[torch.nn.Module]] = nn.BatchNorm2d,
        drop_rate: float = 0.0,
        zero_init_last: bool = True,
        block_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
        ----
            block (nn.Module): class for the residual block
            layers (List[int]): number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in _Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module

        """
        super().__init__()
        block_args = block_args or {}
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False),
                ],
            )
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [{"num_chs": inplanes, "reduction": 2, "module": "act1"}]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = _make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = _create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for _n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, "zero_init_last"):
                    m.zero_init_last()

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return "fc" if name_only else self.fc

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(**kwargs):
    return {
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "fc",
        **kwargs,
    }


def _rcfg(**kwargs):
    return _cfg(
        **dict(
            {
                "interpolation": "bicubic",
                "crop_pct": 0.95,
                "test_input_size": (3, 288, 288),
                "test_crop_pct": 1.0,
            },
            **kwargs,
        ),
    )


default_cfgs = generate_default_cfgs({"resnet50.a1_in1k": _rcfg(), "resnet101.a1h_in1k": _rcfg()})


@register_model
def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model."""
    model_args = {"block": _Bottleneck, "layers": [3, 4, 6, 3]}
    return _create_resnet("resnet50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model."""
    model_args = {"block": _Bottleneck, "layers": [3, 4, 23, 3]}
    return _create_resnet("resnet101", pretrained, **dict(model_args, **kwargs))
