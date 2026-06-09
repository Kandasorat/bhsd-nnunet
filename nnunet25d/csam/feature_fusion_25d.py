from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Type, Union

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class CenterGuidedSliceFusion(nn.Module):
    def __init__(self, channels: int, num_slices: int):
        super().__init__()
        if num_slices < 1:
            raise ValueError("num_slices must be >= 1")
        self.channels = channels
        self.num_slices = num_slices
        self.center_index = num_slices // 2
        self.query_proj = nn.Linear(channels, channels, bias=False)
        self.key_proj = nn.Linear(channels, channels, bias=False)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if x.ndim != 5:
            raise ValueError(f"Expected x to have shape [B, K, C, H, W], got {tuple(x.shape)}")
        batch_size, num_slices, channels, _, _ = x.shape
        if num_slices != self.num_slices:
            raise ValueError(f"Expected {self.num_slices} slices, got {num_slices}")
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        pooled = x.mean(dim=(-1, -2))
        center_query = self.query_proj(pooled[:, self.center_index, :])
        keys = self.key_proj(pooled)
        attn_logits = torch.einsum("bc,bkc->bk", center_query, keys) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=1)
        fused = torch.einsum("bk,bkchw->bchw", attn_weights, x)

        if return_attention:
            return fused, attn_weights
        return fused


class FeatureFusion25DUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_input_slices: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        fusion_mode: str = "bottleneck",
    ):
        super().__init__()
        if num_input_slices < 3 or num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be an odd integer >= 3")
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        spatial_dim = convert_conv_op_to_dim(conv_op)
        if spatial_dim != 2:
            raise ValueError("FeatureFusion25DUNet only supports 2D convolutional architectures")

        self.num_input_slices = num_input_slices
        self.input_channels_per_slice = input_channels
        self.deep_supervision = deep_supervision
        self.fusion_mode = fusion_mode
        self._last_attention_weights = None

        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        if fusion_mode == "bottleneck":
            self.fusion_stage_indices = (len(self.encoder.output_channels) - 1,)
        elif fusion_mode == "multiscale":
            self.fusion_stage_indices = tuple(range(len(self.encoder.output_channels)))
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.fusion_modules = nn.ModuleDict(
            {
                str(stage_idx): CenterGuidedSliceFusion(self.encoder.output_channels[stage_idx], num_input_slices)
                for stage_idx in self.fusion_stage_indices
            }
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=nonlin_first,
        )

    @property
    def last_attention_weights(self):
        return self._last_attention_weights

    @property
    def slice_fusion(self):
        return self.fusion_modules[str(len(self.encoder.output_channels) - 1)]

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.ndim == 4:
            batch_size, num_channels, _, _ = x.shape
            if num_channels != self.num_input_slices * self.input_channels_per_slice:
                raise ValueError(
                    f"Expected {self.num_input_slices * self.input_channels_per_slice} channels, got {num_channels}"
                )
            x = x.reshape(batch_size, self.num_input_slices, self.input_channels_per_slice, *x.shape[-2:])
            return x, batch_size

        if x.ndim == 5:
            batch_size, num_slices, channels_per_slice, _, _ = x.shape
            if num_slices != self.num_input_slices:
                raise ValueError(f"Expected {self.num_input_slices} slices, got {num_slices}")
            if channels_per_slice != self.input_channels_per_slice:
                raise ValueError(
                    f"Expected {self.input_channels_per_slice} channels per slice, got {channels_per_slice}"
                )
            return x, batch_size

        raise ValueError(
            f"Expected input shape [B, K*C, H, W] or [B, K, C, H, W], got {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor):
        x, batch_size = self._prepare_input(x)
        flat_input = x.reshape(batch_size * self.num_input_slices, self.input_channels_per_slice, *x.shape[-2:])
        slice_skips = self.encoder(flat_input)

        fused_skips: List[torch.Tensor] = []
        attention_weights: Dict[int, torch.Tensor] = {}
        for stage_idx, stage_feature in enumerate(slice_skips):
            reshaped = stage_feature.reshape(
                batch_size,
                self.num_input_slices,
                stage_feature.shape[1],
                stage_feature.shape[2],
                stage_feature.shape[3],
            )
            if stage_idx in self.fusion_stage_indices:
                fused_feature, scale_attention = self.fusion_modules[str(stage_idx)](reshaped, return_attention=True)
                attention_weights[stage_idx] = scale_attention.detach()
            else:
                fused_feature = reshaped[:, self.num_input_slices // 2]
            fused_skips.append(fused_feature)

        if self.fusion_mode == "bottleneck":
            self._last_attention_weights = attention_weights[self.fusion_stage_indices[0]]
        else:
            self._last_attention_weights = {
                stage_idx: stage_attention.detach() for stage_idx, stage_attention in attention_weights.items()
            }
        return self.decoder(fused_skips)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        if len(input_size) != 2:
            raise AssertionError("input_size should be the spatial size only, for example (H, W)")
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        module.apply(InitWeights_He(1e-2))


class BottleneckFeatureFusion25DUNet(FeatureFusion25DUNet):
    def __init__(self, *args, **kwargs):
        kwargs["fusion_mode"] = "bottleneck"
        super().__init__(*args, **kwargs)


class MultiScaleFeatureFusion25DUNet(FeatureFusion25DUNet):
    def __init__(self, *args, **kwargs):
        kwargs["fusion_mode"] = "multiscale"
        super().__init__(*args, **kwargs)
