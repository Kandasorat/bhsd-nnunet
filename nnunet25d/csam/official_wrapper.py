from __future__ import annotations

from typing import Sequence, Tuple

import torch
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from nnunet25d.csam.CSAM_networks import C2BAMUNet


class OfficialCSAMCenterSliceWrapper(nn.Module):
    def __init__(
        self,
        input_channels_per_slice: int,
        num_classes: int,
        num_input_slices: int,
        num_layers: int,
        base_num: int = 32,
        semantic: bool = True,
        positional: bool = True,
        slice_attention: bool = True,
        uncertainty: bool = True,
        rank: int = 5,
    ):
        super().__init__()
        if num_input_slices < 3 or num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be an odd integer >= 3")
        self.input_channels_per_slice = input_channels_per_slice
        self.num_input_slices = num_input_slices
        self.center_index = num_input_slices // 2
        self.official_csam = C2BAMUNet(
            input_channels=input_channels_per_slice,
            num_classes=num_classes,
            num_layers=num_layers,
            base_num=base_num,
            batch_size=num_input_slices,
            semantic=semantic,
            positional=positional,
            slice=slice_attention,
            uncertainty=uncertainty,
            rank=rank,
        )

    @property
    def decoder(self):
        return self.official_csam.decoder

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.ndim == 4:
            batch_size, num_channels, _, _ = x.shape
            expected = self.num_input_slices * self.input_channels_per_slice
            if num_channels != expected:
                raise ValueError(f"Expected {expected} channels, got {num_channels}")
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
        raise ValueError(f"Expected input shape [B, K*C, H, W] or [B, K, C, H, W], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = self._prepare_input(x)
        outputs = []
        for b in range(batch_size):
            sample = x[b]
            sample_logits = self.official_csam(sample)
            outputs.append(sample_logits[self.center_index])
        return torch.stack(outputs, dim=0)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        if len(input_size) != 2:
            raise AssertionError("input_size should be the spatial size only, for example (H, W)")
        return 0

    @staticmethod
    def initialize(module):
        module.apply(InitWeights_He(1e-2))
