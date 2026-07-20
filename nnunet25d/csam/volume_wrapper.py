from __future__ import annotations

from typing import Sequence

import torch
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from nnunet25d.csam.CSAM_networks import C2BAMUNet


class OfficialCSAMVolumeWrapper(nn.Module):
    """Official CSAM U-Net operating on an ordered fixed-length slice window."""

    def __init__(self, input_channels: int, num_classes: int, sequence_length: int = 32):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.model = C2BAMUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            num_layers=6,
            base_num=32,
            batch_size=self.sequence_length,
            semantic=True,
            positional=True,
            slice=True,
            uncertainty=True,
            rank=5,
        )
        self.model.apply(InitWeights_He(1e-2))

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [S, C, H, W], got {tuple(x.shape)}")
        if x.shape[0] != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} ordered slices, got {x.shape[0]}")
        return self.model(x)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        return 0
