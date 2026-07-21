from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LightweightSliceAttentionInputAdapter(nn.Module):
    """Vectorized, identity-initialized slice attention in front of a 2D backbone.

    The adapter keeps the sample and slice axes separate while it estimates one
    scalar weight per input slice. The attended slices are then restored to the
    stacked-channel representation expected by the existing nnU-Net 2.5D
    backbone. No feature is pooled across different samples in the batch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_slices: int = 3,
        channels_per_slice: int = 1,
        descriptor_channels: int = 8,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if num_slices < 3 or num_slices % 2 == 0:
            raise ValueError("num_slices must be an odd integer >= 3")
        if channels_per_slice < 1:
            raise ValueError("channels_per_slice must be positive")

        self.backbone = backbone
        self.num_slices = int(num_slices)
        self.channels_per_slice = int(channels_per_slice)
        self.descriptor_channels = int(descriptor_channels)

        self.descriptor = nn.Sequential(
            nn.Conv2d(
                self.channels_per_slice,
                self.descriptor_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.descriptor_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

        hidden = max(self.num_slices * int(expansion), self.num_slices)
        self.slice_mlp = nn.Sequential(
            nn.Linear(self.num_slices, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_slices),
        )

        # With zero logits, 2 * sigmoid(logits) is exactly one. The experiment
        # therefore starts from the unmodified 2.5D input rather than from an
        # arbitrary attenuation of every CT slice.
        nn.init.zeros_(self.slice_mlp[-1].weight)
        nn.init.zeros_(self.slice_mlp[-1].bias)

    @property
    def decoder(self):
        return self.backbone.decoder

    def compute_slice_scales(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S*C, H, W], got {tuple(x.shape)}")
        batch_size, channels, height, width = x.shape
        expected_channels = self.num_slices * self.channels_per_slice
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {channels}")

        slices = x.reshape(
            batch_size * self.num_slices,
            self.channels_per_slice,
            height,
            width,
        )
        features = self.descriptor(slices).reshape(
            batch_size,
            self.num_slices,
            self.descriptor_channels,
            height,
            width,
        )
        avg_descriptor = features.mean(dim=(2, 3, 4))
        max_descriptor = features.amax(dim=(2, 3, 4))
        logits = self.slice_mlp(avg_descriptor) + self.slice_mlp(max_descriptor)
        return 2.0 * torch.sigmoid(logits)

    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        scales = self.compute_slice_scales(x)
        grouped = x.reshape(
            batch_size,
            self.num_slices,
            self.channels_per_slice,
            height,
            width,
        )
        attended = grouped * scales[:, :, None, None, None]
        return self.backbone(attended.reshape(batch_size, -1, height, width))

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        return self.backbone.compute_conv_feature_map_size(input_size)
