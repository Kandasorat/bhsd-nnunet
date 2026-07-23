from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch
from torch import nn


SYMMETRIC_RELIABILITY_METHODS = {
    "e0_symmetric_control",
    "e1_symmetric_lowpass",
    "e2_reliability_gate",
}


class SymmetricReliabilityInputAdapter(nn.Module):
    """Single-pass, neighbor-swap-invariant 2.5D input adapter.

    The backbone receives three invariant channels per modality: center slice,
    neighbor mean, and half absolute neighbor difference. A shared descriptor
    produces a zero-initialized center-channel residual. E0 is the capacity
    control, E1 gates low-frequency persistence without an uncertainty input,
    and E2 adds the symmetric disagreement descriptor to that gate.

    The contrasts are defined on slice indices and are not physical z-axis
    derivatives. Physical spacing is therefore analysed separately.
    """

    def __init__(
        self,
        backbone: nn.Module,
        method: str,
        num_slices: int = 3,
        channels_per_slice: int = 1,
        descriptor_channels: int = 8,
    ) -> None:
        super().__init__()
        if method not in SYMMETRIC_RELIABILITY_METHODS:
            raise ValueError(
                f"Unknown symmetric reliability method {method!r}; "
                f"expected one of {sorted(SYMMETRIC_RELIABILITY_METHODS)}"
            )
        if num_slices != 3:
            raise ValueError("The symmetric reliability adapter requires exactly three slices")
        if channels_per_slice < 1 or descriptor_channels < 1:
            raise ValueError("Channel counts must be positive")

        self.backbone = backbone
        self.method_name = method
        self.num_slices = num_slices
        self.channels_per_slice = channels_per_slice
        self.descriptor_channels = descriptor_channels
        self.backbone_passes_per_prediction = 1

        self.descriptor = nn.Sequential(
            nn.Conv2d(channels_per_slice, descriptor_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(descriptor_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        # Used in every arm so E0-E2 have the same functional parameter count.
        self.reliability_gate = nn.Conv2d(descriptor_channels * 3, channels_per_slice, kernel_size=1)
        self.project = nn.Conv2d(descriptor_channels, channels_per_slice, kernel_size=1, bias=False)
        nn.init.zeros_(self.project.weight)

    @property
    def decoder(self):
        return self.backbone.decoder

    def _group(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S*C, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        expected = self.num_slices * self.channels_per_slice
        if channels != expected:
            raise ValueError(f"Expected {expected} input channels, got {channels}")
        return x.reshape(batch, self.num_slices, self.channels_per_slice, height, width)

    def invariant_basis(self, grouped: torch.Tensor) -> torch.Tensor:
        previous, center, following = grouped.unbind(dim=1)
        neighbor_mean = 0.5 * (previous + following)
        neighbor_disagreement = 0.5 * (previous - following).abs()
        return torch.stack([center, neighbor_mean, neighbor_disagreement], dim=1)

    def encoded_slices(self, grouped: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = grouped.shape
        encoded = self.descriptor(grouped.reshape(batch * slices, channels, height, width))
        return encoded.reshape(batch, slices, self.descriptor_channels, height, width)

    def reliability_and_residual(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        previous, center, following = encoded.unbind(dim=1)
        stable = (previous + center + following) / sqrt(3.0)
        disagreement = 0.5 * ((previous - center).abs() + (following - center).abs())
        persistence = stable - center
        zeros = torch.zeros_like(center)

        if self.method_name == "e0_symmetric_control":
            gate_input = torch.cat([center, center, zeros], dim=1)
            residual_source = center
        elif self.method_name == "e1_symmetric_lowpass":
            gate_input = torch.cat([center, stable, zeros], dim=1)
            residual_source = persistence
        else:
            gate_input = torch.cat([center, stable, disagreement], dim=1)
            residual_source = persistence
        gate = torch.sigmoid(self.reliability_gate(gate_input))
        return gate, residual_source

    def adapted_input(self, x: torch.Tensor) -> torch.Tensor:
        grouped = self._group(x)
        invariant = self.invariant_basis(grouped)
        gate, residual_source = self.reliability_and_residual(self.encoded_slices(grouped))
        # Keep the mathematical order explicit: g * P(S - F0). P is zero
        # initialized, so the shared symmetric basis is unchanged initially.
        delta = gate * self.project(residual_source)
        invariant = invariant.clone()
        invariant[:, 0] = invariant[:, 0] + delta
        return invariant.reshape_as(x)

    def forward(self, x: torch.Tensor):
        return self.backbone(self.adapted_input(x))

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        return self.backbone.compute_conv_feature_map_size(input_size)
