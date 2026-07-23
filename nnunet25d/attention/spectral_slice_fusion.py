from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch
from torch import nn


SPECTRAL_METHODS = {
    "d0_control",
    "d1_lowpass",
    "d2_odd_difference",
    "d3_curvature_gate",
    "d4_orthogonal_all",
    "d5_adaptive_oriented",
    "d6_adaptive_invariant",
}

PREDICTION_MODES = {"native", "original", "swapped", "swap_average"}


def path3_spectral_transform(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal graph-Fourier transform on the three-node slice path."""
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, C, H, W], got {tuple(x.shape)}")
    previous, center, following = x.unbind(dim=1)
    z0 = (previous + center + following) / sqrt(3.0)
    z1 = (previous - following) / sqrt(2.0)
    z2 = (previous - 2.0 * center + following) / sqrt(6.0)
    return z0, z1, z2


def _average_predictions(first, second):
    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor):
            raise TypeError("Backbone prediction structures differ")
        return 0.5 * (first + second)
    if isinstance(first, list):
        if not isinstance(second, list) or len(first) != len(second):
            raise TypeError("Backbone prediction structures differ")
        return [_average_predictions(a, b) for a, b in zip(first, second)]
    if isinstance(first, tuple):
        if not isinstance(second, tuple) or len(first) != len(second):
            raise TypeError("Backbone prediction structures differ")
        return tuple(_average_predictions(a, b) for a, b in zip(first, second))
    raise TypeError(f"Unsupported backbone prediction type: {type(first).__name__}")


class SpectralSliceFusionInputAdapter(nn.Module):
    """Center-only residual adapter using a fixed orthonormal slice-axis basis.

    The basis is the graph-Fourier basis of a three-node path: low-frequency
    persistence, odd orientation-sensitive difference, and even curvature.
    These are discrete slice-index contrasts, not physical derivatives.
    """

    def __init__(
        self,
        backbone: nn.Module,
        method: str,
        num_slices: int = 3,
        channels_per_slice: int = 1,
        descriptor_channels: int = 8,
        prediction_mode: str = "native",
    ) -> None:
        super().__init__()
        if method not in SPECTRAL_METHODS:
            raise ValueError(f"Unknown spectral method {method!r}; expected one of {sorted(SPECTRAL_METHODS)}")
        if num_slices != 3:
            raise ValueError("The path-spectrum adapter requires exactly three slices")
        if channels_per_slice < 1 or descriptor_channels < 1:
            raise ValueError("Channel counts must be positive")

        self.backbone = backbone
        self.method_name = method
        self.num_slices = num_slices
        self.channels_per_slice = channels_per_slice
        self.descriptor_channels = descriptor_channels
        self.set_prediction_mode(prediction_mode)

        self.descriptor = nn.Sequential(
            nn.Conv2d(channels_per_slice, descriptor_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(descriptor_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        if method == "d3_curvature_gate":
            self.curvature_gate = nn.Conv2d(descriptor_channels * 2, 1, kernel_size=1)
        else:
            self.curvature_gate = None
        if method in {"d5_adaptive_oriented", "d6_adaptive_invariant"}:
            self.band_gate = nn.Conv2d(descriptor_channels * 3, 3, kernel_size=1)
        else:
            self.band_gate = None

        # All arms use the same 3C -> input-channel projection. Unused spectral
        # slots are explicit zeros in D0-D3, keeping the functional control fair.
        self.project = nn.Conv2d(descriptor_channels * 3, channels_per_slice, kernel_size=1, bias=False)
        nn.init.zeros_(self.project.weight)

    @property
    def decoder(self):
        return self.backbone.decoder

    def _validate(self, x: torch.Tensor) -> tuple[int, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S*C, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        expected = self.num_slices * self.channels_per_slice
        if channels != expected:
            raise ValueError(f"Expected {expected} input channels, got {channels}")
        return batch, height, width

    def encoded_slices(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = self._validate(x)
        grouped = x.reshape(batch, self.num_slices, self.channels_per_slice, height, width)
        encoded = self.descriptor(grouped.reshape(-1, self.channels_per_slice, height, width))
        encoded = encoded.reshape(batch, self.num_slices, self.descriptor_channels, height, width)
        return grouped, encoded

    def spectral_context(self, encoded: torch.Tensor) -> torch.Tensor:
        z0, z1, z2 = path3_spectral_transform(encoded)
        zeros = torch.zeros_like(z0)

        if self.method_name == "d0_control":
            return torch.cat([encoded[:, 1], zeros, zeros], dim=1)
        if self.method_name == "d1_lowpass":
            return torch.cat([z0, zeros, zeros], dim=1)
        if self.method_name == "d2_odd_difference":
            return torch.cat([zeros, z1, zeros], dim=1)
        if self.method_name == "d3_curvature_gate":
            gate = torch.sigmoid(self.curvature_gate(torch.cat([encoded[:, 1], z2.abs()], dim=1)))
            return torch.cat([zeros, zeros, gate * z2], dim=1)
        if self.method_name == "d4_orthogonal_all":
            return torch.cat([z0, z1, z2], dim=1)

        gate_input = torch.cat([z0, z1.abs(), z2.abs()], dim=1)
        weights = torch.softmax(self.band_gate(gate_input), dim=1)
        odd = z1 if self.method_name == "d5_adaptive_oriented" else z1.abs()
        return torch.cat(
            [
                weights[:, 0:1] * z0,
                weights[:, 1:2] * odd,
                weights[:, 2:3] * z2,
            ],
            dim=1,
        )

    def adapted_input(self, x: torch.Tensor) -> torch.Tensor:
        grouped, encoded = self.encoded_slices(x)
        delta = self.project(self.spectral_context(encoded))
        adapted = grouped.clone()
        adapted[:, 1] = adapted[:, 1] + delta
        return adapted.reshape_as(x)

    def swap_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width = self._validate(x)
        grouped = x.reshape(batch, self.num_slices, self.channels_per_slice, height, width)
        return grouped[:, [2, 1, 0]].reshape_as(x)

    def set_prediction_mode(self, mode: str) -> None:
        if mode not in PREDICTION_MODES:
            raise ValueError(f"Unknown prediction mode {mode!r}; expected one of {sorted(PREDICTION_MODES)}")
        self.prediction_mode = mode

    def forward(self, x: torch.Tensor):
        prediction = self.backbone(self.adapted_input(x))
        mode = self.prediction_mode
        if mode == "original" or (mode == "native" and self.method_name != "d6_adaptive_invariant"):
            return prediction

        swapped_prediction = self.backbone(self.adapted_input(self.swap_neighbors(x)))
        if mode == "swapped":
            return swapped_prediction

        # Reynolds/group averaging over the two-element neighbor-swap group.
        # This is D6's native mode and an inference-only probe for any D arm.
        return _average_predictions(prediction, swapped_prediction)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        return self.backbone.compute_conv_feature_map_size(input_size)
