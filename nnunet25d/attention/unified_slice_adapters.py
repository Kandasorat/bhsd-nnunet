from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class HardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=False) / 6.0


class HardSwish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AdapterControl(nn.Module):
    """Capacity control: the shared descriptor is used without attention."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CSAMSliceGate(nn.Module):
    """Deterministic, batched adaptation of the official CSAM slice module."""

    def __init__(self, num_slices: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = max(num_slices, num_slices * expansion)
        self.mlp = nn.Sequential(
            nn.Linear(num_slices, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_slices),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = x.mean(dim=(2, 3, 4))
        max_descriptor = x.amax(dim=(2, 3, 4))
        weights = torch.sigmoid(self.mlp(avg_descriptor) + self.mlp(max_descriptor))
        return x * weights[:, :, None, None, None]


class ECASliceGate(nn.Module):
    """ECA's GAP -> local Conv1d -> sigmoid rule, adapted to the slice axis."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ECA kernel_size must be odd")
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = x.mean(dim=(2, 3, 4)).unsqueeze(1)
        weights = torch.sigmoid(self.conv(descriptor)).squeeze(1)
        return x * weights[:, :, None, None, None]


class PixelWiseCrossSliceAttention(nn.Module):
    """Pixel-wise softmax over adjacent slices, following XAG-Net's CSA rule."""

    def __init__(self, channels: int, num_slices: int) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.channels = channels
        self.score = nn.Conv2d(channels * num_slices, num_slices, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = x.shape
        logits = self.score(x.reshape(batch, slices * channels, height, width))
        logits = logits.reshape(batch, slices, 1, height, width)
        weights = torch.softmax(logits, dim=1)
        return x + x * weights


class CrossSliceNonLocal(nn.Module):
    """Registered and compact form of CSA-Net's non-local cross-attention block."""

    def __init__(self, channels: int, inter_channels: int) -> None:
        super().__init__()
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, center: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = center.shape
        values = self.g(center).flatten(2).transpose(1, 2)
        theta = self.theta(center).flatten(2)
        phi = self.phi(other).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.matmul(phi, theta), dim=-1)
        attended = torch.matmul(attention, values).transpose(1, 2).contiguous()
        attended = attended.reshape(batch, self.inter_channels, height, width)
        return self.out(attended)


class CSACenterToNeighborAttention(nn.Module):
    """CSA-Net-style center/previous/next non-local attention at a 16x16 grid."""

    def __init__(self, channels: int, token_grid: int = 16) -> None:
        super().__init__()
        self.token_grid = token_grid
        inter_channels = max(1, channels // 2)
        self.previous = CrossSliceNonLocal(channels, inter_channels)
        self.self_attention = CrossSliceNonLocal(channels, inter_channels)
        self.next = CrossSliceNonLocal(channels, inter_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError("CSA center-to-neighbor adapter requires exactly three slices")
        batch, _, channels, height, width = x.shape
        low = F.adaptive_avg_pool2d(x.reshape(batch * 3, channels, height, width), self.token_grid)
        low = low.reshape(batch, 3, channels, self.token_grid, self.token_grid)
        center = low[:, 1]
        fused = self.fuse(
            torch.cat(
                [
                    self.previous(center, low[:, 0]),
                    self.self_attention(center, center),
                    self.next(center, low[:, 2]),
                ],
                dim=1,
            )
        )
        fused = F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=False)
        result = x.clone()
        result[:, 1] = result[:, 1] + fused
        return result


class CBAMAttention(nn.Module):
    """Official CBAM ordering: channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)
        channel_weights = torch.sigmoid(self.mlp(avg) + self.mlp(maximum))[:, :, None, None]
        x = x * channel_weights
        spatial_descriptor = torch.cat([x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(spatial_descriptor))


class CoordinateAttention(nn.Module):
    """Coordinate Attention using the official two-direction pooling formulation."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.activation = HardSwish()
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.activation(self.bn1(self.conv1(torch.cat([pooled_h, pooled_w], dim=2))))
        encoded_h, encoded_w = torch.split(encoded, [height, width], dim=2)
        weights_h = torch.sigmoid(self.conv_h(encoded_h))
        weights_w = torch.sigmoid(self.conv_w(encoded_w.transpose(2, 3)))
        return x * weights_h * weights_w


class AxialSliceConv(nn.Module):
    """P3D-style temporal-only convolution after the shared spatial descriptor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        volume = x.permute(0, 2, 1, 3, 4)
        return (volume + self.temporal(volume)).permute(0, 2, 1, 3, 4)


class AxialCSASequentialFusion(nn.Module):
    """F1: local axial aggregation followed by CSA center-neighbor refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.axial = AxialSliceConv(channels)
        self.csa = CSACenterToNeighborAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csa(self.axial(x))


class AxialCSAParallelFusion(nn.Module):
    """F2: learn a normalized mixture of axial and CSA refinement branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.axial = AxialSliceConv(channels)
        self.csa = CSACenterToNeighborAttention(channels)
        self.branch_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.branch_logits, dim=0)
        return weights[0] * self.axial(x) + weights[1] * self.csa(x)


METHODS = {
    "adapter_control",
    "csam_slice_gate",
    "eca_slice_gate",
    "pixelwise_cross_slice",
    "csa_center_neighbor",
    "cbam",
    "coordinate_attention",
    "axial_slice_conv",
    "axial_csa_sequential",
    "axial_csa_parallel",
}


class UnifiedSliceAdapter(nn.Module):
    """Common input adapter used by every harmonized 2.5D module ablation."""

    def __init__(
        self,
        backbone: nn.Module,
        method: str,
        num_slices: int = 3,
        channels_per_slice: int = 1,
        descriptor_channels: int = 8,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown adapter method {method!r}; expected one of {sorted(METHODS)}")
        if num_slices != 3:
            raise ValueError("The screened adapters are locked to three slices")
        if channels_per_slice < 1:
            raise ValueError("channels_per_slice must be positive")
        self.backbone = backbone
        self.method_name = method
        self.num_slices = num_slices
        self.channels_per_slice = channels_per_slice
        self.descriptor_channels = descriptor_channels

        self.descriptor = nn.Sequential(
            nn.Conv2d(channels_per_slice, descriptor_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(descriptor_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        flat_channels = num_slices * descriptor_channels
        if method == "adapter_control":
            self.mechanism = AdapterControl()
            self._flat_mechanism = False
        elif method == "csam_slice_gate":
            self.mechanism = CSAMSliceGate(num_slices)
            self._flat_mechanism = False
        elif method == "eca_slice_gate":
            self.mechanism = ECASliceGate(kernel_size=3)
            self._flat_mechanism = False
        elif method == "pixelwise_cross_slice":
            self.mechanism = PixelWiseCrossSliceAttention(descriptor_channels, num_slices)
            self._flat_mechanism = False
        elif method == "csa_center_neighbor":
            self.mechanism = CSACenterToNeighborAttention(descriptor_channels)
            self._flat_mechanism = False
        elif method == "cbam":
            self.mechanism = CBAMAttention(flat_channels)
            self._flat_mechanism = True
        elif method == "coordinate_attention":
            self.mechanism = CoordinateAttention(flat_channels)
            self._flat_mechanism = True
        elif method == "axial_slice_conv":
            self.mechanism = AxialSliceConv(descriptor_channels)
            self._flat_mechanism = False
        elif method == "axial_csa_sequential":
            self.mechanism = AxialCSASequentialFusion(descriptor_channels)
            self._flat_mechanism = False
        else:
            self.mechanism = AxialCSAParallelFusion(descriptor_channels)
            self._flat_mechanism = False

        self.project = nn.Conv2d(descriptor_channels, channels_per_slice, kernel_size=1)
        # Every arm begins as exactly the standard stacked-input baseline. The
        # residual projection then learns how much adapted information to add.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

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

    def adapted_input(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width = self._validate(x)
        grouped = x.reshape(batch, self.num_slices, self.channels_per_slice, height, width)
        features = self.descriptor(grouped.reshape(-1, self.channels_per_slice, height, width))
        features = features.reshape(batch, self.num_slices, self.descriptor_channels, height, width)
        if self._flat_mechanism:
            flat = features.reshape(batch, -1, height, width)
            refined = self.mechanism(flat).reshape_as(features)
        else:
            refined = self.mechanism(features)
        delta = self.project(refined.reshape(-1, self.descriptor_channels, height, width))
        delta = delta.reshape_as(grouped)
        return (grouped + delta).reshape(batch, -1, height, width)

    def forward(self, x: torch.Tensor):
        return self.backbone(self.adapted_input(x))

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        return self.backbone.compute_conv_feature_map_size(input_size)
