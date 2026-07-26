from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Stage3Arm = Literal["R0", "R1"]


class SharedSliceStem(nn.Module):
    """The preregistered shared 1->16->16 slice stem."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class LogitResidualBranch(nn.Module):
    """Symmetric, lightweight logit correction fixed by the Stage3 protocol."""

    expected_parameter_count = 18_342

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        if num_classes != 6:
            raise ValueError("Stage3 is frozen to six output channels (background plus five ICH classes)")
        self.num_classes = num_classes
        self.slice_stem = SharedSliceStem()
        self.correction = nn.Sequential(
            nn.Conv2d(38, 32, 3, padding=1, bias=True),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=True),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1, bias=True),
        )
        final = self.correction[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def final_conv(self) -> nn.Conv2d:
        return self.correction[-1]

    def forward(
        self,
        triplet: torch.Tensor,
        center_logits: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ):
        if triplet.ndim != 4 or triplet.shape[1] != 3:
            raise ValueError(f"Expected triplet [N,3,H,W], got {tuple(triplet.shape)}")
        if center_logits.ndim != 4 or center_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected center logits [N,{self.num_classes},H,W], got {tuple(center_logits.shape)}"
            )
        if triplet.shape[0] != center_logits.shape[0] or triplet.shape[2:] != center_logits.shape[2:]:
            raise ValueError("Triplet and full-resolution center logits must share batch and spatial shape")

        previous = self.slice_stem(triplet[:, 0:1])
        center = self.slice_stem(triplet[:, 1:2])
        following = self.slice_stem(triplet[:, 2:3])
        neighbour = (previous + following) / 2.0
        features = torch.cat((center, neighbour, center_logits.detach()), dim=1)
        delta = self.correction(features)
        if return_intermediates:
            return delta, {"center": center, "neighbour": neighbour}
        return delta


class FrozenCenterResidualWrapper(nn.Module):
    """One frozen center backbone plus the locked R0/R1 logit residual."""

    def __init__(self, center: nn.Module, arm: Stage3Arm, num_classes: int = 6) -> None:
        super().__init__()
        normalized_arm = arm.upper()
        if normalized_arm not in {"R0", "R1"}:
            raise ValueError("arm must be exactly R0 or R1")
        self.arm: Stage3Arm = normalized_arm  # type: ignore[assignment]
        self.center = center
        self.residual = LogitResidualBranch(num_classes=num_classes)
        self.register_buffer("delta_enabled", torch.tensor(1.0), persistent=True)
        self._freeze_center()

    def _freeze_center(self) -> None:
        self.center.eval()
        for parameter in self.center.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.center.eval()
        return self

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        if not hasattr(self.center, "decoder") or not hasattr(self.center.decoder, "deep_supervision"):
            raise AttributeError("Frozen center does not expose decoder.deep_supervision")
        self.center.decoder.deep_supervision = bool(enabled)

    def materialize_branch_triplet(self, augmented_triplet: torch.Tensor) -> torch.Tensor:
        if augmented_triplet.ndim != 4 or augmented_triplet.shape[1] != 3:
            raise ValueError(f"Expected augmented input [N,3,H,W], got {tuple(augmented_triplet.shape)}")
        if self.arm == "R1":
            return augmented_triplet
        center = augmented_triplet[:, 1:2]
        return torch.cat((center, center, center), dim=1)

    @staticmethod
    def _full_resolution_logits(base_logits):
        if isinstance(base_logits, (list, tuple)):
            if not base_logits:
                raise ValueError("Center backbone returned an empty deep-supervision sequence")
            return base_logits[0]
        return base_logits

    @staticmethod
    def _add_delta(base_logits, delta_full: torch.Tensor):
        if isinstance(base_logits, tuple):
            return tuple(
                logits + (
                    delta_full
                    if logits.shape[2:] == delta_full.shape[2:]
                    else F.interpolate(delta_full, size=logits.shape[2:], mode="bilinear", align_corners=False)
                )
                for logits in base_logits
            )
        if isinstance(base_logits, list):
            return [
                logits + (
                    delta_full
                    if logits.shape[2:] == delta_full.shape[2:]
                    else F.interpolate(delta_full, size=logits.shape[2:], mode="bilinear", align_corners=False)
                )
                for logits in base_logits
            ]
        return base_logits + delta_full

    def forward(self, augmented_triplet: torch.Tensor):
        if augmented_triplet.ndim != 4 or augmented_triplet.shape[1] != 3:
            raise ValueError(f"Expected Stage3 input [N,3,H,W], got {tuple(augmented_triplet.shape)}")
        center_input = augmented_triplet[:, 1:2]
        self.center.eval()
        with torch.no_grad():
            base_logits = self.center(center_input)

        # This branch is deliberately an exact identity path for S3-02.
        if float(self.delta_enabled.detach().cpu()) == 0.0:
            return base_logits

        full_resolution = self._full_resolution_logits(base_logits)
        branch_triplet = self.materialize_branch_triplet(augmented_triplet)
        delta_full = self.residual(branch_triplet, full_resolution.detach())
        return self._add_delta(base_logits, delta_full)

    def residual_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.residual.parameters())

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
