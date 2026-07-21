from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d.attention.spectral_slice_fusion import (  # noqa: E402
    SPECTRAL_METHODS,
    SpectralSliceFusionInputAdapter,
    path3_spectral_transform,
)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Identity()
        self.head = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor):
        full = self.head(x)
        return [full, nn.functional.avg_pool2d(full, kernel_size=2)]

    def compute_conv_feature_map_size(self, input_size):
        return 0


def main() -> None:
    torch.manual_seed(3407)
    features = torch.randn(2, 3, 8, 19, 17, dtype=torch.float64)
    z0, z1, z2 = path3_spectral_transform(features)
    input_energy = features.square().sum()
    spectral_energy = z0.square().sum() + z1.square().sum() + z2.square().sum()
    if not torch.allclose(input_energy, spectral_energy, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"Path-spectrum energy mismatch: {input_energy} versus {spectral_energy}")

    x = torch.randn(2, 3, 32, 32)
    for method in sorted(SPECTRAL_METHODS):
        adapter = SpectralSliceFusionInputAdapter(_Backbone(), method=method)
        adapted = adapter.adapted_input(x)
        if adapted.shape != x.shape:
            raise AssertionError(f"{method}: adapted shape {adapted.shape} != {x.shape}")
        if not torch.equal(adapted, x):
            raise AssertionError(f"{method}: zero-initialized residual must preserve the stacked input")
        prediction = adapter(x)
        if [tuple(p.shape) for p in prediction] != [(2, 4, 32, 32), (2, 4, 16, 16)]:
            raise AssertionError(f"{method}: unexpected deep-supervision shapes")
        sum(p.square().mean() for p in prediction).backward()
        if adapter.project.weight.grad is None or not torch.isfinite(adapter.project.weight.grad).all():
            raise AssertionError(f"{method}: missing/non-finite residual projection gradient")

    invariant = SpectralSliceFusionInputAdapter(_Backbone(), method="d6_adaptive_invariant")
    swapped = x[:, [2, 1, 0]]
    original_prediction = invariant(x)
    swapped_prediction = invariant(swapped)
    for original, reversed_order in zip(original_prediction, swapped_prediction):
        if not torch.equal(original, reversed_order):
            error = (original - reversed_order).abs().max().item()
            raise AssertionError(f"D6 neighbor-swap invariance failed; max error={error}")

    print("Spectral slice fusion verification passed.")
    print(f"Validated methods: {', '.join(sorted(SPECTRAL_METHODS))}")
    print("Validated: orthonormal energy, zero-init identity, deep supervision, backward gradients, D6 invariance")


if __name__ == "__main__":
    main()
