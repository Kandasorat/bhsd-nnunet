from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d.attention.symmetric_reliability_fusion import (  # noqa: E402
    SYMMETRIC_RELIABILITY_METHODS,
    SymmetricReliabilityInputAdapter,
)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Identity()
        self.head = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor):
        logits = self.head(x)
        return [logits, nn.functional.avg_pool2d(logits, kernel_size=2)]

    def compute_conv_feature_map_size(self, input_size):
        return 0


def main() -> None:
    torch.manual_seed(3407)
    x = torch.randn(2, 3, 32, 30)
    swapped = x[:, [2, 1, 0]]
    parameter_counts = {}

    for method in sorted(SYMMETRIC_RELIABILITY_METHODS):
        adapter = SymmetricReliabilityInputAdapter(_Backbone(), method=method)
        parameter_counts[method] = sum(
            parameter.numel()
            for name, parameter in adapter.named_parameters()
            if not name.startswith("backbone.")
        )

        grouped = adapter._group(x)
        invariant = adapter.invariant_basis(grouped).reshape_as(x)
        if not torch.equal(adapter.adapted_input(x), invariant):
            raise AssertionError(f"{method}: zero-init residual must preserve the invariant input basis")
        if not torch.equal(invariant, adapter.invariant_basis(adapter._group(swapped)).reshape_as(x)):
            raise AssertionError(f"{method}: raw invariant basis changed after neighbor reversal")

        original_prediction = adapter(x)
        swapped_prediction = adapter(swapped)
        for original, reversed_order in zip(original_prediction, swapped_prediction):
            if not torch.equal(original, reversed_order):
                error = (original - reversed_order).abs().max().item()
                raise AssertionError(f"{method}: exact neighbor-swap invariance failed; max error={error}")

        loss = sum(prediction.square().mean() for prediction in original_prediction)
        loss.backward()
        if adapter.project.weight.grad is None or not torch.isfinite(adapter.project.weight.grad).all():
            raise AssertionError(f"{method}: missing/non-finite residual projection gradient")
        if adapter.reliability_gate.weight.grad is None:
            raise AssertionError(f"{method}: reliability gate is not functionally connected")
        if not torch.isfinite(adapter.reliability_gate.weight.grad).all():
            raise AssertionError(f"{method}: non-finite reliability-gate gradient")

    if len(set(parameter_counts.values())) != 1:
        raise AssertionError(f"E0-E2 adapter capacities differ: {parameter_counts}")

    print("Symmetric reliability fusion verification passed.")
    print(f"Validated methods: {', '.join(sorted(SYMMETRIC_RELIABILITY_METHODS))}")
    print(f"Equal adapter parameters per arm: {next(iter(parameter_counts.values()))}")
    print("Validated: invariant basis, zero-init residual, exact swap invariance, deep supervision, gradients")


if __name__ == "__main__":
    main()
