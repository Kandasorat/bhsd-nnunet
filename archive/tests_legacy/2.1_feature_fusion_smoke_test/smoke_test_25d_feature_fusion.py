from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Type

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
os.environ["nnUNet_raw"] = str(NNUNET_DATA_ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NNUNET_DATA_ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(NNUNET_DATA_ROOT / "nnUNet_results")

from nnunet25d.feature_fusion_25d import (
    BottleneckFeatureFusion25DUNet,
    FeatureFusion25DUNet,
    MultiScaleFeatureFusion25DUNet,
)
from nnunet25d.trainer_25d_feature_fusion import (
    nnUNetTrainer25DFeatureFusion,
    nnUNetTrainer25DFeatureFusionBottleneck,
    nnUNetTrainer25DFeatureFusionMultiScale,
    nnUNetTrainer25DFeatureFusionMultiScale_5Slice,
)


def build_model(
    model_class: Type[FeatureFusion25DUNet],
    num_input_slices: int,
    deep_supervision: bool,
    device: torch.device,
) -> FeatureFusion25DUNet:
    return model_class(
        input_channels=1,
        num_classes=6,
        num_input_slices=num_input_slices,
        n_stages=8,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
        conv_op=torch.nn.Conv2d,
        kernel_sizes=[[3, 3]] * 8,
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_conv_per_stage=[2] * 8,
        n_conv_per_stage_decoder=[2] * 7,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    ).to(device)


def summarize_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        return [tuple(o.shape) for o in outputs]
    return tuple(outputs.shape)


def primary_output(outputs: torch.Tensor):
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_attention_is_valid(model: FeatureFusion25DUNet) -> None:
    attention = model.last_attention_weights
    if isinstance(attention, dict):
        assert attention, "Expected multiscale attention weights to be populated"
        for stage_idx, weights in attention.items():
            assert not torch.isnan(weights).any(), f"NaN attention weights at stage {stage_idx}"
            sums = weights.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
                f"Attention weights at stage {stage_idx} do not sum to 1"
            )
    else:
        assert attention is not None, "Expected bottleneck attention weights to be populated"
        assert not torch.isnan(attention).any(), "NaN attention weights"
        sums = attention.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "Attention weights do not sum to 1"


def cpu_forward_test(
    label: str,
    model_class: Type[FeatureFusion25DUNet],
    num_input_slices: int,
    deep_supervision: bool,
) -> None:
    device = torch.device("cpu")
    model = build_model(model_class, num_input_slices, deep_supervision, device)
    model.eval()
    x = torch.randn(2, num_input_slices, 256, 256, device=device)
    with torch.no_grad():
        outputs = model(x)
    first = primary_output(outputs)
    assert tuple(first.shape) == (2, 6, 256, 256), f"Unexpected output shape: {summarize_outputs(outputs)}"
    assert_attention_is_valid(model)
    print(f"CPU forward {label}: {summarize_outputs(outputs)}")
    print(f"Parameter count {label}: {count_parameters(model):,}")


def _first_grad_norm(model: torch.nn.Module) -> float:
    for param in model.parameters():
        if param.grad is not None:
            value = float(param.grad.norm().item())
            if not math.isnan(value):
                return value
    raise AssertionError("No valid gradients were found")


def backward_test(
    label: str,
    model_class: Type[FeatureFusion25DUNet],
    num_input_slices: int,
    device: torch.device,
) -> None:
    model = build_model(model_class, num_input_slices, deep_supervision=True, device=device)
    model.train()
    x = torch.randn(2, num_input_slices, 256, 256, device=device)
    target = torch.randint(0, 6, (2, 256, 256), device=device)
    outputs = model(x)
    first = primary_output(outputs)
    loss = torch.nn.functional.cross_entropy(first, target)
    loss.backward()

    assert torch.isfinite(loss).item(), f"Loss is not finite for {label}"
    assert torch.isfinite(first).all().item(), f"Output contains non-finite values for {label}"
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all().item(), f"Gradient contains non-finite values for {label}: {name}"
    assert_attention_is_valid(model)
    print(
        f"{device.type.upper()} backward {label}: loss={loss.item():.6f}, "
        f"grad_norm={_first_grad_norm(model):.6f}"
    )


def trainer_import_test() -> None:
    trainers = [
        nnUNetTrainer25DFeatureFusion,
        nnUNetTrainer25DFeatureFusionBottleneck,
        nnUNetTrainer25DFeatureFusionMultiScale,
        nnUNetTrainer25DFeatureFusionMultiScale_5Slice,
    ]
    print("Imported trainers:", ", ".join(cls.__name__ for cls in trainers))


def trainer_instantiation_test() -> None:
    from nnunet25d.install_extension import main as install_extension_main

    install_extension_main()

    try:
        from nnunetv2.run.run_training import get_trainer_from_args
    except Exception as exc:
        print(f"nnU-Net trainer lookup skipped: {exc}")
        return

    for trainer_name in (
        "nnUNetTrainer25DFeatureFusionBottleneck",
        "nnUNetTrainer25DFeatureFusionMultiScale",
        "nnUNetTrainer25DFeatureFusionMultiScale_5Slice",
    ):
        trainer = get_trainer_from_args(
            "Dataset001_BHSD",
            "2d",
            0,
            trainer_name,
            device=torch.device("cpu"),
        )
        print(f"nnU-Net lookup trainer class: {type(trainer).__name__}")


def main() -> None:
    trainer_import_test()
    cpu_forward_test("bottleneck K=3 no-DS", BottleneckFeatureFusion25DUNet, num_input_slices=3, deep_supervision=False)
    cpu_forward_test("bottleneck K=3 DS", BottleneckFeatureFusion25DUNet, num_input_slices=3, deep_supervision=True)
    cpu_forward_test("multiscale K=3 no-DS", MultiScaleFeatureFusion25DUNet, num_input_slices=3, deep_supervision=False)
    cpu_forward_test("multiscale K=5 no-DS", MultiScaleFeatureFusion25DUNet, num_input_slices=5, deep_supervision=False)
    backward_test("bottleneck K=3", BottleneckFeatureFusion25DUNet, num_input_slices=3, device=torch.device("cpu"))
    backward_test("multiscale K=3", MultiScaleFeatureFusion25DUNet, num_input_slices=3, device=torch.device("cpu"))
    if torch.cuda.is_available():
        backward_test(
            "multiscale K=3",
            MultiScaleFeatureFusion25DUNet,
            num_input_slices=3,
            device=torch.device("cuda"),
        )
    else:
        print("CUDA backward skipped: CUDA not available")
    trainer_instantiation_test()


if __name__ == "__main__":
    main()
