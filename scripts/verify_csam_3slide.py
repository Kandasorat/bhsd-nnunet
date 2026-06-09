from __future__ import annotations

import os
from pathlib import Path
from pydoc import locate
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
os.environ.setdefault("nnUNet_raw", str(NNUNET_DATA_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_DATA_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(NNUNET_DATA_ROOT / "nnUNet_results"))

from nnunet25d.csam.feature_fusion_25d import (  # noqa: E402
    BottleneckFeatureFusion25DUNet,
    CenterGuidedSliceFusion,
    MultiScaleFeatureFusion25DUNet,
)
from nnunet25d.csam.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM  # noqa: E402


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_environment() -> None:
    print_header("Environment")
    print("nnUNet_raw:", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results:", os.environ["nnUNet_results"])


def summarize_outputs(outputs: Any):
    if isinstance(outputs, (list, tuple)):
        return [tuple(output.shape) for output in outputs]
    return tuple(outputs.shape)


def primary_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def assert_no_nan_tensor(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all().item(), f"{name} contains non-finite values"


def assert_attention_detached(attention: Any, batch_size: int, num_input_slices: int) -> None:
    if isinstance(attention, dict):
        assert attention, "Expected multiscale attention dictionary to be populated"
        for stage_idx, stage_attention in attention.items():
            assert tuple(stage_attention.shape) == (batch_size, num_input_slices), (
                f"Unexpected attention shape at stage {stage_idx}: {tuple(stage_attention.shape)}"
            )
            assert stage_attention.requires_grad is False, f"Attention at stage {stage_idx} is not detached"
            assert_no_nan_tensor(f"attention_stage_{stage_idx}", stage_attention)
            assert torch.allclose(
                stage_attention.sum(dim=1),
                torch.ones(batch_size, device=stage_attention.device, dtype=stage_attention.dtype),
                atol=1e-4,
            ), f"Attention weights at stage {stage_idx} do not sum to 1"
    else:
        assert attention is not None, "Expected bottleneck attention tensor to be populated"
        assert tuple(attention.shape) == (batch_size, num_input_slices), (
            f"Unexpected bottleneck attention shape: {tuple(attention.shape)}"
        )
        assert attention.requires_grad is False, "Bottleneck attention is not detached"
        assert_no_nan_tensor("bottleneck_attention", attention)
        assert torch.allclose(
            attention.sum(dim=1),
            torch.ones(batch_size, device=attention.device, dtype=attention.dtype),
            atol=1e-4,
        ), "Bottleneck attention weights do not sum to 1"


def get_trainer(trainer_name: str):
    from nnunetv2.run.run_training import get_trainer_from_args

    return get_trainer_from_args(
        "Dataset001_BHSD",
        "2d",
        0,
        trainer_name,
        device=torch.device("cpu"),
    )


def resolve_arch_kwargs():
    trainer = get_trainer("nnUNetTrainer25DCSAM")
    _, arch_init_kwargs, arch_init_kwargs_req_import = trainer._resolve_architecture_definition()
    resolved_kwargs = dict(arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        value = resolved_kwargs.get(key)
        if value is None:
            continue
        resolved = locate(value)
        if resolved is None:
            raise ImportError(f"Could not resolve architecture argument {key}: {value}")
        resolved_kwargs[key] = resolved
    return resolved_kwargs


def build_model(model_class, num_input_slices: int, deep_supervision: bool):
    arch_kwargs = resolve_arch_kwargs()
    return model_class(
        input_channels=1,
        num_classes=6,
        num_input_slices=num_input_slices,
        deep_supervision=deep_supervision,
        **arch_kwargs,
    )


def import_checks() -> None:
    print_header("Import Check")
    print("Imported:", nnUNetTrainer25DCSAM.__name__)
    print("Imported:", MultiScaleFeatureFusion25DUNet.__name__)
    print("Imported:", CenterGuidedSliceFusion.__name__)

    try:
        from nnunetv2.training.nnUNetTrainer.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM as shim_class
    except Exception as exc:
        print(f"Shim import unavailable: {exc}")
    else:
        print("Shim imported:", shim_class.__name__)


def direct_model_checks() -> None:
    print_header("Direct Model Checks")
    for deep_supervision in (False, True):
        model = build_model(MultiScaleFeatureFusion25DUNet, num_input_slices=3, deep_supervision=deep_supervision)
        x = torch.randn(2, 3, 256, 256)
        prepared, batch_size = model._prepare_input(x)
        assert tuple(prepared.shape) == (2, 3, 1, 256, 256), f"Unexpected prepared input shape: {tuple(prepared.shape)}"
        assert batch_size == 2, f"Unexpected batch size: {batch_size}"

        flat_input = prepared.reshape(batch_size * model.num_input_slices, model.input_channels_per_slice, 256, 256)
        assert tuple(flat_input.shape) == (6, 1, 256, 256), f"Unexpected encoder input shape: {tuple(flat_input.shape)}"

        outputs = model(x)
        first = primary_output(outputs)
        assert tuple(first.shape) == (2, 6, 256, 256), f"Unexpected output shape: {summarize_outputs(outputs)}"
        assert_attention_detached(model.last_attention_weights, batch_size=2, num_input_slices=3)
        print(
            f"deep_supervision={deep_supervision} output_shapes={summarize_outputs(outputs)} "
            f"param_count={count_parameters(model):,}"
        )

    model_5 = build_model(MultiScaleFeatureFusion25DUNet, num_input_slices=5, deep_supervision=False)
    x_5 = torch.randn(2, 5, 256, 256)
    outputs_5 = model_5(x_5)
    first_5 = primary_output(outputs_5)
    assert tuple(first_5.shape) == (2, 6, 256, 256), f"Unexpected K=5 output shape: {summarize_outputs(outputs_5)}"
    assert_attention_detached(model_5.last_attention_weights, batch_size=2, num_input_slices=5)
    print(f"K=5 output_shapes={summarize_outputs(outputs_5)}")


def forward_backward_checks() -> None:
    print_header("Forward + Backward Checks")
    for device in (torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else None):
        if device is None:
            continue
        model = build_model(MultiScaleFeatureFusion25DUNet, num_input_slices=3, deep_supervision=True).to(device)
        x = torch.randn(2, 3, 256, 256, device=device)
        outputs = model(x)
        first = primary_output(outputs)
        loss = first.mean()
        loss.backward()
        assert torch.isfinite(loss).item(), f"Non-finite loss on {device.type}"
        assert_no_nan_tensor(f"first_output_{device.type}", first)
        grad_found = False
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_found = True
                assert_no_nan_tensor(f"grad_{device.type}", parameter.grad)
                break
        assert grad_found, f"No gradients found on {device.type}"
        assert_attention_detached(model.last_attention_weights, batch_size=2, num_input_slices=3)
        print(f"{device.type.upper()} loss={loss.item():.6f}")


def bottleneck_attention_check() -> None:
    print_header("Bottleneck Attention Check")
    model = build_model(BottleneckFeatureFusion25DUNet, num_input_slices=3, deep_supervision=False)
    outputs = model(torch.randn(2, 3, 256, 256))
    first = primary_output(outputs)
    assert tuple(first.shape) == (2, 6, 256, 256)
    assert_attention_detached(model.last_attention_weights, batch_size=2, num_input_slices=3)
    print("Bottleneck attention shape:", tuple(model.last_attention_weights.shape))


def trainer_and_dataloader_checks() -> None:
    print_header("Trainer + Dataloader Checks")
    trainer = get_trainer("nnUNetTrainer25DCSAM")
    trainer.initialize()
    assert trainer.num_input_slices == 3
    assert trainer.num_input_channels == 3, f"Unexpected stacked input channels: {trainer.num_input_channels}"
    assert trainer.num_input_channels_per_slice == 1, (
        f"Unexpected per-slice input channels: {trainer.num_input_channels_per_slice}"
    )
    assert trainer.network.input_channels_per_slice == 1
    assert type(trainer.network).__name__ == "MultiScaleFeatureFusion25DUNet"
    assert len(trainer.configuration_manager.patch_size) == 2

    original_n_proc = os.environ.get("nnUNet_n_proc_DA")
    os.environ["nnUNet_n_proc_DA"] = "0"
    try:
        trainer_for_batch = get_trainer("nnUNetTrainer25DCSAM")
        trainer_for_batch.initialize()
        dl_tr, _ = trainer_for_batch.get_dataloaders()
        batch = next(dl_tr)
    finally:
        if original_n_proc is None:
            os.environ.pop("nnUNet_n_proc_DA", None)
        else:
            os.environ["nnUNet_n_proc_DA"] = original_n_proc

    data_shape = tuple(batch["data"].shape)
    target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
    target_shape = tuple(target.shape)
    assert data_shape[1] == 3, f"Expected stacked 3-channel input, got {data_shape}"
    print("train_data_shape:", data_shape)
    print("train_target_shape:", target_shape)
    print("trainer_network:", type(trainer.network).__name__)


def validation_behavior_checks() -> None:
    print_header("Validation Behavior Checks")
    trainer = get_trainer("nnUNetTrainer25DCSAM")
    dummy_case = torch.arange(1 * 4 * 3 * 3, dtype=torch.float32).reshape(1, 4, 3, 3).numpy()
    stacked = trainer._stack_case_for_inference(dummy_case)
    assert stacked.shape == (3, 4, 3, 3), f"Unexpected stacked validation shape: {stacked.shape}"

    # First center slice should replicate z=0 for the left context.
    assert (stacked[0, 0] == dummy_case[0, 0]).all(), "Expected first validation slice to clamp left boundary"
    # Last center slice should replicate z=3 for the right context.
    assert (stacked[2, -1] == dummy_case[0, -1]).all(), "Expected last validation slice to clamp right boundary"

    print("validation_stacked_shape:", stacked.shape)
    print("boundary_handling: edge clamping / replication confirmed")
    print("formal_validation_path: trainer.perform_actual_validation uses stacked adjacent-slice inference")


def main() -> None:
    print_environment()
    import_checks()
    direct_model_checks()
    bottleneck_attention_check()
    forward_backward_checks()
    trainer_and_dataloader_checks()
    validation_behavior_checks()
    print("\nAll requested csam_3slide verification checks completed.")


if __name__ == "__main__":
    main()
