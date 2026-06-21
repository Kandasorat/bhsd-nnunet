from __future__ import annotations

import os
from pathlib import Path
from pydoc import locate
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
os.environ["nnUNet_raw"] = str(NNUNET_DATA_ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NNUNET_DATA_ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(NNUNET_DATA_ROOT / "nnUNet_results")

from nnunet25d.csam.feature_fusion_25d import MultiScaleFeatureFusion25DUNet  # noqa: E402


def has_preprocessed_binary_dataset() -> bool:
    dataset_dir = Path(os.environ["nnUNet_preprocessed"]) / "Dataset002_BHSD_Binary"
    return dataset_dir.exists()


def get_trainer(dataset_name: str = "Dataset002_BHSD_Binary"):
    from nnunetv2.run.run_training import get_trainer_from_args

    return get_trainer_from_args(
        dataset_name,
        "2d",
        0,
        "nnUNetTrainer25DCSAM",
        device=torch.device("cpu"),
    )


def resolve_arch_kwargs(dataset_name: str):
    trainer = get_trainer(dataset_name)
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


def main() -> None:
    from nnunet25d.csam.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM

    print(nnUNetTrainer25DCSAM)
    from nnunetv2.training.nnUNetTrainer.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM as shim_class

    print(shim_class)

    arch_dataset = "Dataset002_BHSD_Binary" if has_preprocessed_binary_dataset() else "Dataset001_BHSD"
    model = MultiScaleFeatureFusion25DUNet(
        input_channels=1,
        num_classes=2,
        num_input_slices=3,
        deep_supervision=True,
        **resolve_arch_kwargs(arch_dataset),
    )
    outputs = model(torch.randn(2, 3, 256, 256))
    first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    assert tuple(first.shape) == (2, 2, 256, 256), tuple(first.shape)
    print("binary_output_shape:", tuple(first.shape))
    print("binary_arch_source_dataset:", arch_dataset)

    if not has_preprocessed_binary_dataset():
        print("binary_dataset_preprocessed: missing")
        print("trainer_checks: skipped until Dataset002_BHSD_Binary has completed nnUNetv2_plan_and_preprocess -d 2")
        return

    trainer = get_trainer()
    trainer.initialize()
    assert trainer.label_manager.num_segmentation_heads == 2
    assert trainer.num_input_channels == 3
    assert trainer.num_input_channels_per_slice == 1
    print("binary_dataset_preprocessed: ready")
    print("binary_num_segmentation_heads:", trainer.label_manager.num_segmentation_heads)
    print("binary_trainer_network:", type(trainer.network).__name__)


if __name__ == "__main__":
    main()
