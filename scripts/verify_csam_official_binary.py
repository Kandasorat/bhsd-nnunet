from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
os.environ.setdefault("nnUNet_raw", str(NNUNET_DATA_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_DATA_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(NNUNET_DATA_ROOT / "nnUNet_results"))

from nnunet25d.csam.official_wrapper import OfficialCSAMCenterSliceWrapper  # noqa: E402


def get_trainer(dataset_name: str):
    from nnunetv2.run.run_training import get_trainer_from_args

    return get_trainer_from_args(
        dataset_name,
        "2d",
        0,
        "nnUNetTrainer25DCSAMOfficial",
        device=torch.device("cpu"),
    )


def main():
    dataset_name = (
        "Dataset002_BHSD_Binary"
        if (Path(os.environ["nnUNet_preprocessed"]) / "Dataset002_BHSD_Binary").exists()
        else "Dataset001_BHSD"
    )
    trainer = get_trainer(dataset_name)
    _, arch_init_kwargs, _ = trainer._resolve_architecture_definition()
    model = OfficialCSAMCenterSliceWrapper(
        input_channels_per_slice=1,
        num_classes=1,
        num_input_slices=3,
        num_layers=int(arch_init_kwargs["n_stages"]),
        base_num=int(arch_init_kwargs["features_per_stage"][0]),
    )
    outputs = model(torch.randn(2, 3, 256, 256))
    assert tuple(outputs.shape) == (2, 1, 256, 256), tuple(outputs.shape)
    print("binary_wrapper_forward_shape:", tuple(outputs.shape))
    print("binary_arch_source_dataset:", dataset_name)


if __name__ == "__main__":
    main()
