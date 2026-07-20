from __future__ import annotations

import os
from pathlib import Path
import sys
import json

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
for key, local_path in {
    "nnUNet_raw": NNUNET_DATA_ROOT / "nnUNet_raw",
    "nnUNet_preprocessed": NNUNET_DATA_ROOT / "nnUNet_preprocessed",
    "nnUNet_results": NNUNET_DATA_ROOT / "nnUNet_results",
}.items():
    configured = os.environ.get(key)
    if not configured or not Path(configured).exists():
        os.environ[key] = str(local_path)

from nnunet25d.csam.official_wrapper import OfficialCSAMCenterSliceWrapper  # noqa: E402


def main():
    dataset_name = (
        "Dataset002_BHSD_Binary"
        if (Path(os.environ["nnUNet_preprocessed"]) / "Dataset002_BHSD_Binary").exists()
        else "Dataset001_BHSD"
    )
    plans_path = Path(os.environ["nnUNet_preprocessed"]) / dataset_name / "nnUNetPlans.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    architecture = plans["configurations"]["2d"]["architecture"]
    arch_init_kwargs = architecture.get("arch_kwargs", architecture.get("arch_kwargs_req_import"))
    if not isinstance(arch_init_kwargs, dict):
        raise RuntimeError(f"Could not read 2D architecture kwargs from {plans_path}")
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
