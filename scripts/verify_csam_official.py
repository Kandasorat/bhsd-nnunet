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

from nnunet25d.csam.CSAM_networks import C2BAMUNet  # noqa: E402
from nnunet25d.csam.official_wrapper import OfficialCSAMCenterSliceWrapper  # noqa: E402
from nnunet25d.csam.trainer_official import nnUNetTrainer25DCSAMOfficial  # noqa: E402


def get_trainer():
    from nnunetv2.run.run_training import get_trainer_from_args

    return get_trainer_from_args(
        "Dataset001_BHSD",
        "2d",
        0,
        "nnUNetTrainer25DCSAMOfficial",
        device=torch.device("cpu"),
    )


def build_wrapper():
    trainer = get_trainer()
    _, arch_init_kwargs, _ = trainer._resolve_architecture_definition()
    return OfficialCSAMCenterSliceWrapper(
        input_channels_per_slice=1,
        num_classes=6,
        num_input_slices=3,
        num_layers=int(arch_init_kwargs["n_stages"]),
        base_num=int(arch_init_kwargs["features_per_stage"][0]),
    )


def main():
    print("Imported official network:", C2BAMUNet.__name__)
    print("Imported wrapper:", OfficialCSAMCenterSliceWrapper.__name__)
    print("Imported trainer:", nnUNetTrainer25DCSAMOfficial.__name__)
    try:
        from nnunetv2.training.nnUNetTrainer.trainer_csam_official import nnUNetTrainer25DCSAMOfficial as shim_class
    except Exception as exc:
        print("Shim import skipped:", exc)
    else:
        print("Shim imported:", shim_class.__name__)

    official_model = C2BAMUNet(
        input_channels=1,
        num_classes=6,
        num_layers=6,
        base_num=32,
        batch_size=3,
    )
    raw_outputs = official_model(torch.randn(3, 1, 256, 256))
    assert tuple(raw_outputs.shape) == (3, 6, 256, 256), tuple(raw_outputs.shape)
    print("official_forward_shape:", tuple(raw_outputs.shape))

    wrapper = build_wrapper()
    wrapped_outputs = wrapper(torch.randn(2, 3, 256, 256))
    assert tuple(wrapped_outputs.shape) == (2, 6, 256, 256), tuple(wrapped_outputs.shape)
    print("wrapper_forward_shape:", tuple(wrapped_outputs.shape))

    loss = wrapped_outputs.mean()
    loss.backward()
    print("wrapper_cpu_loss:", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()
