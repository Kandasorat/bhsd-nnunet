from __future__ import annotations

import os
import random

import numpy as np
import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunet25d.attention.spectral_slice_fusion import SpectralSliceFusionInputAdapter
from nnunet25d.attention.unified_slice_adapters import UnifiedSliceAdapter
from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase as _Historical25DBase
from nnunet25d.common.dataloader_25d import nnUNetDataLoader25D


class BHSDFixed1000Policy:
    """Reproducible nnU-Net policy with a non-overridable 1000-epoch horizon.

    This mixin deliberately has no relationship to ``BHSDEarlyStoppingMixin``.
    The only training loop is nnU-Net's standard fixed-length loop.
    """

    fixed_model_seed = 3407
    fixed_data_seed = 1_003_410
    fixed_num_epochs = 1000
    fixed_iterations_per_epoch = 250
    fixed_val_iterations_per_epoch = 50
    fixed_initial_lr = 0.01
    fixed_weight_decay = 3e-5
    fixed_poly_exponent = 0.9

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        requested_seed = int(os.environ.get("BHSD_SEED", str(self.fixed_model_seed)))
        requested_data_seed = int(os.environ.get("BHSD_DATA_SEED", str(self.fixed_data_seed)))
        if requested_seed != self.fixed_model_seed:
            raise RuntimeError(
                f"{self.__class__.__name__} is locked to model seed {self.fixed_model_seed}, "
                f"but BHSD_SEED={requested_seed}"
            )
        if requested_data_seed != self.fixed_data_seed:
            raise RuntimeError(
                f"{self.__class__.__name__} is locked to data seed {self.fixed_data_seed}, "
                f"but BHSD_DATA_SEED={requested_data_seed}"
            )
        self.bhsd_seed = self.fixed_model_seed
        self.bhsd_data_seed = self.fixed_data_seed
        self.bhsd_deterministic = True
        self.num_epochs = self.fixed_num_epochs
        self.num_iterations_per_epoch = self.fixed_iterations_per_epoch
        self.num_val_iterations_per_epoch = self.fixed_val_iterations_per_epoch
        self.initial_lr = self.fixed_initial_lr
        self.weight_decay = self.fixed_weight_decay
        self.performance_early_stopping = False

    def _apply_reproducibility_settings(self, base_seed: int) -> None:
        seed = int(base_seed) + int(getattr(self, "local_rank", 0))
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.fixed_initial_lr,
            weight_decay=self.fixed_weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(
            optimizer,
            self.fixed_initial_lr,
            self.fixed_num_epochs,
            exponent=self.fixed_poly_exponent,
        )
        return optimizer, scheduler

    def on_train_start(self):
        if os.environ.get("nnUNet_n_proc_DA") != "0":
            raise RuntimeError("fixed1000 requires nnUNet_n_proc_DA=0 for an explicit reproducible data RNG stream")
        if not self.was_initialized:
            self._apply_reproducibility_settings(self.bhsd_seed)
            self.initialize()
        self._apply_reproducibility_settings(self.bhsd_data_seed)
        super().on_train_start()
        self.print_to_log_file(
            "BHSD fixed1000 policy: "
            f"model_seed={self.bhsd_seed}, data_seed={self.bhsd_data_seed}, epochs={self.num_epochs}, "
            f"iterations_per_epoch={self.num_iterations_per_epoch}, val_iterations={self.num_val_iterations_per_epoch}, "
            "early_stopping=disabled, primary_checkpoint=checkpoint_final.pth",
            also_print_to_console=True,
        )

    def on_train_epoch_start(self):
        self._apply_reproducibility_settings(self.bhsd_data_seed + 2 * self.current_epoch)
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self._apply_reproducibility_settings(self.bhsd_data_seed + 2 * self.current_epoch + 1)
        super().on_validation_epoch_start()


class nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407(
    BHSDFixed1000Policy, nnUNetTrainer
):
    """Locked standard 2D/3D core trainer; the configuration selects dimensionality."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if len(self.configuration_manager.patch_size) == 2:
            self.configuration_manager.configuration["patch_size"] = [256, 256]


class _nnUNetTrainer25DFixed1000Base(BHSDFixed1000Policy, nnUNetTrainer):
    """Fixed-length copy of the historical 2.5D execution policy.

    Methods are reused as unbound implementations, not inherited from the
    historical class. Therefore the MRO cannot reach its early-stopping mixin,
    while inference, center supervision, boundary replication and dataloading
    remain byte-for-byte the same implementations.
    """

    num_input_slices = 3
    dataloader_class = nnUNetDataLoader25D
    dataloader_kwargs = {}

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["patch_size"] = [256, 256]

    _resolve_architecture_definition = _Historical25DBase._resolve_architecture_definition
    _do_i_compile = _Historical25DBase._do_i_compile
    _get_slice_indices = _Historical25DBase._get_slice_indices
    _stack_case_for_inference = _Historical25DBase._stack_case_for_inference
    perform_actual_validation = _Historical25DBase.perform_actual_validation
    initialize = _Historical25DBase.initialize
    get_dataloaders = _Historical25DBase.get_dataloaders

    def _adapt_network(self, network: torch.nn.Module, channels_per_slice: int) -> torch.nn.Module:
        return network


class nnUNetTrainer_25D_A0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407(
    _nnUNetTrainer25DFixed1000Base
):
    """Three consecutive slices, center supervision, replicated z boundaries."""


class _Fixed1000UnifiedAdapterBase(_nnUNetTrainer25DFixed1000Base):
    adapter_method: str

    def _adapt_network(self, network: torch.nn.Module, channels_per_slice: int) -> torch.nn.Module:
        return UnifiedSliceAdapter(
            backbone=network,
            method=self.adapter_method,
            num_slices=self.num_input_slices,
            channels_per_slice=channels_per_slice,
            descriptor_channels=8,
        )


class _Fixed1000SpectralAdapterBase(_nnUNetTrainer25DFixed1000Base):
    spectral_method: str

    def _adapt_network(self, network: torch.nn.Module, channels_per_slice: int) -> torch.nn.Module:
        return SpectralSliceFusionInputAdapter(
            backbone=network,
            method=self.spectral_method,
            num_slices=self.num_input_slices,
            channels_per_slice=channels_per_slice,
            descriptor_channels=8,
        )


def _seeded_class(name: str, base: type, seed: int) -> type:
    return type(
        name,
        (base,),
        {
            "fixed_model_seed": seed,
            "__module__": __name__,
            "__doc__": f"Preregistered fixed1000 {base.__name__} trainer for model seed {seed}.",
        },
    )


class _C1Fixed1000(_Fixed1000UnifiedAdapterBase):
    adapter_method = "adapter_control"


class _C2Fixed1000(_Fixed1000UnifiedAdapterBase):
    adapter_method = "csa_center_neighbor"


class _D0Fixed1000(_Fixed1000SpectralAdapterBase):
    spectral_method = "d0_control"


class _D1Fixed1000(_Fixed1000SpectralAdapterBase):
    spectral_method = "d1_lowpass"


for _mechanism, _base in (("C1", _C1Fixed1000), ("C2", _C2Fixed1000), ("D0", _D0Fixed1000), ("D1", _D1Fixed1000)):
    for _seed in (3407, 1234, 5678):
        _name = (
            f"nnUNetTrainer_25D_{_mechanism}Fixed1000NoEarlyStopping"
            f"FinalCheckpointPrimarySeed{_seed}"
        )
        globals()[_name] = _seeded_class(_name, _base, _seed)


__all__ = [name for name in globals() if name.startswith("nnUNetTrainer_")]

