from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase
from nnunet25d.stage3.dataloader import Stage3TripletDataLoader
from nnunet25d.stage3.model import FrozenCenterResidualWrapper
from nnunet25d.stage3.provenance import checkpoint_path, validate_locked_checkpoint
from nnunet25d.stage3.transforms import synchronize_stage3_intensity_transforms


_TRAINING_APPROVAL_VALUE = "APPROVED_AFTER_STAGE3_TEST_GATE"


class _nnUNetTrainerStage3FrozenResidual(_nnUNetTrainer25DBase):
    """Locked Stage3 trainer. Training remains hard-gated pending a later approval."""

    num_input_slices = 3
    dataloader_class = Stage3TripletDataLoader
    stage3_arm: str
    stage3_model_seed: int

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if configuration != "2d":
            raise ValueError("Stage3 is locked to the nnU-Net 2d configuration")
        self.bhsd_seed = int(self.stage3_model_seed)
        self.bhsd_data_seed = 1_003_410
        self.bhsd_deterministic = True
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.num_epochs = 1000
        self.early_stop_min_epochs = 300
        self.early_stop_patience = 100
        self.early_stop_min_delta = 1e-4
        self.early_stop_metric = "ema_fg_dice"
        self._early_stop_enabled = True
        self._early_stop_best = None
        self._early_stop_bad_epochs = 0
        self._early_stop_triggered = False
        self.b_checkpoint_path: Path | None = None
        self.b_checkpoint_sha256: str | None = None

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        transforms = _nnUNetTrainer25DBase.get_training_transforms(*args, **kwargs)
        synchronize_stage3_intensity_transforms(transforms)
        return transforms

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("Stage3 trainer was initialized twice")

        self._set_batch_size_and_oversample()
        base_num_input_channels = determine_num_input_channels(
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        if base_num_input_channels != 1:
            raise RuntimeError(f"Stage3 BHSD expects exactly one CT channel, got {base_num_input_channels}")
        self.num_input_channels = 3

        arch_class_name, arch_init_kwargs, arch_init_kwargs_req_import = self._resolve_architecture_definition()
        center = get_network_from_plans(
            arch_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            1,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=self.enable_deep_supervision,
        )

        configured_root = os.environ.get("BHSD_STAGE3_B_CHECKPOINT_ROOT")
        if configured_root is None:
            if os.name != "nt":
                raise EnvironmentError(
                    "BHSD_STAGE3_B_CHECKPOINT_ROOT is required outside the audited Windows workstation"
                )
            configured_root = str(checkpoint_path(0).parents[1])
        root = Path(configured_root)
        b_path = checkpoint_path(int(self.fold), root)
        b_hash = validate_locked_checkpoint(b_path, int(self.fold))
        checkpoint = torch.load(b_path, map_location="cpu", weights_only=False)
        if checkpoint.get("trainer_name") != "nnUNetTrainer_BHSDEarlyStop":
            raise RuntimeError(f"Unexpected B trainer: {checkpoint.get('trainer_name')}")
        center.load_state_dict(checkpoint["network_weights"], strict=True)
        del checkpoint

        self.network = FrozenCenterResidualWrapper(
            center=center,
            arm=self.stage3_arm,
            num_classes=self.label_manager.num_segmentation_heads,
        ).to(self.device)
        self.b_checkpoint_path = b_path
        self.b_checkpoint_sha256 = b_hash
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True

    def _stage3_wrapper(self) -> FrozenCenterResidualWrapper:
        network = self.network.module if self.is_ddp else self.network
        if not isinstance(network, FrozenCenterResidualWrapper):
            raise TypeError(f"Unexpected Stage3 network type: {type(network).__name__}")
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        self._stage3_wrapper().set_deep_supervision_enabled(enabled)

    def configure_optimizers(self):
        if self.network is None:
            raise RuntimeError("Network must exist before configuring Stage3 optimizer")
        residual_parameters = [
            parameter
            for name, parameter in self.network.named_parameters()
            if name.startswith("residual.") and parameter.requires_grad
        ]
        if not residual_parameters:
            raise RuntimeError("No trainable residual parameters found")
        unexpected = [
            name
            for name, parameter in self.network.named_parameters()
            if parameter.requires_grad and not name.startswith("residual.")
        ]
        if unexpected:
            raise RuntimeError(f"Non-residual trainable parameters found: {unexpected}")
        optimizer = torch.optim.SGD(
            residual_parameters,
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def _assert_training_authorized(self) -> None:
        if int(self.fold) == 0:
            raise RuntimeError("Stage3 fold0 training is permanently forbidden")
        if self.stage3_model_seed in {1234, 5678} and int(self.fold) != 1:
            raise RuntimeError("Seeds 1234 and 5678 are preregistered only for the fold1 gate")
        if os.environ.get("BHSD_STAGE3_TRAINING_APPROVAL") != _TRAINING_APPROVAL_VALUE:
            raise RuntimeError(
                "Stage3 training is not approved. Complete the test/freeze gate and obtain separate approval first."
            )

    def run_training(self):
        self._assert_training_authorized()
        return super().run_training()

    def perform_actual_validation(self, save_probabilities: bool = False):
        if int(self.fold) == 0:
            raise RuntimeError("Stage3 fold0 performance evaluation is forbidden")
        return super().perform_actual_validation(save_probabilities=save_probabilities)


class nnUNetTrainer_Stage3_R0(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R0"
    stage3_model_seed = 3407


class nnUNetTrainer_Stage3_R1(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R1"
    stage3_model_seed = 3407


class nnUNetTrainer_Stage3_R0Seed1234(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R0"
    stage3_model_seed = 1234


class nnUNetTrainer_Stage3_R1Seed1234(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R1"
    stage3_model_seed = 1234


class nnUNetTrainer_Stage3_R0Seed5678(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R0"
    stage3_model_seed = 5678


class nnUNetTrainer_Stage3_R1Seed5678(_nnUNetTrainerStage3FrozenResidual):
    stage3_arm = "R1"
    stage3_model_seed = 5678


__all__ = [name for name in globals() if name.startswith("nnUNetTrainer_Stage3_")]
