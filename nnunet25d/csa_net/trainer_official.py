from __future__ import annotations

import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase
from nnunet25d.csa_net.official_wrapper import OfficialCSANet3SliceWrapper
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer25DCSANetOfficial(_nnUNetTrainer25DBase):
    """Three-slice center-prediction CSA-Net, adapted to BHSD/nnU-Net I/O."""

    num_input_slices = 3

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device,
    ):
        # Keep an explicit nnU-Net trainer signature. nnU-Net records these
        # arguments for checkpoint restore and cannot introspect *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.batch_size = int(os.environ.get("BHSD_CSA_BATCH_SIZE", "2"))

    def set_deep_supervision_enabled(self, enabled: bool):
        # The official CSA-Net decoder emits one full-resolution output.
        self.enable_deep_supervision = False

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("Trainer is already initialized")

        self._set_batch_size_and_oversample()
        self.batch_size = int(os.environ.get("BHSD_CSA_BATCH_SIZE", "2"))
        base_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        self.num_input_channels = base_channels * self.num_input_slices
        self.enable_deep_supervision = False

        self.network = OfficialCSANet3SliceWrapper(
            input_channels_per_slice=base_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            image_size=256,
            pretrained_path=os.environ.get("BHSD_CSA_PRETRAINED"),
        ).to(self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True
        empty_cache(self.device)


__all__ = ["nnUNetTrainer25DCSANetOfficial"]
