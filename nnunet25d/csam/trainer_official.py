from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase
from nnunet25d.csam.official_wrapper import OfficialCSAMCenterSliceWrapper
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainer25DCSAMOfficial(_nnUNetTrainer25DBase):
    num_input_slices = 3
    uncertainty = True
    semantic = True
    positional = True
    slice_attention = True
    rank = 5

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

        self._set_batch_size_and_oversample()
        base_num_input_channels = determine_num_input_channels(
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        self.num_input_channels = base_num_input_channels * self.num_input_slices
        self.num_input_channels_per_slice = base_num_input_channels
        self.enable_deep_supervision = False

        _, arch_init_kwargs, _ = self._resolve_architecture_definition()
        n_stages = int(arch_init_kwargs["n_stages"])
        features_per_stage = arch_init_kwargs["features_per_stage"]
        base_num = int(features_per_stage[0] if isinstance(features_per_stage, (list, tuple)) else features_per_stage)

        self.network = OfficialCSAMCenterSliceWrapper(
            input_channels_per_slice=base_num_input_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            num_input_slices=self.num_input_slices,
            num_layers=n_stages,
            base_num=base_num,
            semantic=self.semantic,
            positional=self.positional,
            slice_attention=self.slice_attention,
            uncertainty=self.uncertainty,
            rank=self.rank,
        ).to(self.device)
        self.network.initialize(self.network)

        if self._do_i_compile():
            self.print_to_log_file("Using torch.compile...")
            self.network = torch.compile(self.network)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True
        empty_cache(self.device)


class nnUNetTrainer25DCSAMOfficialNoUncertainty(nnUNetTrainer25DCSAMOfficial):
    uncertainty = False


__all__ = [
    "nnUNetTrainer25DCSAMOfficial",
    "nnUNetTrainer25DCSAMOfficialNoUncertainty",
]
