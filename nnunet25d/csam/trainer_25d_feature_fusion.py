from __future__ import annotations

from pydoc import locate

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase
from nnunet25d.csam.feature_fusion_25d import (
    BottleneckFeatureFusion25DUNet,
    FeatureFusion25DUNet,
    MultiScaleFeatureFusion25DUNet,
)
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class _nnUNetTrainer25DFeatureFusionBase(_nnUNetTrainer25DBase):
    num_input_slices = 3
    network_class: type[FeatureFusion25DUNet] = BottleneckFeatureFusion25DUNet

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

        _, arch_init_kwargs, arch_init_kwargs_req_import = self._resolve_architecture_definition()
        arch_init_kwargs = dict(arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if arch_init_kwargs.get(key) is not None:
                arch_init_kwargs[key] = locate(arch_init_kwargs[key])
        if len(self.configuration_manager.patch_size) != 2:
            raise RuntimeError(f"{self.__class__.__name__} only supports 2D configurations")

        self.network = self.network_class(
            input_channels=base_num_input_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            num_input_slices=self.num_input_slices,
            deep_supervision=self.enable_deep_supervision,
            **arch_init_kwargs,
        ).to(self.device)
        self.network_class.initialize(self.network)

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


class nnUNetTrainer25DFeatureFusionBottleneck(_nnUNetTrainer25DFeatureFusionBase):
    network_class = BottleneckFeatureFusion25DUNet


class nnUNetTrainer25DFeatureFusionMultiScale(_nnUNetTrainer25DFeatureFusionBase):
    network_class = MultiScaleFeatureFusion25DUNet


class nnUNetTrainer25DFeatureFusionMultiScale_5Slice(nnUNetTrainer25DFeatureFusionMultiScale):
    num_input_slices = 5


class nnUNetTrainer25DFeatureFusion(nnUNetTrainer25DFeatureFusionBottleneck):
    """Backward-compatible alias for the original bottleneck-only feature-fusion trainer."""


class nnUNetTrainer25DCSAMBottleneck(nnUNetTrainer25DFeatureFusionBottleneck):
    """Alias using the CSAM naming requested for the new center-guided attention models."""


class nnUNetTrainer25DCSAM(nnUNetTrainer25DFeatureFusionMultiScale):
    """Primary CSAM trainer alias for the multi-scale feature-fusion model."""


class nnUNetTrainer25DCSAM_5Slide(nnUNetTrainer25DFeatureFusionMultiScale_5Slice):
    """Optional 5-slide CSAM trainer alias."""


__all__ = [
    "nnUNetTrainer25DFeatureFusion",
    "nnUNetTrainer25DFeatureFusionBottleneck",
    "nnUNetTrainer25DFeatureFusionMultiScale",
    "nnUNetTrainer25DFeatureFusionMultiScale_5Slice",
    "nnUNetTrainer25DCSAMBottleneck",
    "nnUNetTrainer25DCSAM",
    "nnUNetTrainer25DCSAM_5Slide",
]
