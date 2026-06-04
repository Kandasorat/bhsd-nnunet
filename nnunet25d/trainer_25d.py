from pathlib import Path

import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from nnunet25d.dataloader_25d import nnUNetDataLoader25D
from nnunet25d.dataloader_spacing_aware import nnUNetDataLoaderSpacingAware25D


class _nnUNetTrainer25DBase(nnUNetTrainer):
    num_input_slices = 3
    dataloader_class = nnUNetDataLoader25D
    dataloader_kwargs = {}

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

        # Call the current nnU-Net base implementation explicitly so the
        # custom 2.5D trainers stay compatible across nnU-Net API changes.
        self.network = nnUNetTrainer.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            self.enable_deep_supervision,
        ).to(self.device)

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

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 2:
            raise RuntimeError(f"{self.__class__.__name__} only supports 2D configurations")

        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = self.dataloader_class(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            probabilistic_oversampling=self.probabilistic_oversampling,
            transforms=tr_transforms,
            num_input_slices=self.num_input_slices,
            **self.dataloader_kwargs,
        )
        dl_val = self.dataloader_class(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            probabilistic_oversampling=self.probabilistic_oversampling,
            transforms=val_transforms,
            num_input_slices=self.num_input_slices,
            **self.dataloader_kwargs,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainer_25D(_nnUNetTrainer25DBase):
    num_input_slices = 3


class nnUNetTrainer_25D_5Slice(_nnUNetTrainer25DBase):
    num_input_slices = 5


class nnUNetTrainer_SpacingAware25D(_nnUNetTrainer25DBase):
    num_input_slices = 3
    dataloader_class = nnUNetDataLoaderSpacingAware25D
    dataloader_kwargs = {
        "spacing_csv": Path.cwd() / "bhsd_spacing_summary.csv",
        "target_context_mm": 2.5,
        "min_slice_step": 1,
        "max_slice_step": 5,
    }
