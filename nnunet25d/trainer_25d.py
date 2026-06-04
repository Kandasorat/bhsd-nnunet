import multiprocessing
from pathlib import Path
from time import sleep
import warnings

import numpy as np

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from nnunet25d.dataloader_25d import nnUNetDataLoader25D
from nnunet25d.dataloader_spacing_aware import nnUNetDataLoaderSpacingAware25D


class _nnUNetTrainer25DBase(nnUNetTrainer):
    num_input_slices = 3
    dataloader_class = nnUNetDataLoader25D
    dataloader_kwargs = {}

    def _resolve_architecture_definition(self):
        if all(
            hasattr(self.configuration_manager, attr)
            for attr in (
                "network_arch_class_name",
                "network_arch_init_kwargs",
                "network_arch_init_kwargs_req_import",
            )
        ):
            return (
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
            )

        plans_dict = getattr(self.plans_manager, "plans", None)
        config_name = getattr(self, "configuration_name", None)
        if not isinstance(plans_dict, dict) or config_name is None:
            raise AttributeError("Could not resolve architecture definition for the current nnU-Net version")

        config_dict = plans_dict["configurations"][config_name]
        architecture = config_dict["architecture"]
        return (
            architecture["network_class_name"],
            architecture["arch_kwargs"],
            architecture.get("_kw_requires_import", []),
        )

    def _do_i_compile(self):
        # torch.compile is stable for the stock trainers we use, but it has
        # repeatedly produced opaque runtime errors for the custom 2.5D
        # trainers on this project. Keep 2.5D on eager mode for reliability.
        return False

    def _get_slice_indices(self, center_slice: int, num_slices: int):
        half = self.num_input_slices // 2
        return [min(max(center_slice + offset, 0), num_slices - 1) for offset in range(-half, half + 1)]

    def _stack_case_for_inference(self, data: np.ndarray) -> np.ndarray:
        """
        Convert a raw (C, Z, Y, X) case into the channel-stacked representation
        expected by the custom 2.5D trainers: (C * num_input_slices, Z, Y, X).
        """
        if data.ndim != 4:
            raise RuntimeError(
                f"{self.__class__.__name__} expects validation data with shape (C, Z, Y, X), got {data.shape}"
            )
        num_modalities, num_slices, height, width = data.shape
        stacked = np.empty(
            (num_modalities * self.num_input_slices, num_slices, height, width),
            dtype=data.dtype,
        )
        for center_slice in range(num_slices):
            stacked[:, center_slice] = np.concatenate(
                [data[:, slice_idx] for slice_idx in self._get_slice_indices(center_slice, num_slices)],
                axis=0,
            )
        return stacked

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Reuse the stock nnU-Net validation/export machinery, but first stack the
        validation case into the 2.5D channel layout expected by this trainer.
        """
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, "validation")
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // torch.distributed.get_world_size() - 1
                val_keys = val_keys[self.local_rank :: torch.distributed.get_world_size()]
            else:
                last_barrier_at_idx = None

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, "predicted_next_stage", n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        segmentation_export_pool, worker_list, results, allowed_num_queued=2
                    )

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack(
                        (data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels, output_dtype=data.dtype))
                    )

                data = self._stack_case_for_inference(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f"{k}, shape {data.shape}, rank {self.local_rank}")
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data).cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits,
                        (
                            (
                                prediction,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                output_filename_truncated,
                                save_probabilities,
                            ),
                        ),
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(
                            nnUNet_preprocessed,
                            self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier,
                        )
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                "Run the preprocessing for this configuration first!"
                            )
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, "predicted_next_stage", n)
                        output_file_truncated = join(output_folder, k)

                        results.append(
                            segmentation_export_pool.starmap_async(
                                resample_and_save,
                                (
                                    (
                                        prediction,
                                        target_shape,
                                        output_file_truncated,
                                        self.plans_manager,
                                        self.configuration_manager,
                                        properties,
                                        self.dataset_json,
                                        default_num_processes,
                                        dataset_class,
                                    ),
                                ),
                            )
                        )

                if self.is_ddp and last_barrier_at_idx is not None and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    torch.distributed.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            torch.distributed.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                validation_output_folder,
                join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
                num_processes=default_num_processes * torch.distributed.get_world_size() if self.is_ddp else default_num_processes,
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", metrics["foreground_mean"]["Dice"], also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        empty_cache(self.device)
        compute_gaussian.cache_clear()

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

        arch_class_name, arch_init_kwargs, arch_init_kwargs_req_import = self._resolve_architecture_definition()
        self.network = get_network_from_plans(
            arch_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=self.enable_deep_supervision,
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
