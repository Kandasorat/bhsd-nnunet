from __future__ import annotations

import os

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from nnunet25d.common.early_stopping import BHSDEarlyStoppingMixin
from nnunet25d.csam.volume_dataloader import nnUNetDataLoaderCSAMVolume
from nnunet25d.csam.volume_wrapper import OfficialCSAMVolumeWrapper


class nnUNetTrainerCSAMVolumeOfficial(BHSDEarlyStoppingMixin, nnUNetTrainer):
    """Volume-context CSAM using ordered 32-slice windows and 256x256 patches."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["patch_size"] = [256, 256]
        self.sequence_length = int(os.environ.get("BHSD_CSAM_SEQUENCE_LENGTH", "32"))
        self.enable_deep_supervision = False
        self.initialize_early_stopping()

    def _do_i_compile(self):
        return False

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("Trainer is already initialized")
        base_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        self.num_input_channels = base_channels
        self.network = OfficialCSAMVolumeWrapper(
            input_channels=base_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            sequence_length=self.sequence_length,
        ).to(self.device)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.loss = self._build_loss()
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        self.was_initialized = True
        empty_cache(self.device)

    def get_dataloaders(self):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        patch = self.configuration_manager.patch_size
        common = dict(
            batch_size=1,
            patch_size=patch,
            final_patch_size=patch,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=True,
            transforms=None,
            sequence_length=self.sequence_length,
        )
        dl_tr = nnUNetDataLoaderCSAMVolume(dataset_tr, training=True, **common)
        dl_val = nnUNetDataLoaderCSAMVolume(dataset_val, training=False, **common)
        train = SingleThreadedAugmenter(dl_tr, None)
        val = SingleThreadedAugmenter(dl_val, None)
        _ = next(train)
        _ = next(val)
        return train, val

    @staticmethod
    def _starts(size: int, patch: int) -> list[int]:
        if size <= patch:
            return [0]
        stride = max(1, patch // 2)
        starts = list(range(0, size - patch + 1, stride))
        if starts[-1] != size - patch:
            starts.append(size - patch)
        return starts

    def _predict_volume_logits(self, data: np.ndarray) -> torch.Tensor:
        if data.ndim != 4:
            raise ValueError(f"Expected [C, Z, Y, X], got {data.shape}")
        _, z_size, height, width = data.shape
        patch_h, patch_w = (int(v) for v in self.configuration_manager.patch_size)
        z_starts = nnUNetDataLoaderCSAMVolume._window_starts(z_size, self.sequence_length)
        y_starts = self._starts(height, patch_h)
        x_starts = self._starts(width, patch_w)
        num_classes = self.label_manager.num_segmentation_heads
        logits_sum = np.zeros((num_classes, z_size, height, width), dtype=np.float32)
        counts = np.zeros((z_size, height, width), dtype=np.float32)

        self.network.eval()
        with torch.no_grad():
            for z_start in z_starts:
                z_indices = np.clip(
                    np.arange(z_start, z_start + self.sequence_length), 0, z_size - 1
                ).astype(int)
                for y_start in y_starts:
                    for x_start in x_starts:
                        bbox = [
                            [0, self.sequence_length],
                            [y_start, y_start + patch_h],
                            [x_start, x_start + patch_w],
                        ]
                        tile = crop_and_pad_nd(data[:, z_indices], bbox, 0).transpose(1, 0, 2, 3)
                        tile_tensor = torch.from_numpy(np.ascontiguousarray(tile)).float().to(self.device)
                        with torch.autocast(device_type="cuda", enabled=True) if self.device.type == "cuda" else torch.no_grad():
                            tile_logits = self.network(tile_tensor)
                        tile_logits = tile_logits.float().cpu().numpy()
                        y_end = min(y_start + patch_h, height)
                        x_end = min(x_start + patch_w, width)
                        valid_h = y_end - y_start
                        valid_w = x_end - x_start
                        for local_z, global_z in enumerate(z_indices):
                            logits_sum[:, global_z, y_start:y_end, x_start:x_end] += tile_logits[
                                local_z, :, :valid_h, :valid_w
                            ]
                            counts[global_z, y_start:y_end, x_start:x_end] += 1

        logits_sum /= np.maximum(counts[None], 1.0)
        return torch.from_numpy(logits_sum)

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.network.eval()
        validation_output_folder = join(self.output_folder, "validation")
        maybe_mkdir_p(validation_output_folder)
        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )

        for key in dataset_val.identifiers:
            self.print_to_log_file(f"predicting {key} with volume-wise CSAM")
            data, _, _, properties = dataset_val.load_case(key)
            prediction = self._predict_volume_logits(data[:])
            export_prediction_from_logits(
                prediction,
                properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                join(validation_output_folder, key),
                save_probabilities,
            )

        metrics = compute_metrics_on_folder(
            join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
            validation_output_folder,
            join(validation_output_folder, "summary.json"),
            self.plans_manager.image_reader_writer_class(),
            self.dataset_json["file_ending"],
            self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else self.label_manager.foreground_labels,
            self.label_manager.ignore_label,
            chill=True,
            num_processes=default_num_processes,
        )
        self.print_to_log_file("Validation complete", also_print_to_console=True)
        self.print_to_log_file(
            "Mean Validation Dice: ", metrics["foreground_mean"]["Dice"], also_print_to_console=True
        )
        empty_cache(self.device)


__all__ = ["nnUNetTrainerCSAMVolumeOfficial"]
