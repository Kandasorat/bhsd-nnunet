from __future__ import annotations

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDataLoaderCSAMVolume(nnUNetDataLoader):
    """Yield one ordered, fixed-length slice sequence from one volume."""

    def __init__(self, *args, sequence_length: int = 32, training: bool = True, **kwargs):
        kwargs["batch_size"] = 1
        kwargs["transforms"] = None
        super().__init__(*args, **kwargs)
        self.sequence_length = int(sequence_length)
        self.training = bool(training)
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be at least 3")
        if not self.patch_size_was_2d:
            raise RuntimeError("CSAM volume mode requires a 2D nnU-Net configuration")
        self._validation_case_cursor = 0
        self._validation_window_cursor = 0

    @staticmethod
    def _window_starts(num_slices: int, length: int) -> list[int]:
        if num_slices <= length:
            return [0]
        stride = max(1, length // 2)
        starts = list(range(0, num_slices - length + 1, stride))
        if starts[-1] != num_slices - length:
            starts.append(num_slices - length)
        return starts

    @staticmethod
    def _spatial_start(size: int, patch: int, center: int | None = None) -> int:
        lower = min(0, size - patch)
        upper = max(0, size - patch)
        if center is None:
            return int(np.random.randint(lower, upper + 1))
        return int(min(max(center - patch // 2, lower), upper))

    @staticmethod
    def _foreground_voxel(class_locations: dict | None):
        if not class_locations:
            return None
        eligible = [value for key, value in class_locations.items() if key != -1 and len(value) > 0]
        if not eligible:
            return None
        locations = eligible[np.random.randint(len(eligible))]
        return locations[np.random.randint(len(locations))]

    def _select_validation_key_and_start(self):
        key = self.indices[self._validation_case_cursor]
        data, _, _, _ = self._data.load_case(key)
        starts = self._window_starts(data.shape[1], self.sequence_length)
        start = starts[self._validation_window_cursor]
        self._validation_window_cursor += 1
        if self._validation_window_cursor >= len(starts):
            self._validation_window_cursor = 0
            self._validation_case_cursor = (self._validation_case_cursor + 1) % len(self.indices)
        return key, start

    def generate_train_batch(self):
        if self.training:
            key = self.get_indices()[0]
            data, seg, seg_prev, properties = self._data.load_case(key)
            force_fg = np.random.random() < self.oversample_foreground_percent
            voxel = self._foreground_voxel(properties.get("class_locations")) if force_fg else None
            starts = self._window_starts(data.shape[1], self.sequence_length)
            if voxel is None:
                z_start = starts[np.random.randint(len(starts))]
                y_center = x_center = None
            else:
                z, y_center, x_center = (int(v) for v in voxel[-3:])
                z_start = min(max(z - self.sequence_length // 2, 0), max(0, data.shape[1] - self.sequence_length))
        else:
            key, z_start = self._select_validation_key_and_start()
            data, seg, seg_prev, properties = self._data.load_case(key)
            y_center = x_center = None

        z_indices = np.clip(
            np.arange(z_start, z_start + self.sequence_length), 0, data.shape[1] - 1
        ).astype(int)
        patch_h, patch_w = (int(v) for v in self.final_patch_size[1:])
        y_start = self._spatial_start(data.shape[2], patch_h, y_center)
        x_start = self._spatial_start(data.shape[3], patch_w, x_center)
        bbox = [[0, self.sequence_length], [y_start, y_start + patch_h], [x_start, x_start + patch_w]]

        selected_data = crop_and_pad_nd(data[:, z_indices], bbox, 0)
        selected_seg = crop_and_pad_nd(seg[:, z_indices], bbox, -1)
        selected_seg[selected_seg < 0] = 0

        # Every geometric transform is shared by all ordered slices.
        if self.training and np.random.random() < 0.5:
            selected_data = selected_data[..., ::-1]
            selected_seg = selected_seg[..., ::-1]
        if self.training and np.random.random() < 0.5:
            selected_data = selected_data[..., ::-1, :]
            selected_seg = selected_seg[..., ::-1, :]

        data_tensor = torch.from_numpy(np.ascontiguousarray(selected_data.transpose(1, 0, 2, 3))).float()
        seg_tensor = torch.from_numpy(np.ascontiguousarray(selected_seg.transpose(1, 0, 2, 3))).to(torch.int16)
        return {"data": data_tensor, "target": seg_tensor, "keys": [key]}
