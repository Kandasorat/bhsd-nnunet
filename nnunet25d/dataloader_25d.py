from typing import List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDataLoader25D(nnUNetDataLoader):
    def __init__(self, *args, num_input_slices: int = 3, **kwargs):
        if num_input_slices < 3 or num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be an odd integer >= 3")
        self.num_input_slices = num_input_slices
        self.slice_offsets = tuple(range(-(num_input_slices // 2), num_input_slices // 2 + 1))
        super().__init__(*args, **kwargs)
        if not self.patch_size_was_2d:
            raise RuntimeError("nnUNetDataLoader25D only supports 2D nnU-Net configurations")

    def determine_shapes(self):
        data, seg, seg_prev, _ = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0] * self.num_input_slices

        data_shape = (self.batch_size, num_color_channels, *self.patch_size[1:])
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size[1:])
        return data_shape, seg_shape

    def _get_slice_indices(self, center_slice: int, num_slices: int) -> List[int]:
        return [min(max(center_slice + offset, 0), num_slices - 1) for offset in self.slice_offsets]

    @staticmethod
    def _crop_2d_slice(array_2d_or_chw: np.ndarray, bbox_lbs_2d: List[int], bbox_ubs_2d: List[int], pad_value: int):
        bbox_2d = [[i, j] for i, j in zip(bbox_lbs_2d, bbox_ubs_2d)]
        return crop_and_pad_nd(array_2d_or_chw, bbox_2d, pad_value)

    def _stack_input_slices(self, data: np.ndarray, center_slice: int, bbox_lbs_2d: List[int], bbox_ubs_2d: List[int]):
        stacked_slices = []
        num_slices = data.shape[1]
        for slice_idx in self._get_slice_indices(center_slice, num_slices):
            slice_data = data[:, slice_idx]
            stacked_slices.append(self._crop_2d_slice(slice_data, bbox_lbs_2d, bbox_ubs_2d, 0))
        return np.concatenate(stacked_slices, axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, identifier in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(identifier)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            center_slice = min(max(bbox_lbs[0], 0), shape[0] - 1)
            bbox_lbs_2d = bbox_lbs[1:]
            bbox_ubs_2d = bbox_ubs[1:]

            data_all[j] = self._stack_input_slices(data, center_slice, bbox_lbs_2d, bbox_ubs_2d)

            center_seg = self._crop_2d_slice(seg[:, center_slice], bbox_lbs_2d, bbox_ubs_2d, -1)
            if seg_prev is not None:
                center_prev = self._crop_2d_slice(seg_prev[None, center_slice], bbox_lbs_2d, bbox_ubs_2d, -1)
                center_seg = np.vstack((center_seg, center_prev))
            seg_all[j] = center_seg

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {"data": data_all, "target": seg_all, "keys": selected_keys}

        return {"data": data_all, "target": seg_all, "keys": selected_keys}
