from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from nnunet25d.dataloader_25d import nnUNetDataLoader25D


class nnUNetDataLoaderSpacingAware25D(nnUNetDataLoader25D):
    def __init__(
        self,
        *args,
        spacing_csv: str | Path | None = None,
        target_context_mm: float = 2.5,
        min_slice_step: int = 1,
        max_slice_step: int = 5,
        **kwargs,
    ):
        self.spacing_csv = Path(spacing_csv) if spacing_csv is not None else None
        self.target_context_mm = float(target_context_mm)
        self.min_slice_step = int(min_slice_step)
        self.max_slice_step = int(max_slice_step)
        self.spacing_map = self._load_spacing_map(self.spacing_csv)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_case_id(case_id: str) -> str:
        return case_id.replace("_0000", "")

    def _load_spacing_map(self, spacing_csv: Path | None) -> Dict[str, float]:
        if spacing_csv is None or not spacing_csv.exists():
            return {}
        mapping: Dict[str, float] = {}
        with spacing_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = self._normalize_case_id(row["case_id"])
                mapping[case_id] = float(row["spacing_z_mm"])
        return mapping

    def _get_case_slice_step(self, identifier: str) -> int:
        z_spacing = self.spacing_map.get(self._normalize_case_id(identifier))
        if z_spacing is None or z_spacing <= 0:
            return 1
        step = round(self.target_context_mm / z_spacing)
        step = max(self.min_slice_step, step)
        step = min(self.max_slice_step, step)
        return int(step)

    def _get_slice_indices_for_case(self, identifier: str, center_slice: int, num_slices: int) -> List[int]:
        step = self._get_case_slice_step(identifier)
        offsets = (-step, 0, step)
        return [min(max(center_slice + offset, 0), num_slices - 1) for offset in offsets]

    def _stack_input_slices(self, data, center_slice, bbox_lbs_2d, bbox_ubs_2d, identifier=None):
        if identifier is None:
            return super()._stack_input_slices(data, center_slice, bbox_lbs_2d, bbox_ubs_2d)
        stacked_slices = []
        num_slices = data.shape[1]
        for slice_idx in self._get_slice_indices_for_case(identifier, center_slice, num_slices):
            slice_data = data[:, slice_idx]
            stacked_slices.append(self._crop_2d_slice(slice_data, bbox_lbs_2d, bbox_ubs_2d, 0))
        return __import__("numpy").concatenate(stacked_slices, axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = __import__("numpy").zeros(self.data_shape, dtype=__import__("numpy").float32)
        seg_all = __import__("numpy").zeros(self.seg_shape, dtype=__import__("numpy").int16)

        for j, identifier in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(identifier)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
            center_slice = min(max(bbox_lbs[0], 0), shape[0] - 1)
            bbox_lbs_2d = bbox_lbs[1:]
            bbox_ubs_2d = bbox_ubs[1:]

            data_all[j] = self._stack_input_slices(data, center_slice, bbox_lbs_2d, bbox_ubs_2d, identifier=identifier)

            center_seg = self._crop_2d_slice(seg[:, center_slice], bbox_lbs_2d, bbox_ubs_2d, -1)
            if seg_prev is not None:
                center_prev = self._crop_2d_slice(seg_prev[None, center_slice], bbox_lbs_2d, bbox_ubs_2d, -1)
                center_seg = __import__("numpy").vstack((center_seg, center_prev))
            seg_all[j] = center_seg

        return self._finalize_batch(data_all, seg_all, selected_keys)

    def _finalize_batch(self, data_all, seg_all, selected_keys):
        import torch
        from threadpoolctl import threadpool_limits

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
