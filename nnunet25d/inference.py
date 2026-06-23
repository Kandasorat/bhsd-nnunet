from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


class StackedSlicePredictor(nnUNetPredictor):
    def __init__(self, *args, num_input_slices: int = 3, **kwargs):
        if num_input_slices < 1 or num_input_slices % 2 != 1:
            raise ValueError(f"num_input_slices must be a positive odd integer, got {num_input_slices}")
        super().__init__(*args, **kwargs)
        self.num_input_slices = int(num_input_slices)

    def _get_slice_indices(self, center_slice: int, num_slices: int) -> List[int]:
        half = self.num_input_slices // 2
        return [min(max(center_slice + offset, 0), num_slices - 1) for offset in range(-half, half + 1)]

    def _stack_case_for_inference(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 4:
            raise RuntimeError(
                "StackedSlicePredictor expects preprocessed data with shape (C, Z, Y, X), "
                f"got {data.shape}"
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

    @torch.inference_mode()
    def predict_from_files_sequential_stacked(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        folder_with_segs_from_prev_stage: str | None = None,
    ):
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = (
            self._manage_input_and_output_lists(
                list_of_lists_or_source_folder,
                output_folder_or_list_of_truncated_output_files,
                folder_with_segs_from_prev_stage,
                overwrite,
                0,
                1,
                save_probabilities,
            )
        )
        if len(list_of_lists_or_source_folder) == 0:
            return []

        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
        ret = []

        try:
            for input_files, output_file, prev_stage_file in zip(
                list_of_lists_or_source_folder,
                output_filename_truncated,
                seg_from_prev_stage_files,
            ):
                data, _, data_properties = preprocessor.run_case(
                    input_files,
                    prev_stage_file,
                    self.plans_manager,
                    self.configuration_manager,
                    self.dataset_json,
                )
                stacked = self._stack_case_for_inference(data)
                prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(stacked)).cpu()

                if output_file is not None:
                    export_prediction_from_logits(
                        prediction,
                        data_properties,
                        self.configuration_manager,
                        self.plans_manager,
                        self.dataset_json,
                        output_file,
                        save_probabilities,
                    )
                else:
                    ret.append(
                        convert_predicted_logits_to_segmentation_with_correct_shape(
                            prediction,
                            self.plans_manager,
                            self.configuration_manager,
                            self.label_manager,
                            data_properties,
                            save_probabilities,
                        )
                    )
        finally:
            compute_gaussian.cache_clear()
            empty_cache(self.device)

        return ret
