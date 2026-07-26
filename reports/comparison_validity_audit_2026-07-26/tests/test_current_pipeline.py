from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunet25d.common.dataloader_25d import nnUNetDataLoader25D  # noqa: E402
from nnunet25d.baseline.trainer_25d import _nnUNetTrainer25DBase  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402


class _SyntheticDataset:
    identifiers = ["synthetic_case"]

    def __init__(self, data: np.ndarray, seg: np.ndarray) -> None:
        self.data = data
        self.seg = seg

    def load_case(self, _identifier: str):
        properties = {"class_locations": {1: np.empty((0, 4), dtype=np.int64)}}
        return self.data, self.seg, None, properties


class _LoaderHarness(nnUNetDataLoader25D):
    def __init__(self, data: np.ndarray, seg: np.ndarray, center: int) -> None:
        self.num_input_slices = 3
        self.slice_offsets = (-1, 0, 1)
        self.batch_size = 1
        self._data = _SyntheticDataset(data, seg)
        self.data_shape = (1, 3, data.shape[2], data.shape[3])
        self.seg_shape = (1, 1, seg.shape[2], seg.shape[3])
        self.transforms = None
        self.center = center

    def get_indices(self):
        return ["synthetic_case"]

    def get_do_oversample(self, _sample_idx: int) -> bool:
        return False

    def get_bbox(self, shape, force_fg, class_locations):
        del force_fg, class_locations
        return [self.center, 0, 0], [self.center + 1, shape[1], shape[2]]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _test_stack_and_target() -> dict:
    z_count, height, width = 5, 9, 11
    data = np.stack(
        [np.full((z_count, height, width), z, dtype=np.float32) for z in range(1)],
        axis=0,
    )
    for z in range(z_count):
        data[:, z] = float(z)
    seg = np.zeros((1, z_count, height, width), dtype=np.int16)
    for z in range(z_count):
        seg[:, z] = z

    expected = {
        0: ([0.0, 0.0, 1.0], 0),
        2: ([1.0, 2.0, 3.0], 2),
        4: ([3.0, 4.0, 4.0], 4),
    }
    rows = []
    for center, (channel_values, target_value) in expected.items():
        batch = _LoaderHarness(data, seg, center).generate_train_batch()
        observed_channels = [float(batch["data"][0, c, 0, 0]) for c in range(3)]
        observed_target = int(batch["target"][0, 0, 0, 0])
        _assert(observed_channels == channel_values, f"center {center}: {observed_channels}")
        _assert(observed_target == target_value, f"center target {center}: {observed_target}")
        rows.append(
            {
                "center": center,
                "observed_channels": observed_channels,
                "observed_target": observed_target,
            }
        )

    trainer = object.__new__(_nnUNetTrainer25DBase)
    trainer.num_input_slices = 3
    stacked = trainer._stack_case_for_inference(data)
    _assert(stacked.shape == (3, z_count, height, width), f"inference shape {stacked.shape}")
    for center, (channel_values, _) in expected.items():
        observed = [float(stacked[c, center, 0, 0]) for c in range(3)]
        _assert(observed == channel_values, f"inference center {center}: {observed}")
    return {"status": "PASS", "boundary_and_center_rows": rows}


def _training_transform():
    return nnUNetTrainer.get_training_transforms(
        np.asarray([32, 32]),
        (-0.35, 0.35),
        None,
        (0, 1),
        False,
        [False],
        False,
        [1, 2, 3, 4, 5],
        None,
        None,
    )


def _test_geometry_sync() -> dict:
    pipeline = _training_transform()
    spatial = pipeline.transforms[0]
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    marker = (((yy - 22) ** 2 + (xx - 27) ** 2) <= 45).float()
    image = marker.repeat(3, 1, 1)
    seg = marker.to(torch.int16).unsqueeze(0)
    minimum_alignment_dice = 1.0
    for seed in range(24):
        torch.manual_seed(seed)
        result = spatial(image=image.clone(), segmentation=seg.clone())
        transformed = result["image"]
        target = result["segmentation"]
        _assert(torch.allclose(transformed[0], transformed[1], atol=0, rtol=0), "channel 0/1 geometry differs")
        _assert(torch.allclose(transformed[1], transformed[2], atol=0, rtol=0), "channel 1/2 geometry differs")
        pred_mask = transformed[0] >= 0.5
        target_mask = target[0] >= 0.5
        denom = int(pred_mask.sum() + target_mask.sum())
        dice = 1.0 if denom == 0 else float(2 * torch.logical_and(pred_mask, target_mask).sum() / denom)
        minimum_alignment_dice = min(minimum_alignment_dice, dice)
    _assert(minimum_alignment_dice >= 0.95, f"image/GT alignment Dice={minimum_alignment_dice}")
    return {"status": "PASS", "trials": 24, "minimum_thresholded_alignment_dice": minimum_alignment_dice}


def _test_intensity_behavior() -> dict:
    pipeline = _training_transform()
    settings = []
    for wrapper in pipeline.transforms:
        inner = getattr(wrapper, "transform", wrapper)
        if hasattr(inner, "synchronize_channels"):
            settings.append(
                {
                    "transform": type(inner).__name__,
                    "synchronize_channels": bool(inner.synchronize_channels),
                }
            )

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij")
    base = (yy + 2 * xx + 3).float()
    image = base.repeat(3, 1, 1)
    seg = (base > 3).to(torch.int16).unsqueeze(0)
    differing_trials = 0
    for seed in range(40):
        torch.manual_seed(seed)
        result = pipeline(image=image.clone(), segmentation=seg.clone())
        transformed = result["image"]
        if not (
            torch.allclose(transformed[0], transformed[1], atol=1e-6, rtol=1e-6)
            and torch.allclose(transformed[1], transformed[2], atol=1e-6, rtol=1e-6)
        ):
            differing_trials += 1
    _assert(differing_trials > 0, "full training transform never exposed channel-wise intensity behavior")
    return {
        "status": "PASS_WITH_RECORDED_BEHAVIOR",
        "trials": 40,
        "trials_with_nonidentical_output_channels_from_identical_inputs": differing_trials,
        "intensity_transform_channel_settings": settings,
        "interpretation": "geometry is shared, but most intensity transforms are configured per channel",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "slice_stack_center_target_and_boundaries": _test_stack_and_target(),
        "training_geometry_sync": _test_geometry_sync(),
        "training_intensity_behavior": _test_intensity_behavior(),
    }
    results["overall_status"] = "PASS_WITH_INTENSITY_CAUTION"
    payload = json.dumps(results, indent=2, sort_keys=True)
    (args.output_dir / "current_pipeline_synthetic_results.json").write_text(payload + "\n", encoding="utf-8")
    log_lines = [
        "PASS slice stack: [z-1,z,z+1]",
        "PASS center-slice target",
        "PASS first/last replication boundaries",
        "PASS shared spatial transform for all image channels and GT",
        "CAUTION intensity transforms are mostly channel-independent",
        "OVERALL PASS_WITH_INTENSITY_CAUTION",
    ]
    (args.output_dir / "current_pipeline_synthetic_test.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
