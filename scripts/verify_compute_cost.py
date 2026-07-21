from __future__ import annotations

import sys
from pathlib import Path

from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_25d_compute import parameter_counts  # noqa: E402
from scripts.run_experiment import NvidiaSmiMonitor  # noqa: E402
from scripts.summarize_controlled_screens import (  # noqa: E402
    CONFIG_DIR,
    SCREEN_PATTERNS,
    add_cost_assessment,
    experiment_id,
)


class _WrappedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.adapter = nn.Linear(4, 2)


def main() -> None:
    model = _WrappedNetwork()
    total, trainable, adapter = parameter_counts(model)
    if (total, trainable, adapter) != (26, 26, 10):
        raise AssertionError(f"Unexpected parameter accounting: {(total, trainable, adapter)}")

    import os

    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-test-uuid,3"
        if NvidiaSmiMonitor(0).gpu_selector != "GPU-test-uuid":
            raise AssertionError("GPU UUID selection does not respect CUDA_VISIBLE_DEVICES")
        if NvidiaSmiMonitor(1).gpu_selector != "3":
            raise AssertionError("GPU index selection does not respect CUDA_VISIBLE_DEVICES")
    finally:
        if previous_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices

    for screen, expected_ids in {
        "fusion": ["C0", "C1", "C2", "C3", "F1", "F2"],
        "spectral": ["D0", "D1", "D2", "D3", "D4", "D5", "D6"],
    }.items():
        ids = [experiment_id(path) for path in sorted(CONFIG_DIR.glob(SCREEN_PATTERNS[screen]))]
        if ids != expected_ids:
            raise AssertionError(f"Unexpected {screen} mapping: {ids}")

    rows = [
        {
            "id": "D0",
            "status": "complete",
            "foreground_mean_dice": 0.30,
            "dice_label_1": 0.30,
            "runner_duration_hours": 1.0,
            "training_max_gpu_memory_mb": 1000,
            "total_parameters": 100,
        },
        {
            "id": "D1",
            "status": "complete",
            "foreground_mean_dice": 0.32,
            "dice_label_1": 0.31,
            "runner_duration_hours": 1.1,
            "training_max_gpu_memory_mb": 1050,
            "total_parameters": 105,
        },
        {
            "id": "D2",
            "status": "complete",
            "foreground_mean_dice": 0.315,
            "dice_label_1": 0.305,
            "runner_duration_hours": 1.3,
            "training_max_gpu_memory_mb": 1200,
            "total_parameters": 120,
        },
        {
            "id": "D3",
            "status": "complete",
            "foreground_mean_dice": 0.325,
            "dice_label_1": 0.26,
            "runner_duration_hours": 2.0,
            "training_max_gpu_memory_mb": 1500,
            "total_parameters": 150,
        },
    ]
    add_cost_assessment(rows, "D0", material_gain=0.01, class_regression=0.02)
    statuses = {row["id"]: row["cost_aware_status"] for row in rows}
    expected = {
        "D0": "control",
        "D1": "candidate_for_multiseed_confirmation",
        "D2": "pareto_dominated",
        "D3": "major_class_regression",
    }
    if statuses != expected:
        raise AssertionError(f"Unexpected cost-aware decisions: {statuses}")

    missing_cost_rows = [dict(row) for row in rows[:2]]
    missing_cost_rows[1].pop("runner_duration_hours")
    add_cost_assessment(missing_cost_rows, "D0", material_gain=0.01, class_regression=0.02)
    if missing_cost_rows[1]["cost_aware_status"] != "missing_compute_cost":
        raise AssertionError(f"Missing cost must block advancement: {missing_cost_rows[1]}")

    fusion_rows = [
        {
            "id": arm_id,
            "status": "complete",
            "foreground_mean_dice": dice,
            "dice_label_1": dice,
            "runner_duration_hours": duration,
            "training_max_gpu_memory_mb": memory,
            "total_parameters": parameters,
        }
        for arm_id, dice, duration, memory, parameters in (
            ("C0", 0.30, 1.0, 1000, 100),
            ("C3", 0.32, 1.1, 1050, 105),
            ("F1", 0.325, 1.3, 1100, 110),
            ("F2", 0.335, 1.4, 1150, 115),
        )
    ]
    add_cost_assessment(
        fusion_rows,
        "C0",
        material_gain=0.01,
        class_regression=0.02,
        advancement_references={"F1": "C3", "F2": "C3"},
    )
    fusion_statuses = {row["id"]: row["cost_aware_status"] for row in fusion_rows}
    if fusion_statuses["F1"] != "insufficient_dice_gain":
        raise AssertionError(f"F1 must be compared with C3: {fusion_statuses}")
    if fusion_statuses["F2"] != "candidate_for_multiseed_confirmation":
        raise AssertionError(f"F2 should pass the C3-aware rule: {fusion_statuses}")

    print("Compute-cost verification passed.")
    print(f"Validated parameter accounting, arm mappings, Pareto and advancement rules: {statuses}")
    print(f"Validated C3-aware fusion advancement: {fusion_statuses}")


if __name__ == "__main__":
    main()
