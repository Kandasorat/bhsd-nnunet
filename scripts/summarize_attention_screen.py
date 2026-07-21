from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


EXPERIMENTS = (
    ("A0", "standard_25d", "nnUNetTrainer_25D_HarmonizedMin300Patience100", "baseline_25d_3slide_harmonized_min300_patience100_fold0"),
    ("A1", "adapter_control", "nnUNetTrainer_25D_AdapterControl", "screen_25d_a1_adapter_control_multiclass_fold0"),
    ("A2", "csam_slice_gate", "nnUNetTrainer_25D_CSAMSliceGate", "screen_25d_a2_csam_slice_gate_multiclass_fold0"),
    ("A3", "eca_slice_gate", "nnUNetTrainer_25D_ECASliceGate", "screen_25d_a3_eca_slice_gate_multiclass_fold0"),
    ("A4", "pixelwise_cross_slice", "nnUNetTrainer_25D_PixelWiseCrossSlice", "screen_25d_a4_pixelwise_cross_slice_multiclass_fold0"),
    ("A5", "csa_center_neighbor", "nnUNetTrainer_25D_CSACenterNeighbor", "screen_25d_a5_csa_center_neighbor_multiclass_fold0"),
    ("A6", "cbam", "nnUNetTrainer_25D_CBAM", "screen_25d_a6_cbam_multiclass_fold0"),
    ("A7", "coordinate_attention", "nnUNetTrainer_25D_CoordinateAttention", "screen_25d_a7_coordinate_attention_multiclass_fold0"),
    ("A8", "axial_slice_conv", "nnUNetTrainer_25D_AxialSliceConv", "screen_25d_a8_axial_slice_conv_multiclass_fold0"),
)


def dice_fields(summary: dict) -> dict[str, float | str]:
    row: dict[str, float | str] = {"foreground_mean_dice": summary["foreground_mean"]["Dice"]}
    for label, metrics in summary.get("mean", {}).items():
        if str(label) != "0" and isinstance(metrics, dict) and "Dice" in metrics:
            row[f"dice_label_{label}"] = metrics["Dice"]
    return row


def timing_fields(metadata_root: Path | None, experiment_name: str) -> dict[str, float | int | str]:
    if metadata_root is None:
        return {}
    metrics_path = metadata_root / experiment_name / "stage_metrics.csv"
    if not metrics_path.is_file():
        return {"timing_status": "missing", "timing_path": str(metrics_path)}

    with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
        attempts = [row for row in csv.DictReader(handle) if row.get("stage") == "train_fold_0"]
    if not attempts:
        return {"timing_status": "missing_train_fold_0", "timing_path": str(metrics_path)}

    durations = [float(row["duration_seconds"]) for row in attempts if row.get("duration_seconds")]
    memory = [float(row["max_gpu_memory_used_mb"]) for row in attempts if row.get("max_gpu_memory_used_mb")]
    utilization = [float(row["mean_gpu_utilization_pct"]) for row in attempts if row.get("mean_gpu_utilization_pct")]
    exit_codes = [row.get("exit_code", "") for row in attempts]
    return {
        "timing_status": "complete",
        "timing_attempts": len(attempts),
        "runner_duration_seconds": round(sum(durations), 3),
        "runner_duration_hours": round(sum(durations) / 3600.0, 6),
        "start_time_utc": attempts[0].get("start_time_utc", ""),
        "end_time_utc": attempts[-1].get("end_time_utc", ""),
        "exit_codes": ";".join(exit_codes),
        "mean_gpu_utilization_pct": round(sum(utilization) / len(utilization), 3) if utilization else "",
        "max_gpu_memory_used_mb": round(max(memory), 3) if memory else "",
        "timing_path": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect A0-A8 fold-0 nnU-Net summary Dice values")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ["nnUNet_results"]) if "nnUNet_results" in os.environ else None,
    )
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(os.environ["BHSD_RESULTS_DIR"]) if "BHSD_RESULTS_DIR" in os.environ else None,
        help="experiment metadata root containing per-experiment stage_metrics.csv",
    )
    args = parser.parse_args()
    if args.results_root is None:
        parser.error("set nnUNet_results or pass --results-root")

    rows = []
    for experiment_id, method, trainer, experiment_name in EXPERIMENTS:
        summary_path = (
            args.results_root
            / "Dataset001_BHSD"
            / f"{trainer}__nnUNetPlans__2d"
            / "fold_0"
            / "validation"
            / "summary.json"
        )
        row: dict[str, float | str] = {
            "id": experiment_id,
            "method": method,
            "trainer": trainer,
            "experiment_name": experiment_name,
            "status": "missing",
            "summary_path": str(summary_path),
        }
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                row.update(dice_fields(json.load(handle)))
            row["status"] = "complete"
        row.update(timing_fields(args.metadata_root, experiment_name))
        rows.append(row)

    fieldnames = list(rows[0])
    extra_fields = sorted({key for row in rows for key in row}.difference(fieldnames))
    fieldnames.extend(extra_fields)
    output_handle = args.output.open("w", newline="", encoding="utf-8-sig") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output_handle.close()


if __name__ == "__main__":
    main()
