from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


EXPERIMENTS = (
    ("A0", "standard_25d", "nnUNetTrainer_25D_HarmonizedMin300Patience100"),
    ("A1", "adapter_control", "nnUNetTrainer_25D_AdapterControl"),
    ("A2", "csam_slice_gate", "nnUNetTrainer_25D_CSAMSliceGate"),
    ("A3", "eca_slice_gate", "nnUNetTrainer_25D_ECASliceGate"),
    ("A4", "pixelwise_cross_slice", "nnUNetTrainer_25D_PixelWiseCrossSlice"),
    ("A5", "csa_center_neighbor", "nnUNetTrainer_25D_CSACenterNeighbor"),
    ("A6", "cbam", "nnUNetTrainer_25D_CBAM"),
    ("A7", "coordinate_attention", "nnUNetTrainer_25D_CoordinateAttention"),
    ("A8", "axial_slice_conv", "nnUNetTrainer_25D_AxialSliceConv"),
)


def dice_fields(summary: dict) -> dict[str, float | str]:
    row: dict[str, float | str] = {"foreground_mean_dice": summary["foreground_mean"]["Dice"]}
    for label, metrics in summary.get("mean", {}).items():
        if str(label) != "0" and isinstance(metrics, dict) and "Dice" in metrics:
            row[f"dice_label_{label}"] = metrics["Dice"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect A0-A8 fold-0 nnU-Net summary Dice values")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ["nnUNet_results"]) if "nnUNet_results" in os.environ else None,
    )
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    args = parser.parse_args()
    if args.results_root is None:
        parser.error("set nnUNet_results or pass --results-root")

    rows = []
    for experiment_id, method, trainer in EXPERIMENTS:
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
            "status": "missing",
            "summary_path": str(summary_path),
        }
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                row.update(dice_fields(json.load(handle)))
            row["status"] = "complete"
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
