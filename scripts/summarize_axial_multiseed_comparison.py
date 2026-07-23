from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_symmetric_multiseed import (
    describe,
    ground_truth_present_case_class_macro,
    load_config,
    model_specific_case_class_macro,
    summary_path,
    write_csv,
)


ARM_CONFIGS = {
    3407: {
        "E0": "symmetric_25d_e0_control_fold0",
        "E2": "symmetric_25d_e2_reliability_gate_fold0",
        "A8_controlled": "fusion_25d_c3_axial_fold0",
    },
    1234: {
        "E0": "symmetric_25d_e0_control_fold0_seed1234",
        "E2": "symmetric_25d_e2_reliability_gate_fold0_seed1234",
        "A8_controlled": "axial_25d_a8_controlled_fold0_seed1234",
    },
    5678: {
        "E0": "symmetric_25d_e0_control_fold0_seed5678",
        "E2": "symmetric_25d_e2_reliability_gate_fold0_seed5678",
        "A8_controlled": "axial_25d_a8_controlled_fold0_seed5678",
    },
}


def arm_key(arm: str) -> str:
    return arm.lower()


def load_seed_row(seed: int, results_root: Path) -> dict:
    row: dict = {"seed": seed, "data_seed": 1003410, "status": "complete"}
    summaries: dict[str, dict] = {}
    for arm, config_name in ARM_CONFIGS[seed].items():
        key = arm_key(arm)
        config = load_config(config_name)
        path = summary_path(results_root, config)
        row[f"{key}_config"] = config_name
        row[f"{key}_trainer"] = config["trainer"]
        row[f"{key}_summary_path"] = str(path)
        if not path.is_file():
            row["status"] = "missing"
            row[f"{key}_missing"] = True
            continue
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))

    if len(summaries) != len(ARM_CONFIGS[seed]):
        return row

    for arm, summary in summaries.items():
        key = arm_key(arm)
        row[f"nnunet_foreground_mean_dice_{key}"] = float(summary["foreground_mean"]["Dice"])
        row[f"model_specific_case_class_macro_dice_{key}"] = model_specific_case_class_macro(summary)
        row[f"ground_truth_present_case_class_macro_dice_{key}"] = (
            ground_truth_present_case_class_macro(summary)
        )
        for class_id in range(1, 6):
            row[f"nnunet_class_{class_id}_dice_{key}"] = float(summary["mean"][str(class_id)]["Dice"])

    comparisons = (
        ("e2", "e0", "e2_minus_e0"),
        ("a8_controlled", "e0", "a8_controlled_minus_e0"),
        ("e2", "a8_controlled", "e2_minus_a8_controlled"),
    )
    for candidate, reference, label in comparisons:
        for metric in (
            "nnunet_foreground_mean_dice",
            "model_specific_case_class_macro_dice",
            "ground_truth_present_case_class_macro_dice",
        ):
            row[f"{metric}_delta_{label}"] = row[f"{metric}_{candidate}"] - row[f"{metric}_{reference}"]
        for class_id in range(1, 6):
            row[f"nnunet_class_{class_id}_delta_{label}"] = (
                row[f"nnunet_class_{class_id}_dice_{candidate}"]
                - row[f"nnunet_class_{class_id}_dice_{reference}"]
            )
    return row


def deltas(rows: list[dict], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row["status"] == "complete"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare controlled A8, E0, and E2 across the pre-specified fold-0 model seeds."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ["nnUNet_results"]) if "nnUNet_results" in os.environ else None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.results_root is None:
        parser.error("set nnUNet_results or pass --results-root")

    rows = [load_seed_row(seed, args.results_root) for seed in ARM_CONFIGS]
    missing = [row["seed"] for row in rows if row["status"] != "complete"]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"Missing E0/E2/controlled-A8 summary triplets for model seeds: {missing}")

    complete = [row for row in rows if row["status"] == "complete"]
    aggregate = {
        "fold": 0,
        "pre_specified_model_seeds": list(ARM_CONFIGS),
        "fixed_data_seed": 1003410,
        "complete_model_seeds": [row["seed"] for row in complete],
        "missing_model_seeds": missing,
        "controlled_a8_seed_3407_source": (
            "completed C3 controlled replication; the original A8 screen result is not reused"
        ),
        "primary_model_comparison": {
            "name": "nnUNet validation/summary.json foreground_mean.Dice paired E2-minus-controlled-A8",
            **describe(deltas(complete, "nnunet_foreground_mean_dice_delta_e2_minus_a8_controlled")),
        },
        "e2_mechanism_confirmation_reference": {
            "name": "nnUNet validation/summary.json foreground_mean.Dice paired E2-minus-E0",
            **describe(deltas(complete, "nnunet_foreground_mean_dice_delta_e2_minus_e0")),
        },
        "secondary_model_specific_macro_comparison": {
            "name": "model-specific finite-class per-case macro Dice paired E2-minus-controlled-A8",
            **describe(
                deltas(complete, "model_specific_case_class_macro_dice_delta_e2_minus_a8_controlled")
            ),
        },
        "secondary_ground_truth_present_macro_comparison": {
            "name": "ground-truth-present class per-case macro Dice paired E2-minus-controlled-A8",
            **describe(
                deltas(
                    complete,
                    "ground_truth_present_case_class_macro_dice_delta_e2_minus_a8_controlled",
                )
            ),
        },
        "metric_separation": {
            "online_ema_dice": "checkpoint-selection signal only; not read or aggregated",
            "nnunet_summary_dice": "primary direct model-performance comparison",
            "model_specific_case_class_macro_dice": (
                "secondary endpoint with model-specific finite supports; not interchangeable with the primary"
            ),
            "ground_truth_present_case_class_macro_dice": (
                "secondary common-support endpoint restricted to ground-truth-present classes"
            ),
        },
        "interpretation_limit": (
            "The A8-versus-E2 comparison evaluates model performance, not a matched mechanism effect. "
            "Three model seeds on fold 0 do not replace folds 1-4."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "controlled_a8_e0_e2_multiseed_per_seed.csv"
    json_path = args.output_dir / "controlled_a8_e0_e2_multiseed_aggregate.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} per-seed rows to {csv_path}")
    print(f"Wrote paired aggregate to {json_path}")
    print(f"Complete seeds: {len(complete)}; missing seeds: {missing or 'none'}")


if __name__ == "__main__":
    main()
