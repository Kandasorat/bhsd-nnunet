from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SEED_CONFIGS = {
    3407: {
        "E0": "symmetric_25d_e0_control_fold0",
        "E2": "symmetric_25d_e2_reliability_gate_fold0",
    },
    1234: {
        "E0": "symmetric_25d_e0_control_fold0_seed1234",
        "E2": "symmetric_25d_e2_reliability_gate_fold0_seed1234",
    },
    5678: {
        "E0": "symmetric_25d_e0_control_fold0_seed5678",
        "E2": "symmetric_25d_e2_reliability_gate_fold0_seed5678",
    },
}


def load_config(name: str) -> dict:
    payload = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {name}")
    return payload


def summary_path(results_root: Path, config: dict) -> Path:
    return (
        results_root
        / str(config["dataset_name"])
        / f"{config['trainer']}__{config.get('plans', 'nnUNetPlans')}__{config['configuration']}"
        / "fold_0"
        / "validation"
        / "summary.json"
    )


def model_specific_case_class_macro(summary: dict) -> float:
    case_means = []
    for case in summary.get("metric_per_case", []):
        values = [
            float(metrics["Dice"])
            for metrics in case.get("metrics", {}).values()
            if "Dice" in metrics and math.isfinite(float(metrics["Dice"]))
        ]
        if values:
            case_means.append(fmean(values))
    if not case_means:
        raise ValueError("summary.json contains no finite per-case/class Dice values")
    return fmean(case_means)


def ground_truth_present_case_class_macro(summary: dict) -> float:
    case_means = []
    for case in summary.get("metric_per_case", []):
        values = [
            float(metrics["Dice"])
            for metrics in case.get("metrics", {}).values()
            if int(metrics.get("n_ref", 0)) > 0 and math.isfinite(float(metrics["Dice"]))
        ]
        if values:
            case_means.append(fmean(values))
    if not case_means:
        raise ValueError("summary.json contains no ground-truth-present per-case/class Dice values")
    return fmean(case_means)


def load_seed_row(seed: int, results_root: Path) -> dict:
    row: dict = {"seed": seed, "data_seed": 1003410, "status": "complete"}
    summaries: dict[str, dict] = {}
    for arm in ("E0", "E2"):
        config_name = SEED_CONFIGS[seed][arm]
        config = load_config(config_name)
        path = summary_path(results_root, config)
        row[f"{arm.lower()}_config"] = config_name
        row[f"{arm.lower()}_trainer"] = config["trainer"]
        row[f"{arm.lower()}_summary_path"] = str(path)
        if not path.is_file():
            row["status"] = "missing"
            row[f"{arm.lower()}_missing"] = True
            continue
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))

    if len(summaries) != 2:
        return row

    for arm, summary in summaries.items():
        key = arm.lower()
        row[f"nnunet_foreground_mean_dice_{key}"] = float(summary["foreground_mean"]["Dice"])
        row[f"model_specific_case_class_macro_dice_{key}"] = model_specific_case_class_macro(summary)
        row[f"ground_truth_present_case_class_macro_dice_{key}"] = (
            ground_truth_present_case_class_macro(summary)
        )
        for class_id in range(1, 6):
            row[f"nnunet_class_{class_id}_dice_{key}"] = float(summary["mean"][str(class_id)]["Dice"])

    row["nnunet_foreground_mean_delta_e2_minus_e0"] = (
        row["nnunet_foreground_mean_dice_e2"] - row["nnunet_foreground_mean_dice_e0"]
    )
    row["model_specific_case_class_macro_delta_e2_minus_e0"] = (
        row["model_specific_case_class_macro_dice_e2"]
        - row["model_specific_case_class_macro_dice_e0"]
    )
    row["ground_truth_present_case_class_macro_delta_e2_minus_e0"] = (
        row["ground_truth_present_case_class_macro_dice_e2"]
        - row["ground_truth_present_case_class_macro_dice_e0"]
    )
    for class_id in range(1, 6):
        row[f"nnunet_class_{class_id}_delta_e2_minus_e0"] = (
            row[f"nnunet_class_{class_id}_dice_e2"] - row[f"nnunet_class_{class_id}_dice_e0"]
        )
    return row


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    sample_sd = stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": fmean(values),
        "sample_sd": sample_sd,
        "standard_error": sample_sd / math.sqrt(len(values)) if sample_sd is not None else None,
        "minimum": min(values),
        "maximum": max(values),
        "positive_seed_count": sum(value > 0 for value in values),
        "all_seeds_positive": all(value > 0 for value in values),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    fields.extend(sorted({key for row in rows for key in row}.difference(fields)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired E2-minus-E0 effects across the pre-specified fold-0 model seeds."
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

    rows = [load_seed_row(seed, args.results_root) for seed in SEED_CONFIGS]
    missing = [row["seed"] for row in rows if row["status"] != "complete"]
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"Missing E0/E2 summary pairs for model seeds: {missing}")

    complete = [row for row in rows if row["status"] == "complete"]
    primary_deltas = [float(row["nnunet_foreground_mean_delta_e2_minus_e0"]) for row in complete]
    model_specific_macro_deltas = [
        float(row["model_specific_case_class_macro_delta_e2_minus_e0"]) for row in complete
    ]
    present_class_macro_deltas = [
        float(row["ground_truth_present_case_class_macro_delta_e2_minus_e0"])
        for row in complete
    ]
    aggregate = {
        "fold": 0,
        "pre_specified_model_seeds": list(SEED_CONFIGS),
        "fixed_data_seed": 1003410,
        "complete_model_seeds": [row["seed"] for row in complete],
        "missing_model_seeds": missing,
        "primary_endpoint": {
            "name": "nnUNet validation/summary.json foreground_mean.Dice paired E2-minus-E0",
            **describe(primary_deltas),
        },
        "secondary_model_specific_macro_endpoint": {
            "name": "model-specific finite-class per-case macro Dice paired E2-minus-E0",
            **describe(model_specific_macro_deltas),
        },
        "secondary_ground_truth_present_macro_endpoint": {
            "name": "ground-truth-present class per-case macro Dice paired E2-minus-E0",
            **describe(present_class_macro_deltas),
        },
        "metric_separation": {
            "online_ema_dice": "checkpoint-selection signal only; not read or aggregated by this script",
            "nnunet_summary_dice": "primary multi-seed endpoint",
            "model_specific_case_class_macro_dice": "secondary endpoint using each model's finite Dice support; not numerically interchangeable with nnU-Net foreground_mean.Dice",
            "ground_truth_present_case_class_macro_dice": "secondary paired endpoint restricted to classes present in ground truth",
        },
        "interpretation_limit": "Three model seeds on fold 0 estimate initialization sensitivity; they do not replace multi-fold confirmation.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "symmetric_e0_e2_multiseed_per_seed.csv"
    json_path = args.output_dir / "symmetric_e0_e2_multiseed_aggregate.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} per-seed rows to {csv_path}")
    print(f"Wrote paired aggregate to {json_path}")
    print(f"Complete seeds: {len(complete)}; missing seeds: {missing or 'none'}")


if __name__ == "__main__":
    main()
