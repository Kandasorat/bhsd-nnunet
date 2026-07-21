from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SCREEN_PATTERNS = {
    "fusion": "fusion_25d_*_fold0.yaml",
    "spectral": "spectral_25d_*_fold0.yaml",
}
CONTROL_IDS = {"fusion": "C0", "spectral": "D0"}
ADVANCEMENT_REFERENCES = {"fusion": {"F1": "C3", "F2": "C3"}, "spectral": {}}


def experiment_id(path: Path) -> str:
    parts = path.stem.split("_")
    return next(part.upper() for part in parts if len(part) == 2 and part[0] in "cdf" and part[1].isdigit())


def load_rows(screen: str, results_root: Path, metadata_root: Path) -> list[dict]:
    rows = []
    for config_path in sorted(CONFIG_DIR.glob(SCREEN_PATTERNS[screen])):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        arm_id = experiment_id(config_path)
        fold_dir = (
            results_root
            / str(config["dataset_name"])
            / f"{config['trainer']}__{config.get('plans', 'nnUNetPlans')}__{config['configuration']}"
            / "fold_0"
        )
        summary_path = fold_dir / "validation" / "summary.json"
        profile_path = metadata_root / str(config["experiment_name"]) / "compute_profile.json"
        if not profile_path.is_file() and (fold_dir / "compute_profile.json").is_file():
            profile_path = fold_dir / "compute_profile.json"
        metrics_path = metadata_root / str(config["experiment_name"]) / "stage_metrics.csv"
        row = {
            "screen": screen,
            "id": arm_id,
            "experiment_name": config["experiment_name"],
            "trainer": config["trainer"],
            "status": "missing",
            "summary_path": str(summary_path),
            "compute_profile_path": str(profile_path),
            "stage_metrics_path": str(metrics_path),
        }
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row["foreground_mean_dice"] = float(summary["foreground_mean"]["Dice"])
            for label, metrics in summary.get("mean", {}).items():
                if str(label) != "0" and isinstance(metrics, dict) and "Dice" in metrics:
                    row[f"dice_label_{label}"] = float(metrics["Dice"])
            row["status"] = "complete"
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            for key in (
                "total_parameters",
                "adapter_parameters",
                "backbone_passes_per_prediction",
                "forward_latency_median_ms",
                "peak_allocated_memory_mb",
                "incremental_peak_memory_mb",
            ):
                row[key] = profile.get(key, "")
        if metrics_path.is_file():
            with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
                attempts = [item for item in csv.DictReader(handle) if item.get("stage") == "train_fold_0"]
            if attempts:
                row["runner_duration_hours"] = sum(float(item["duration_seconds"]) for item in attempts) / 3600.0
                memory = [float(item["max_gpu_memory_used_mb"]) for item in attempts if item.get("max_gpu_memory_used_mb")]
                row["training_max_gpu_memory_mb"] = max(memory) if memory else ""
                row["training_attempts"] = len(attempts)
                row["training_exit_codes"] = ";".join(item.get("exit_code", "") for item in attempts)
        rows.append(row)
    return rows


def ratio(value, control_value):
    if value in {None, ""} or control_value in {None, "", 0}:
        return ""
    return float(value) / float(control_value)


def add_cost_assessment(
    rows: list[dict],
    control_id: str,
    material_gain: float,
    class_regression: float,
    advancement_references: dict[str, str] | None = None,
) -> None:
    advancement_references = advancement_references or {}
    control = next((row for row in rows if row["id"] == control_id and row["status"] == "complete"), None)
    if control is None:
        for row in rows:
            row["cost_aware_status"] = "waiting_for_control"
        return

    for row in rows:
        row["cost_memory_mb"] = row.get("training_max_gpu_memory_mb") or row.get("peak_allocated_memory_mb", "")

    cost_fields = ("runner_duration_hours", "cost_memory_mb", "total_parameters")
    for row in rows:
        if row["status"] != "complete":
            row["cost_aware_status"] = "incomplete"
            continue
        row["delta_dice_vs_control"] = row["foreground_mean_dice"] - control["foreground_mean_dice"]
        row["runtime_ratio_vs_control"] = ratio(row.get("runner_duration_hours"), control.get("runner_duration_hours"))
        row["memory_ratio_vs_control"] = ratio(row.get("cost_memory_mb"), control.get("cost_memory_mb"))
        row["parameter_ratio_vs_control"] = ratio(row.get("total_parameters"), control.get("total_parameters"))
        label_deltas = [
            row[key] - control[key]
            for key in row
            if key.startswith("dice_label_") and key in control
        ]
        row["worst_class_delta_vs_control"] = min(label_deltas) if label_deltas else ""

        reference_id = advancement_references.get(row["id"], control_id)
        reference = next(
            (candidate for candidate in rows if candidate["id"] == reference_id and candidate["status"] == "complete"),
            None,
        )
        row["advancement_reference"] = reference_id
        if reference is not None:
            row["delta_dice_vs_advancement_reference"] = (
                row["foreground_mean_dice"] - reference["foreground_mean_dice"]
            )
            reference_label_deltas = [
                row[key] - reference[key]
                for key in row
                if key.startswith("dice_label_") and key in reference
            ]
            row["worst_class_delta_vs_advancement_reference"] = (
                min(reference_label_deltas) if reference_label_deltas else ""
            )

    complete = [row for row in rows if row["status"] == "complete"]
    for row in complete:
        row["pareto_efficient"] = True
        for other in complete:
            if other is row or other["foreground_mean_dice"] < row["foreground_mean_dice"]:
                continue
            comparable = [(other.get(field), row.get(field)) for field in cost_fields]
            if not all(a not in {None, ""} and b not in {None, ""} for a, b in comparable):
                continue
            no_more_costly = all(float(a) <= float(b) for a, b in comparable)
            strictly_better = other["foreground_mean_dice"] > row["foreground_mean_dice"] or any(
                float(a) < float(b) for a, b in comparable
            )
            if no_more_costly and strictly_better:
                row["pareto_efficient"] = False
                break

        if row["id"] == control_id:
            row["cost_aware_status"] = "control"
        elif any(control.get(field) in {None, ""} for field in cost_fields):
            row["cost_aware_status"] = "missing_control_compute_cost"
        elif any(row.get(field) in {None, ""} for field in cost_fields):
            row["cost_aware_status"] = "missing_compute_cost"
        elif "delta_dice_vs_advancement_reference" not in row:
            row["cost_aware_status"] = "waiting_for_advancement_reference"
        elif row["delta_dice_vs_advancement_reference"] < material_gain:
            row["cost_aware_status"] = "insufficient_dice_gain"
        elif (
            row["worst_class_delta_vs_advancement_reference"] != ""
            and row["worst_class_delta_vs_advancement_reference"] < -class_regression
        ):
            row["cost_aware_status"] = "major_class_regression"
        elif not row["pareto_efficient"]:
            row["cost_aware_status"] = "pareto_dominated"
        else:
            row["cost_aware_status"] = "candidate_for_multiseed_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Dice and compute cost for controlled 2.5D screens")
    parser.add_argument("--screen", choices=("fusion", "spectral"), required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ["nnUNet_results"]) if "nnUNet_results" in os.environ else None,
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(os.environ["BHSD_RESULTS_DIR"]) if "BHSD_RESULTS_DIR" in os.environ else None,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material-dice-gain", type=float, default=0.01)
    parser.add_argument("--major-class-regression", type=float, default=0.02)
    args = parser.parse_args()
    if args.results_root is None or args.metadata_root is None:
        parser.error("set nnUNet_results/BHSD_RESULTS_DIR or pass both root arguments")

    rows = load_rows(args.screen, args.results_root, args.metadata_root)
    add_cost_assessment(
        rows,
        CONTROL_IDS[args.screen],
        material_gain=args.material_dice_gain,
        class_regression=args.major_class_regression,
        advancement_references=ADVANCEMENT_REFERENCES[args.screen],
    )
    fields = list(rows[0])
    fields.extend(sorted({key for row in rows for key in row}.difference(fields)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cost-aware rows to {args.output}")


if __name__ == "__main__":
    main()
