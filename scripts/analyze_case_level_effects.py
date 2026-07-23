from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLASS_NAMES = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}


def case_id_from_path(value: str) -> str:
    name = Path(value).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected MODEL=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty MODEL=PATH")
    resolved = Path(path)
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {resolved}")
    return name, resolved


def rows_from_summary(model: str, path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for case in payload.get("metric_per_case", []):
        case_id = case_id_from_path(case["prediction_file"])
        reference_file = case.get("reference_file")
        for class_id, metrics in case["metrics"].items():
            rows.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "class_id": int(class_id),
                    "dice": float(metrics["Dice"]),
                    "n_ref": int(metrics.get("n_ref", 0)),
                    "n_pred": int(metrics.get("n_pred", 0)),
                    "reference_file": reference_file,
                }
            )
    return rows


def rows_from_csv(model: str, path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"case_id", "class_id", "dice"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["model"] = model
    if "n_ref" not in frame:
        frame["n_ref"] = np.nan
    if "n_pred" not in frame:
        frame["n_pred"] = np.nan
    if "reference_file" not in frame:
        frame["reference_file"] = None
    return frame[
        ["model", "case_id", "class_id", "dice", "n_ref", "n_pred", "reference_file"]
    ].to_dict("records")


def load_metric_rows(inputs: Iterable[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for model, path in inputs:
        rows.extend(rows_from_summary(model, path) if path.suffix.lower() == ".json" else rows_from_csv(model, path))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No case metrics were found")
    frame["class_id"] = frame["class_id"].astype(int)
    frame["class_name"] = frame["class_id"].map(CLASS_NAMES).fillna("unknown")
    return frame


def find_ground_truth(case_id: str, reference_file: str | None, ground_truth_dir: Path | None) -> Path | None:
    if ground_truth_dir is not None:
        candidate = ground_truth_dir / f"{case_id}.nii.gz"
        return candidate if candidate.is_file() else None
    if reference_file:
        candidate = Path(reference_file)
        return candidate if candidate.is_file() else None
    return None


def image_metadata(case_id: str, reference_file: str | None, ground_truth_dir: Path | None) -> dict:
    path = find_ground_truth(case_id, reference_file, ground_truth_dir)
    if path is None:
        return {"case_id": case_id, "spacing_z_mm": np.nan, "voxel_volume_mm3": np.nan}
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    labels = sitk.GetArrayViewFromImage(image)
    spacing = tuple(float(value) for value in image.GetSpacing())
    row = {
        "case_id": case_id,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": float(np.prod(spacing)),
    }
    row.update({f"class_{class_id}_voxels": int(np.count_nonzero(labels == class_id)) for class_id in CLASS_NAMES})
    row["image_lesion_voxels"] = int(np.count_nonzero(labels > 0))
    return row


def add_case_context(long_metrics: pd.DataFrame, ground_truth_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_model = long_metrics["model"].iloc[0]
    reference_rows = long_metrics[long_metrics["model"] == first_model]
    metadata_rows = []
    for case_id, group in reference_rows.groupby("case_id", sort=True):
        reference_file = next((str(v) for v in group["reference_file"] if pd.notna(v) and v), None)
        metadata_rows.append(image_metadata(case_id, reference_file, ground_truth_dir))
    metadata = pd.DataFrame(metadata_rows)

    for class_id in CLASS_NAMES:
        count_column = f"class_{class_id}_voxels"
        if count_column in metadata:
            count_map = metadata.set_index("case_id")[count_column]
            missing = long_metrics["n_ref"].isna() & long_metrics["class_id"].eq(class_id)
            long_metrics.loc[missing, "n_ref"] = long_metrics.loc[missing, "case_id"].map(count_map)

    reference_counts = reference_rows.groupby("case_id", as_index=False)["n_ref"].sum(min_count=1)
    reference_counts = reference_counts.rename(columns={"n_ref": "lesion_voxels"})
    cases = metadata.merge(reference_counts, on="case_id", how="outer")
    if "image_lesion_voxels" in cases:
        cases["lesion_voxels"] = cases["lesion_voxels"].fillna(cases["image_lesion_voxels"])
    cases["lesion_volume_ml"] = cases["lesion_voxels"] * cases["voxel_volume_mm3"] / 1000.0
    cases["spacing_group"] = pd.cut(
        cases["spacing_z_mm"],
        bins=[-np.inf, 3.0, 5.0, np.inf],
        labels=["thin_lt3mm", "intermediate_3to5mm", "thick_ge5mm"],
        right=False,
    ).astype("object")
    cases["lesion_size_group"] = pd.cut(
        cases["lesion_volume_ml"],
        bins=[-np.inf, 1.0, 10.0, np.inf],
        labels=["small_lt1ml", "medium_1to10ml", "large_ge10ml"],
        right=False,
    ).astype("object")
    return long_metrics.merge(cases, on="case_id", how="left"), cases


def case_scores(long_metrics: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    scores = (
        long_metrics.groupby(["case_id", "model"], as_index=False)["dice"]
        .mean()
        .pivot(index="case_id", columns="model", values="dice")
        .reset_index()
    )
    if reference_model not in scores:
        raise ValueError(f"Reference model {reference_model!r} is absent; available: {list(scores.columns[1:])}")
    for model in scores.columns[1:]:
        if model != reference_model:
            scores[f"delta_{model}_vs_{reference_model}"] = scores[model] - scores[reference_model]
    return scores


def present_class_case_scores(long_metrics: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    present = long_metrics[long_metrics["n_ref"] > 0]
    scores = (
        present.groupby(["case_id", "model"], as_index=False)["dice"]
        .mean()
        .pivot(index="case_id", columns="model", values="dice")
        .add_prefix("present_class_macro_")
        .reset_index()
    )
    reference_column = f"present_class_macro_{reference_model}"
    if reference_column not in scores:
        raise ValueError(f"Reference model {reference_model!r} has no ground-truth-present class scores")
    for model in long_metrics["model"].drop_duplicates():
        candidate_column = f"present_class_macro_{model}"
        if model != reference_model and candidate_column in scores:
            scores[f"delta_present_class_macro_{model}_vs_{reference_model}"] = (
                scores[candidate_column] - scores[reference_column]
            )
    return scores


def subgroup_summary(
    case_frame: pd.DataFrame,
    reference_model: str,
    models: list[str],
    *,
    column_prefix: str = "",
    delta_prefix: str = "",
    metric_family: str = "model_specific_case_class_macro",
) -> pd.DataFrame:
    rows = []
    reference_column = f"{column_prefix}{reference_model}"
    for model in models:
        if model == reference_model:
            continue
        candidate_column = f"{column_prefix}{model}"
        delta_column = f"delta_{delta_prefix}{model}_vs_{reference_model}"
        for field in ("spacing_group", "lesion_size_group"):
            for group_name, group in case_frame.groupby(field, dropna=False):
                delta = group[delta_column].dropna()
                rows.append(
                    {
                        "metric_family": metric_family,
                        "candidate": model,
                        "reference": reference_model,
                        "subgroup_variable": field,
                        "subgroup": "missing" if pd.isna(group_name) else str(group_name),
                        "n_cases": int(len(delta)),
                        "reference_mean_dice": float(group[reference_column].mean()),
                        "candidate_mean_dice": float(group[candidate_column].mean()),
                        "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                        "median_delta": float(delta.median()) if len(delta) else np.nan,
                        "improved_fraction": float((delta > 0).mean()) if len(delta) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def class_effect_summary(long_metrics: pd.DataFrame, reference_model: str, models: list[str]) -> pd.DataFrame:
    reference_rows = long_metrics[long_metrics["model"] == reference_model][
        ["case_id", "class_id", "class_name", "n_ref"]
    ]
    present_keys = reference_rows[reference_rows["n_ref"] > 0][
        ["case_id", "class_id", "class_name"]
    ]
    paired = present_keys.merge(
        long_metrics.pivot(
            index=["case_id", "class_id", "class_name"], columns="model", values="dice"
        ).reset_index(),
        on=["case_id", "class_id", "class_name"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for model in models:
        if model == reference_model:
            continue
        for (class_id, class_name), group in paired.groupby(["class_id", "class_name"], sort=True):
            if group[[reference_model, model]].isna().any().any():
                raise ValueError(f"Ground-truth-present class {class_id} has non-finite paired Dice")
            delta = group[model] - group[reference_model]
            rows.append(
                {
                    "candidate": model,
                    "reference": reference_model,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "n_ground_truth_present_cases": int(len(delta)),
                    "paired_reference_mean_dice": float(group[reference_model].mean()),
                    "paired_candidate_mean_dice": float(group[model].mean()),
                    "paired_mean_delta": float(delta.mean()),
                    "paired_median_delta": float(delta.median()),
                    "paired_improved_fraction": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def absent_class_false_positive_summary(
    long_metrics: pd.DataFrame, reference_model: str, models: list[str]
) -> pd.DataFrame:
    reference_rows = long_metrics[long_metrics["model"] == reference_model][
        ["case_id", "class_id", "class_name", "n_ref"]
    ]
    absent_keys = reference_rows[reference_rows["n_ref"] == 0][
        ["case_id", "class_id", "class_name"]
    ]
    predictions = long_metrics.pivot(
        index=["case_id", "class_id", "class_name"], columns="model", values="n_pred"
    ).reset_index()
    absent = absent_keys.merge(
        predictions,
        on=["case_id", "class_id", "class_name"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for model in models:
        if model == reference_model:
            continue
        for (class_id, class_name), group in absent.groupby(["class_id", "class_name"], sort=True):
            if group[[reference_model, model]].isna().any().any():
                raise ValueError(f"Ground-truth-absent class {class_id} has missing n_pred values")
            reference_fp = group[reference_model] > 0
            candidate_fp = group[model] > 0
            rows.append(
                {
                    "candidate": model,
                    "reference": reference_model,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "n_ground_truth_absent_cases": int(len(group)),
                    "reference_false_positive_case_fraction": float(reference_fp.mean()),
                    "candidate_false_positive_case_fraction": float(candidate_fp.mean()),
                    "false_positive_case_fraction_delta": float(candidate_fp.mean() - reference_fp.mean()),
                    "false_positive_cases_resolved": int((reference_fp & ~candidate_fp).sum()),
                    "false_positive_cases_introduced": int((~reference_fp & candidate_fp).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate case-level Dice effects to spacing, lesion size, and class.")
    parser.add_argument("--metrics", action="append", type=parse_named_path, required=True, help="MODEL=summary.json or MODEL=case_metrics.csv; repeat for each model")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    long_metrics = load_metric_rows(args.metrics)
    long_metrics, cases = add_case_context(long_metrics, args.ground_truth_dir)
    scores = case_scores(long_metrics, args.reference_model)
    case_frame = cases.merge(scores, on="case_id", how="right")
    models = [column for column in scores.columns if column != "case_id" and not column.startswith("delta_")]
    present_scores = present_class_case_scores(long_metrics, args.reference_model)
    case_frame = case_frame.merge(present_scores, on="case_id", how="left", validate="one_to_one")

    expected_keys = set(long_metrics[long_metrics["model"] == models[0]][["case_id", "class_id"]].itertuples(index=False, name=None))
    for model in models:
        model_rows = long_metrics[long_metrics["model"] == model]
        if model_rows.duplicated(["case_id", "class_id"]).any():
            raise ValueError(f"Model {model!r} contains duplicate case/class metric rows")
        model_keys = set(model_rows[["case_id", "class_id"]].itertuples(index=False, name=None))
        if model_keys != expected_keys:
            raise ValueError(f"Model {model!r} does not contain the same case/class keys as {models[0]!r}")
    n_ref_disagreement = long_metrics.groupby(["case_id", "class_id"])["n_ref"].nunique(dropna=True) > 1
    if n_ref_disagreement.any():
        raise ValueError("Models disagree on ground-truth voxel counts for one or more case/class pairs")
    if args.ground_truth_dir is not None:
        missing_metadata = cases["spacing_z_mm"].isna()
        if missing_metadata.any():
            missing_cases = cases.loc[missing_metadata, "case_id"].astype(str).tolist()
            raise FileNotFoundError(
                f"Ground truth metadata is missing for {len(missing_cases)} cases: {missing_cases[:5]}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_metrics.to_csv(args.output_dir / "class_case_metrics.csv", index=False)
    case_frame.to_csv(args.output_dir / "case_effects.csv", index=False)
    subgroup_summary(case_frame, args.reference_model, models).to_csv(
        args.output_dir / "subgroup_effects.csv", index=False
    )
    subgroup_summary(
        case_frame,
        args.reference_model,
        models,
        column_prefix="present_class_macro_",
        delta_prefix="present_class_macro_",
        metric_family="ground_truth_present_class_macro",
    ).to_csv(args.output_dir / "present_class_subgroup_effects.csv", index=False)
    class_effect_summary(long_metrics, args.reference_model, models).to_csv(
        args.output_dir / "class_effects.csv", index=False
    )
    absent_class_false_positive_summary(long_metrics, args.reference_model, models).to_csv(
        args.output_dir / "absent_class_false_positive_effects.csv", index=False
    )
    (
        long_metrics.groupby(["model", "class_id", "class_name"], as_index=False)["dice"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_csv(args.output_dir / "class_summary.csv", index=False)
    )
    manifest = {
        "schema_version": 1,
        "reference_model": args.reference_model,
        "models": models,
        "n_cases": int(case_frame["case_id"].nunique()),
        "n_classes": int(long_metrics["class_id"].nunique()),
        "n_class_case_metric_rows": int(len(long_metrics)),
        "missing_spacing_cases": int(case_frame["spacing_z_mm"].isna().sum()),
        "missing_lesion_volume_cases": int(case_frame["lesion_volume_ml"].isna().sum()),
        "metric_definitions": {
            "case_effects": "model-specific mean Dice across finite class values within each case; supports can differ when both truth and prediction are empty",
            "present_class_effects": "paired Dice restricted to classes present in ground truth; common support across models",
            "absent_class_false_positive_effects": "case-level false-positive rates where a class is absent from ground truth",
            "nnunet_foreground_mean_dice": "not recomputed here; read validation/summary.json directly for the primary model score",
            "online_ema_dice": "not read or reported by this analysis",
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_names = (
        "absent_class_false_positive_effects.csv",
        "analysis_manifest.json",
        "case_effects.csv",
        "class_case_metrics.csv",
        "class_effects.csv",
        "class_summary.csv",
        "present_class_subgroup_effects.csv",
        "subgroup_effects.csv",
    )
    checksum_lines = []
    for name in artifact_names:
        digest = hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {name}")
    (args.output_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

    print(f"Wrote case-level analysis to {args.output_dir}")
    print(f"Cases: {case_frame['case_id'].nunique()}; models: {', '.join(models)}")


if __name__ == "__main__":
    main()
