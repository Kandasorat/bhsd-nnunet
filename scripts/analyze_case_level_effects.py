from __future__ import annotations

import argparse
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
    if "reference_file" not in frame:
        frame["reference_file"] = None
    return frame[["model", "case_id", "class_id", "dice", "n_ref", "reference_file"]].to_dict("records")


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


def subgroup_summary(case_frame: pd.DataFrame, reference_model: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        if model == reference_model:
            continue
        delta_column = f"delta_{model}_vs_{reference_model}"
        for field in ("spacing_group", "lesion_size_group"):
            for group_name, group in case_frame.groupby(field, dropna=False):
                delta = group[delta_column].dropna()
                rows.append(
                    {
                        "candidate": model,
                        "reference": reference_model,
                        "subgroup_variable": field,
                        "subgroup": "missing" if pd.isna(group_name) else str(group_name),
                        "n_cases": int(len(delta)),
                        "reference_mean_dice": float(group[reference_model].mean()),
                        "candidate_mean_dice": float(group[model].mean()),
                        "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                        "median_delta": float(delta.median()) if len(delta) else np.nan,
                        "improved_fraction": float((delta > 0).mean()) if len(delta) else np.nan,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_metrics.to_csv(args.output_dir / "class_case_metrics.csv", index=False)
    case_frame.to_csv(args.output_dir / "case_effects.csv", index=False)
    subgroup_summary(case_frame, args.reference_model, models).to_csv(
        args.output_dir / "subgroup_effects.csv", index=False
    )
    (
        long_metrics.groupby(["model", "class_id", "class_name"], as_index=False)["dice"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_csv(args.output_dir / "class_summary.csv", index=False)
    )

    print(f"Wrote case-level analysis to {args.output_dir}")
    print(f"Cases: {case_frame['case_id'].nunique()}; models: {', '.join(models)}")


if __name__ == "__main__":
    main()
