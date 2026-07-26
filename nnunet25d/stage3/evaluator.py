from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from nnunet25d.stage3.metrics import assert_geometry_match, evaluate_patient_class


def evaluate_folder(
    gt_folder: str | Path,
    prediction_folder: str | Path,
    *,
    fold: int,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict:
    if fold == 0:
        raise RuntimeError("Stage3 fold0 metric computation is forbidden")
    gt_folder = Path(gt_folder)
    prediction_folder = Path(prediction_folder)
    gt_files = sorted(gt_folder.glob("*.nii.gz"))
    if not gt_files:
        raise FileNotFoundError(f"No GT files in {gt_folder}")
    rows = []
    for gt_path in gt_files:
        prediction_path = prediction_folder / gt_path.name
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        gt_image = sitk.ReadImage(str(gt_path))
        prediction_image = sitk.ReadImage(str(prediction_path))
        assert_geometry_match(gt_image, prediction_image)
        gt = sitk.GetArrayFromImage(gt_image)
        prediction = sitk.GetArrayFromImage(prediction_image)
        if not set(np.unique(gt)).issubset(set(range(6))):
            raise ValueError(f"Unexpected GT labels in {gt_path.name}")
        if not set(np.unique(prediction)).issubset(set(range(6))):
            raise ValueError(f"Unexpected prediction labels in {prediction_path.name}")
        spacing_zyx = tuple(reversed(gt_image.GetSpacing()))
        for class_id in range(1, 6):
            result = evaluate_patient_class(gt == class_id, prediction == class_id, spacing_zyx)
            lesions = result.pop("lesions")
            row = {
                "patient_id": gt_path.name.removesuffix(".nii.gz"),
                "fold": fold,
                "class_id": class_id,
                **result,
            }
            if lesions is not None:
                row.update({f"lesion_{key}": value for key, value in lesions.items() if key != "matched_pairs"})
            rows.append(row)

    output_json = Path(output_json)
    output_csv = Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"fold": fold, "rows": rows}, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"fold": fold, "patients": len(gt_files), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage3 physical-space evaluator")
    parser.add_argument("--gt-folder", required=True)
    parser.add_argument("--prediction-folder", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    print(evaluate_folder(**vars(args)))


if __name__ == "__main__":
    main()
