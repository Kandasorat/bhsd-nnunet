from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import binary_metrics


def read_segmentation(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def collect_prediction_files(prediction_dir: Path) -> list[Path]:
    return sorted(prediction_dir.glob("*.nii.gz"))


def collect_ground_truth_files(gt_dir: Path) -> list[Path]:
    return sorted(gt_dir.glob("*.nii.gz"))


def _validate_binary_labels(array: np.ndarray, path: Path, label_name: str) -> None:
    unique_values = set(int(v) for v in np.unique(array))
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{label_name} at {path} contains non-binary labels: {sorted(unique_values)}")


def _slice_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    gt_positive = gt.reshape(gt.shape[0], -1).any(axis=1)
    pred_positive = pred.reshape(pred.shape[0], -1).any(axis=1)

    gt_negative = ~gt_positive
    negative_count = int(gt_negative.sum())
    positive_count = int(gt_positive.sum())

    negative_fp_rate = float((pred_positive[gt_negative]).mean()) if negative_count else 0.0
    positive_recall = float((pred_positive[gt_positive]).mean()) if positive_count else 0.0
    return negative_fp_rate, positive_recall


def evaluate_binary_folder(pred_dir: Path, gt_dir: Path, model_name: str, out_dir: Path) -> tuple[Path, Path]:
    rows = []
    prediction_files = collect_prediction_files(pred_dir)
    ground_truth_files = collect_ground_truth_files(gt_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")
    if not ground_truth_files:
        raise FileNotFoundError(f"No ground-truth files found in {gt_dir}")

    prediction_names = {path.name for path in prediction_files}
    ground_truth_names = {path.name for path in ground_truth_files}
    missing_predictions = sorted(ground_truth_names - prediction_names)
    missing_ground_truth = sorted(prediction_names - ground_truth_names)
    if missing_predictions or missing_ground_truth:
        raise ValueError(
            "Prediction and ground-truth case IDs do not match. "
            f"Missing predictions: {missing_predictions[:10]}; missing ground truth: {missing_ground_truth[:10]}"
        )

    for prediction_path in prediction_files:
        gt_path = gt_dir / prediction_path.name

        pred = read_segmentation(prediction_path)
        gt = read_segmentation(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {prediction_path.name}: pred {pred.shape}, gt {gt.shape}")

        _validate_binary_labels(gt, gt_path, "Ground-truth")
        _validate_binary_labels(pred, prediction_path, "Prediction")

        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)
        metrics = binary_metrics(pred_bool, gt_bool, compute_hausdorff=True)
        intersection_voxels = int(np.logical_and(pred_bool, gt_bool).sum())
        pred_positive_voxels = int(pred_bool.sum())
        gt_positive_voxels = int(gt_bool.sum())
        false_positive_case = bool(pred_positive_voxels > 0 and gt_positive_voxels == 0)
        false_negative_case = bool(pred_positive_voxels == 0 and gt_positive_voxels > 0)
        negative_slice_fp_rate, positive_slice_recall = _slice_metrics(pred_bool, gt_bool)

        rows.append(
            {
                "model": model_name,
                "case_id": prediction_path.stem.replace(".nii", ""),
                "dice": metrics.dice,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "intersection_voxels": intersection_voxels,
                "pred_positive_voxels": pred_positive_voxels,
                "gt_positive_voxels": gt_positive_voxels,
                "false_positive_case": false_positive_case,
                "false_negative_case": false_negative_case,
                "gt_negative_slice_false_positive_rate": negative_slice_fp_rate,
                "gt_positive_slice_recall": positive_slice_recall,
                "hausdorff": metrics.hausdorff,
            }
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = out_dir / "binary_segmentation_per_case.csv"
    df.to_csv(per_case_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "n_cases": int(len(df)),
                "mean_dice": float(df["dice"].mean()),
                "median_dice": float(df["dice"].median()),
                "std_dice": float(df["dice"].std(ddof=1)) if len(df) > 1 else 0.0,
                "min_dice": float(df["dice"].min()),
                "max_dice": float(df["dice"].max()),
                "mean_precision": float(df["precision"].mean()),
                "mean_recall": float(df["recall"].mean()),
                "n_false_positive_cases": int(df["false_positive_case"].sum()),
                "n_false_negative_cases": int(df["false_negative_case"].sum()),
                "mean_gt_negative_slice_false_positive_rate": float(df["gt_negative_slice_false_positive_rate"].mean()),
                "mean_gt_positive_slice_recall": float(df["gt_positive_slice_recall"].mean()),
                "mean_hausdorff": float(df["hausdorff"].dropna().mean()) if df["hausdorff"].notna().any() else float("nan"),
            }
        ]
    )
    summary_csv = out_dir / "binary_segmentation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    return per_case_csv, summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluate_binary_folder(
        pred_dir=Path(args.pred_dir),
        gt_dir=Path(args.gt_dir),
        model_name=args.model_name,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
