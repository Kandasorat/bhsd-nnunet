from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import SimpleITK as sitk

from evaluation.metrics import per_class_metrics


DEFAULT_CLASS_MAP: Dict[int, str] = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}


def read_segmentation(path: Path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def collect_prediction_files(prediction_dir: Path) -> List[Path]:
    return sorted(prediction_dir.glob("*.nii.gz"))


def evaluate_folder(
    prediction_dir: Path,
    ground_truth_dir: Path,
    output_csv: Path,
    model_name: str,
    class_map: Dict[int, str] | None = None,
    compute_hausdorff: bool = False,
) -> pd.DataFrame:
    class_map = class_map or DEFAULT_CLASS_MAP
    rows = []

    for prediction_path in collect_prediction_files(prediction_dir):
        gt_path = ground_truth_dir / prediction_path.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found for prediction: {prediction_path.name}")

        pred = read_segmentation(prediction_path)
        target = read_segmentation(gt_path)
        metrics = per_class_metrics(pred, target, class_map.keys(), compute_hausdorff=compute_hausdorff)

        for class_id, class_metrics in metrics.items():
            rows.append(
                {
                    "case_id": prediction_path.stem.replace(".nii", ""),
                    "model": model_name,
                    "class_id": class_id,
                    "class_name": class_map[class_id],
                    "dice": class_metrics.dice,
                    "iou": class_metrics.iou,
                    "precision": class_metrics.precision,
                    "recall": class_metrics.recall,
                    "hausdorff": class_metrics.hausdorff,
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hausdorff", action="store_true")
    args = parser.parse_args()

    evaluate_folder(
        prediction_dir=Path(args.prediction_dir),
        ground_truth_dir=Path(args.ground_truth_dir),
        output_csv=Path(args.output_csv),
        model_name=args.model_name,
        compute_hausdorff=args.hausdorff,
    )


if __name__ == "__main__":
    main()
