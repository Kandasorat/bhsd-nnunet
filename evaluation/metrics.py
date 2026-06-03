from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

try:
    from scipy.spatial.distance import directed_hausdorff
except Exception:  # pragma: no cover
    directed_hausdorff = None


@dataclass
class SegmentationMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    hausdorff: float | None = None


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(pred: np.ndarray, target: np.ndarray, compute_hausdorff: bool = False) -> SegmentationMetrics:
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, np.logical_not(target)).sum())
    fn = float(np.logical_and(np.logical_not(pred), target).sum())

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    hausdorff = None
    if compute_hausdorff and directed_hausdorff is not None and pred.any() and target.any():
        pred_points = np.argwhere(pred)
        target_points = np.argwhere(target)
        hausdorff = max(
            directed_hausdorff(pred_points, target_points)[0],
            directed_hausdorff(target_points, pred_points)[0],
        )

    return SegmentationMetrics(dice=dice, iou=iou, precision=precision, recall=recall, hausdorff=hausdorff)


def per_class_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_labels: Iterable[int],
    compute_hausdorff: bool = False,
) -> Dict[int, SegmentationMetrics]:
    return {
        int(class_label): binary_metrics(pred == class_label, target == class_label, compute_hausdorff=compute_hausdorff)
        for class_label in class_labels
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()) if arr.size else 0.0, "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}
