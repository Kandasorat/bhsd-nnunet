from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label
from scipy.optimize import linear_sum_assignment


CONNECTIVITY_26 = generate_binary_structure(3, 3)


def _as_binary(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 3:
        raise ValueError(f"Stage3 physical metrics require a 3D mask, got {result.shape}")
    return result


def _spacing_zyx(spacing_zyx: Iterable[float]) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in spacing_zyx)
    if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError(f"Invalid positive ZYX spacing: {spacing}")
    return spacing


def dice_score(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    n_reference = int(reference.sum())
    n_prediction = int(prediction.sum())
    if n_reference == 0 and n_prediction == 0:
        return float("nan")
    intersection = int(np.logical_and(reference, prediction).sum())
    return 2.0 * intersection / (n_reference + n_prediction)


def physical_image_diagonal_mm(shape: Iterable[int], spacing_zyx: Iterable[float]) -> float:
    shape_array = np.asarray(tuple(int(value) for value in shape), dtype=np.float64)
    spacing_array = np.asarray(_spacing_zyx(spacing_zyx), dtype=np.float64)
    if shape_array.shape != (3,) or np.any(shape_array <= 0):
        raise ValueError(f"Invalid image shape: {tuple(shape_array)}")
    return float(np.linalg.norm((shape_array - 1.0) * spacing_array))


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = _as_binary(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_and(mask, np.logical_not(binary_erosion(mask, structure=CONNECTIVITY_26, border_value=0)))


def bidirectional_surface_distances_mm(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_zyx: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    reference = _as_binary(reference)
    prediction = _as_binary(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    spacing = _spacing_zyx(spacing_zyx)
    reference_surface = _surface(reference)
    prediction_surface = _surface(prediction)
    if not reference_surface.any() or not prediction_surface.any():
        raise ValueError("Surface distances require both masks to be nonempty")
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    distance_to_reference = distance_transform_edt(~reference_surface, sampling=spacing)
    return distance_to_prediction[reference_surface], distance_to_reference[prediction_surface]


def hd95_mm(reference: np.ndarray, prediction: np.ndarray, spacing_zyx: Iterable[float]) -> float:
    reference = _as_binary(reference)
    prediction = _as_binary(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    if not reference.any():
        return float("nan")
    if not prediction.any():
        return physical_image_diagonal_mm(reference.shape, spacing_zyx)
    directed = bidirectional_surface_distances_mm(reference, prediction, spacing_zyx)
    return float(np.percentile(np.concatenate(directed), 95))


def nsd_mm(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_zyx: Iterable[float],
    tolerance_mm: float = 3.0,
) -> float:
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be nonnegative")
    reference = _as_binary(reference)
    prediction = _as_binary(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    if not reference.any():
        return float("nan")
    if not prediction.any():
        return 0.0
    reference_to_prediction, prediction_to_reference = bidirectional_surface_distances_mm(
        reference, prediction, spacing_zyx
    )
    within = np.count_nonzero(reference_to_prediction <= tolerance_mm)
    within += np.count_nonzero(prediction_to_reference <= tolerance_mm)
    return float(within / (reference_to_prediction.size + prediction_to_reference.size))


@dataclass(frozen=True)
class LesionMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    f1: float
    recall: float
    small_gt_lesions: int
    small_true_positive: int
    small_lesion_recall: float
    matched_pairs: tuple[tuple[int, int, int], ...]


def lesion_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_zyx: Iterable[float],
    small_threshold_ml: float = 1.0,
) -> LesionMetrics:
    reference = _as_binary(reference)
    prediction = _as_binary(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    spacing = _spacing_zyx(spacing_zyx)
    reference_labels, n_reference = label(reference, structure=CONNECTIVITY_26)
    prediction_labels, n_prediction = label(prediction, structure=CONNECTIVITY_26)

    intersections = np.zeros((n_reference, n_prediction), dtype=np.int64)
    overlap = np.logical_and(reference_labels > 0, prediction_labels > 0)
    if overlap.any():
        pairs, counts = np.unique(
            np.stack((reference_labels[overlap], prediction_labels[overlap]), axis=1),
            axis=0,
            return_counts=True,
        )
        for (reference_id, prediction_id), count in zip(pairs, counts):
            intersections[reference_id - 1, prediction_id - 1] = int(count)

    matched = []
    if n_reference and n_prediction:
        reference_indices, prediction_indices = linear_sum_assignment(-intersections)
        for reference_index, prediction_index in zip(reference_indices, prediction_indices):
            intersection = int(intersections[reference_index, prediction_index])
            if intersection > 0:
                matched.append((int(reference_index + 1), int(prediction_index + 1), intersection))

    true_positive = len(matched)
    false_positive = int(n_prediction - true_positive)
    false_negative = int(n_reference - true_positive)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = float(2 * true_positive / f1_denominator) if f1_denominator else float("nan")
    recall = float(true_positive / n_reference) if n_reference else float("nan")

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    reference_sizes = np.bincount(reference_labels.ravel(), minlength=n_reference + 1)[1:]
    small_reference_ids = {
        index + 1
        for index, voxel_count in enumerate(reference_sizes)
        if float(voxel_count) * voxel_volume_ml < small_threshold_ml
    }
    matched_reference_ids = {reference_id for reference_id, _, _ in matched}
    small_true_positive = len(small_reference_ids & matched_reference_ids)
    small_recall = (
        float(small_true_positive / len(small_reference_ids)) if small_reference_ids else float("nan")
    )
    return LesionMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        f1=f1,
        recall=recall,
        small_gt_lesions=len(small_reference_ids),
        small_true_positive=small_true_positive,
        small_lesion_recall=small_recall,
        matched_pairs=tuple(matched),
    )


def evaluate_patient_class(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_zyx: Iterable[float],
) -> dict[str, float | int | bool | list]:
    reference = _as_binary(reference)
    prediction = _as_binary(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    spacing = _spacing_zyx(spacing_zyx)
    reference_voxels = int(reference.sum())
    prediction_voxels = int(prediction.sum())
    intersection_voxels = int(np.logical_and(reference, prediction).sum())
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    present = reference_voxels > 0
    lesions = lesion_metrics(reference, prediction, spacing) if present else None
    return {
        "gt_present": present,
        "reference_voxels": reference_voxels,
        "prediction_voxels": prediction_voxels,
        "intersection_voxels": intersection_voxels,
        "dice": dice_score(reference, prediction),
        "absent_any_fp": bool(not present and prediction_voxels > 0),
        "absent_fp_volume_ml": float(prediction_voxels * voxel_volume_ml) if not present else float("nan"),
        "hd95_mm": hd95_mm(reference, prediction, spacing) if present else float("nan"),
        "nsd_3mm": nsd_mm(reference, prediction, spacing, 3.0) if present else float("nan"),
        "lesions": asdict(lesions) if lesions is not None else None,
    }


def assert_geometry_match(reference_image, prediction_image, *, atol: float = 1e-6) -> None:
    """Validate SimpleITK image identity before metric computation."""

    if reference_image.GetSize() != prediction_image.GetSize():
        raise ValueError("Prediction/GT size mismatch")
    for name, left, right in (
        ("spacing", reference_image.GetSpacing(), prediction_image.GetSpacing()),
        ("origin", reference_image.GetOrigin(), prediction_image.GetOrigin()),
        ("direction", reference_image.GetDirection(), prediction_image.GetDirection()),
    ):
        if not np.allclose(left, right, rtol=0.0, atol=atol):
            raise ValueError(f"Prediction/GT {name} mismatch: {left} != {right}")
