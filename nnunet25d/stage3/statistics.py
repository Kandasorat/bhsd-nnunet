from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PresentDiceRow:
    patient_id: str
    class_id: int
    model: str
    gt_present: bool
    dice: float


def _finite_present(rows: Iterable[PresentDiceRow], model: str) -> list[PresentDiceRow]:
    selected = [row for row in rows if row.model == model and row.gt_present and np.isfinite(row.dice)]
    if not selected:
        raise ValueError(f"No finite GT-present rows for model {model}")
    return selected


def per_class_present_dice(
    rows: Iterable[PresentDiceRow],
    model: str,
    class_ids: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in _finite_present(rows, model):
        grouped[row.class_id].append(float(row.dice))
    missing = [class_id for class_id in class_ids if not grouped[class_id]]
    if missing:
        raise ValueError(f"Missing GT-present support for classes {missing}")
    return {class_id: float(np.mean(grouped[class_id])) for class_id in class_ids}


def class_balanced_present_macro(
    rows: Iterable[PresentDiceRow],
    model: str,
    class_ids: Sequence[int] = (1, 2, 3, 4, 5),
) -> float:
    class_means = per_class_present_dice(rows, model, class_ids)
    return float(np.mean([class_means[class_id] for class_id in class_ids]))


def pooled_present_case_class_dice(rows: Iterable[PresentDiceRow], model: str) -> float:
    selected = _finite_present(rows, model)
    return float(np.mean([row.dice for row in selected]))


def validate_paired_support(rows: Iterable[PresentDiceRow], treatment: str, control: str) -> None:
    support: dict[str, set[tuple[str, int]]] = defaultdict(set)
    duplicates: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        if row.model not in {treatment, control} or not row.gt_present:
            continue
        key = (row.patient_id, row.class_id)
        if key in support[row.model]:
            duplicates[row.model].add(key)
        support[row.model].add(key)
    if duplicates:
        raise ValueError(f"Duplicate patient/class rows: {dict(duplicates)}")
    if support[treatment] != support[control]:
        raise ValueError("Treatment/control GT-present support differs")


def patient_cluster_bootstrap_present_delta(
    rows: Sequence[PresentDiceRow],
    treatment: str,
    control: str,
    *,
    iterations: int = 10_000,
    seed: int = 20_260_726,
    class_ids: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict[str, float | int | list[float]]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    validate_paired_support(rows, treatment, control)
    patient_ids = sorted(
        {row.patient_id for row in rows if row.model in {treatment, control} and row.gt_present}
    )
    if not patient_ids:
        raise ValueError("No paired patients")
    by_patient: dict[str, list[PresentDiceRow]] = defaultdict(list)
    for row in rows:
        if row.patient_id in patient_ids and row.model in {treatment, control}:
            by_patient[row.patient_id].append(row)

    observed = class_balanced_present_macro(rows, treatment, class_ids)
    observed -= class_balanced_present_macro(rows, control, class_ids)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    accepted = 0
    rejected_missing_class = 0
    maximum_draws = max(iterations * 100, iterations + 1000)
    draws = 0
    while accepted < iterations and draws < maximum_draws:
        draws += 1
        sampled_indices = rng.integers(0, len(patient_ids), size=len(patient_ids))
        sampled_rows = []
        for draw, patient_index in enumerate(sampled_indices):
            # A draw-specific ID preserves duplicate clusters without collapsing their weight.
            original_id = patient_ids[int(patient_index)]
            sampled_rows.extend(
                PresentDiceRow(
                    patient_id=f"draw_{draw}",
                    class_id=row.class_id,
                    model=row.model,
                    gt_present=row.gt_present,
                    dice=row.dice,
                )
                for row in by_patient[original_id]
            )
        try:
            value = class_balanced_present_macro(sampled_rows, treatment, class_ids)
            value -= class_balanced_present_macro(sampled_rows, control, class_ids)
        except ValueError as error:
            if "Missing GT-present support" not in str(error):
                raise
            rejected_missing_class += 1
            continue
        bootstrap[accepted] = value
        accepted += 1

    if accepted != iterations:
        raise RuntimeError(
            f"Could obtain only {accepted}/{iterations} bootstrap replicates with all five classes present"
        )

    lower, upper = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "treatment": treatment,
        "control": control,
        "patients": len(patient_ids),
        "iterations": iterations,
        "seed": seed,
        "rejected_missing_class_draws": rejected_missing_class,
        "observed_delta": float(observed),
        "ci95_percentile": [float(lower), float(upper)],
    }


def rows_from_dicts(records: Iterable[Mapping]) -> list[PresentDiceRow]:
    return [
        PresentDiceRow(
            patient_id=str(record["patient_id"]),
            class_id=int(record["class_id"]),
            model=str(record["model"]),
            gt_present=bool(record["gt_present"]),
            dice=float(record["dice"]),
        )
        for record in records
    ]
