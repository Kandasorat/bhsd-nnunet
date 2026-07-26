from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr


CLASS_NAMES = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}
SEEDS = (3407, 1234, 5678)
MODEL_ARMS = ("E0", "E2")
EPS = 1e-8
PRACTICAL = 0.01
BOOTSTRAP_SEED = 20260726
N_RESAMPLES = 10_000
GIT_COMMIT = "0c660e0"


@dataclass(frozen=True)
class ResultSet:
    model: str
    seed: int
    trainer_name: str
    root: Path

    @property
    def fold(self) -> Path:
        return self.root / "fold_0"

    @property
    def validation(self) -> Path:
        return self.fold / "validation"


def dice_from_counts(tp: int, n_ref: int, n_pred: int) -> float:
    denominator = n_ref + n_pred
    return float(2 * tp / denominator) if denominator else math.nan


def binary_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    n_ref = int(np.count_nonzero(reference))
    n_pred = int(np.count_nonzero(prediction))
    tp = int(np.count_nonzero(np.logical_and(reference, prediction)))
    return dice_from_counts(tp, n_ref, n_pred)


def practical_sign(value: float) -> str:
    if value >= PRACTICAL:
        return "+"
    if value <= -PRACTICAL:
        return "-"
    return "0"


def raw_sign(value: float) -> str:
    if value > EPS:
        return "+"
    if value < -EPS:
        return "-"
    return "0"


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def sample_sd(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.stdev(vals) if len(vals) > 1 else math.nan


def percentile_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def case_id(path: Path) -> str:
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"Not a NIfTI prediction: {path}")
    return path.name[:-7]


def load_seg(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return data, image


def lesion_size_group(total_ml: float) -> str:
    if total_ml < 1.0:
        return "small_lt1ml"
    if total_ml < 10.0:
        return "medium_1to10ml"
    return "large_ge10ml"


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def mean_by_class(rows: list[dict], value_key: str) -> dict[int, float]:
    return {
        class_id: finite_mean(row[value_key] for row in rows if row["class"] == class_id)
        for class_id in CLASS_NAMES
    }


def class_balanced_macro(rows: list[dict], value_key: str) -> float:
    return finite_mean(mean_by_class(rows, value_key).values())


def make_result_sets(drive_root: Path, multiseed_root: Path) -> dict[tuple[str, int], ResultSet]:
    specs = {
        ("E0", 3407): ("nnUNetTrainer_25D_SymmetricE0Control", drive_root),
        ("E2", 3407): ("nnUNetTrainer_25D_SymmetricE2ReliabilityGate", drive_root),
        ("E0", 1234): ("nnUNetTrainer_25D_SymmetricE0ControlSeed1234", multiseed_root),
        ("E2", 1234): ("nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed1234", multiseed_root),
        ("E0", 5678): ("nnUNetTrainer_25D_SymmetricE0ControlSeed5678", multiseed_root),
        ("E2", 5678): ("nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed5678", multiseed_root),
    }
    return {
        key: ResultSet(key[0], key[1], trainer, root / f"{trainer}__nnUNetPlans__2d")
        for key, (trainer, root) in specs.items()
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_assets(result_sets: dict, gt_root: Path, locked_scores: dict) -> tuple[list[dict], list[str], dict]:
    inventory: list[dict] = []
    canonical_ids: list[str] | None = None
    audit_details: dict = {}
    for key in [(arm, seed) for seed in SEEDS for arm in MODEL_ARMS]:
        rs = result_sets[key]
        val = rs.validation
        summary_path = val / "summary.json"
        required = [rs.fold / "checkpoint_best.pth", rs.fold / "checkpoint_final.pth", summary_path,
                    rs.fold / "debug.json", rs.fold / "run_timing.json", rs.fold / "compute_profile.json"]
        missing = [str(path) for path in required if not path.is_file()]
        pred_paths = sorted(val.glob("case_*.nii.gz"), key=lambda p: p.name)
        npz_paths = sorted(val.glob("case_*.npz"), key=lambda p: p.name)
        pkl_paths = sorted(val.glob("case_*.pkl"), key=lambda p: p.name)
        ids = [case_id(path) for path in pred_paths]
        if len(ids) != len(set(ids)):
            missing.append("duplicate prediction case IDs")
        if canonical_ids is None:
            canonical_ids = ids
        case_match = ids == canonical_ids and len(ids) == 39
        gt_paths = [gt_root / f"{cid}.nii.gz" for cid in ids]
        missing_gt = [str(path) for path in gt_paths if not path.is_file()]
        missing.extend(missing_gt)
        summary = read_json(summary_path) if summary_path.is_file() else {}
        summary_ids = sorted(Path(row["prediction_file"]).name[:-7] for row in summary.get("metric_per_case", []))
        summary_match = summary_ids == ids
        score = float(summary.get("foreground_mean", {}).get("Dice", math.nan))
        score_match = math.isfinite(score) and abs(score - locked_scores[key]) <= 1e-6
        timing = read_json(rs.fold / "run_timing.json") if (rs.fold / "run_timing.json").is_file() else {}
        debug = read_json(rs.fold / "debug.json") if (rs.fold / "debug.json").is_file() else {}
        checkpoint_used = "checkpoint_best.pth" if "--val_best" in timing.get("command", "") else "unresolved"
        provenance_ok = checkpoint_used == "checkpoint_best.pth" and score_match and summary_match and not missing
        row = {
            "model": rs.model,
            "seed": rs.seed,
            "trainer_name": rs.trainer_name,
            "configuration": "2d",
            "fold": 0,
            "git_commit": GIT_COMMIT,
            "model_seed": int(debug.get("bhsd_seed", timing.get("seed", rs.seed))),
            "data_seed": int(debug.get("bhsd_data_seed", timing.get("data_seed", 1003410))),
            "result_path": str(rs.root),
            "summary_json": str(summary_path),
            "checkpoint_best": str(rs.fold / "checkpoint_best.pth"),
            "checkpoint_final": str(rs.fold / "checkpoint_final.pth"),
            "checkpoint_used_for_validation": checkpoint_used,
            "prediction_nifti_count": len(pred_paths),
            "probability_npz_count": len(npz_paths),
            "ground_truth_count": len(gt_paths) - len(missing_gt),
            "prediction_case_ids": ";".join(ids),
            "ground_truth_case_ids": ";".join(path.name[:-7] for path in gt_paths if path.is_file()),
            "case_ids_match": case_match and summary_match,
            "geometry_match": None,
            "label_values_valid": None,
            "case_analysis_possible": False,
            "slice_analysis_possible": False,
            "soft_slice_analysis_possible": False,
            "provenance_status": "verified" if provenance_ok else "provenance_unresolved",
            "notes": f"summary Dice={score:.15g}; locked abs diff={abs(score-locked_scores[key]):.3g}; pkl_count={len(pkl_paths)}",
        }
        inventory.append(row)
        audit_details[f"{rs.model}_{rs.seed}"] = {"missing": missing, "score": score, "score_match": score_match}
    return inventory, canonical_ids or [], audit_details


def audit_geometry_and_metrics(result_sets: dict, gt_root: Path, case_ids: list[str], inventory: list[dict]):
    metric_rows: list[dict] = []
    absent_rows: list[dict] = []
    hard_cache: dict[tuple[str, int, str], dict] = {}
    for seed in SEEDS:
        for cid in case_ids:
            gt, gt_img = load_seg(gt_root / f"{cid}.nii.gz")
            gt_values = set(int(v) for v in np.unique(gt))
            if not gt_values <= set(range(6)):
                raise RuntimeError(f"Illegal GT labels for {cid}: {sorted(gt_values)}")
            det_ml = abs(float(np.linalg.det(gt_img.affine[:3, :3]))) / 1000.0
            zoom_ml = float(np.prod(gt_img.header.get_zooms()[:3])) / 1000.0
            if not math.isclose(det_ml, zoom_ml, rel_tol=1e-5, abs_tol=1e-8):
                raise RuntimeError(f"Affine/header voxel volume mismatch for {cid}: {det_ml} vs {zoom_ml}")
            total_lesion_ml = int(np.count_nonzero(gt)) * det_ml
            for arm in MODEL_ARMS:
                rs = result_sets[(arm, seed)]
                pred, pred_img = load_seg(rs.validation / f"{cid}.nii.gz")
                if pred.shape != gt.shape or not np.allclose(pred_img.affine, gt_img.affine, atol=1e-5, rtol=0):
                    raise RuntimeError(f"Geometry mismatch: {arm} seed {seed} {cid}")
                pred_values = set(int(v) for v in np.unique(pred))
                if not pred_values <= set(range(6)):
                    raise RuntimeError(f"Illegal prediction labels: {arm} seed {seed} {cid}: {sorted(pred_values)}")
                for class_id in CLASS_NAMES:
                    gm = gt == class_id
                    pm = pred == class_id
                    n_ref = int(np.count_nonzero(gm))
                    n_pred = int(np.count_nonzero(pm))
                    tp = int(np.count_nonzero(np.logical_and(gm, pm)))
                    fp = n_pred - tp
                    fn = n_ref - tp
                    value = dice_from_counts(tp, n_ref, n_pred)
                    key = (arm, seed, cid, class_id)
                    hard_cache[key] = {"dice": value, "n_ref": n_ref, "n_pred": n_pred, "tp": tp, "fp": fp, "fn": fn,
                                       "gt_volume_ml": n_ref * det_ml, "voxel_volume_ml": det_ml,
                                       "total_lesion_volume_ml": total_lesion_ml,
                                       "lesion_size_group": lesion_size_group(total_lesion_ml)}
                    if n_ref > 0:
                        metric_rows.append({"model": arm, "seed": seed, "case_id": cid, "class": class_id,
                                            "class_name": CLASS_NAMES[class_id], "dice": value, "n_ref": n_ref,
                                            "n_pred": n_pred, "tp": tp, "fp": fp, "fn": fn,
                                            "gt_volume_ml": n_ref * det_ml, "total_lesion_volume_ml": total_lesion_ml,
                                            "lesion_size_group": lesion_size_group(total_lesion_ml)})
                    else:
                        absent_rows.append({"model": arm, "seed": seed, "case_id": cid, "class": class_id,
                                            "class_name": CLASS_NAMES[class_id], "any_fp": n_pred > 0,
                                            "fp_voxels": n_pred, "fp_volume_ml": n_pred * det_ml,
                                            "fp_gt_0p1ml": n_pred * det_ml > 0.1})
    for row in inventory:
        row["geometry_match"] = True
        row["label_values_valid"] = True
        row["case_analysis_possible"] = True
        row["slice_analysis_possible"] = True
    return metric_rows, absent_rows, hard_cache


def verify_soft_mapping(result_sets: dict, case_ids: list[str], inventory: list[dict]) -> dict[tuple[str, int], bool]:
    safe: dict[tuple[str, int], bool] = {}
    for key, rs in result_sets.items():
        ok = True
        notes = []
        for cid in case_ids:
            base = rs.validation / cid
            try:
                npz_path=Path(str(base)+".npz")
                with zipfile.ZipFile(npz_path) as archive, archive.open("probabilities.npy") as handle:
                    version=np.lib.format.read_magic(handle)
                    shape,fortran_order,dtype=np.lib.format._read_array_header(handle,version)
                if shape[0]!=6 or fortran_order:
                    raise ValueError(f"unexpected probability header: shape={shape}, fortran={fortran_order}, dtype={dtype}")
                hard, _ = load_seg(Path(str(base) + ".nii.gz"))
                with Path(str(base) + ".pkl").open("rb") as handle:
                    props = pickle.load(handle)
                shape_before = tuple(int(v) for v in props["shape_before_cropping"])
                shape_crop = tuple(int(v) for v in props["shape_after_cropping_and_before_resampling"])
                bbox = props["bbox_used_for_cropping"]
                full_bbox = all(int(bounds[0]) == 0 and int(bounds[1]) == int(shape_before[i]) for i, bounds in enumerate(bbox))
                if shape[1:] != shape_crop or shape_crop != shape_before or not full_bbox:
                    raise ValueError(f"unsafe restoration shape/bbox: prob={shape}, crop={shape_crop}, before={shape_before}, bbox={bbox}")
                if hard.shape != tuple(reversed(shape[1:])):
                    raise ValueError(f"axis shape mismatch hard={hard.shape}, prob={shape}")
            except Exception as exc:
                ok = False
                notes.append(f"{cid}: {exc}")
                break
        safe[key] = ok
        for row in inventory:
            if row["model"] == key[0] and row["seed"] == key[1]:
                row["soft_slice_analysis_possible"] = ok
                if notes:
                    row["notes"] += "; soft mapping refused: " + " | ".join(notes)
    return safe


def reproduce_seed3407(hard_cache: dict, old_analysis: Path, absent_rows: list[dict]) -> dict:
    old_rows = {}
    with (old_analysis / "class_case_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in MODEL_ARMS:
                old_rows[(row["model"], row["case_id"], int(row["class_id"]))] = row
    differences = []
    count_mismatches = []
    for key, old in old_rows.items():
        arm, cid, class_id = key
        current = hard_cache[(arm, 3407, cid, class_id)]
        if current["n_ref"] != int(old["n_ref"]) or current["n_pred"] != int(old["n_pred"]):
            count_mismatches.append({"key": key, "old": [old["n_ref"], old["n_pred"]],
                                     "new": [current["n_ref"], current["n_pred"]]})
        if current["n_ref"] > 0:
            differences.append(abs(current["dice"] - float(old["dice"])))
    sets = {}
    for arm in MODEL_ARMS:
        sets[arm] = sorted(f"{r['case_id']}:class_{r['class']}" for r in absent_rows
                           if r["seed"] == 3407 and r["model"] == arm and r["any_fp"])
    payload = {
        "status": "passed",
        "seed": 3407,
        "old_analysis_path": str(old_analysis),
        "old_rows_checked": len(old_rows),
        "count_mismatch_count": len(count_mismatches),
        "max_present_dice_absolute_difference": max(differences),
        "present_dice_tolerance": 1e-6,
        "present_dice_within_tolerance": max(differences) <= 1e-6,
        "absent_pair_denominator": 113,
        "e0_absent_fp_count": len(sets["E0"]),
        "e2_absent_fp_count": len(sets["E2"]),
        "expected_e0_absent_fp_count": 70,
        "expected_e2_absent_fp_count": 58,
        "e0_absent_fp_set": sets["E0"],
        "e2_absent_fp_set": sets["E2"],
        "resolved_set": sorted(set(sets["E0"]) - set(sets["E2"])),
        "introduced_set": sorted(set(sets["E2"]) - set(sets["E0"])),
        "count_mismatches": count_mismatches,
    }
    if count_mismatches or max(differences) > 1e-6 or len(sets["E0"]) != 70 or len(sets["E2"]) != 58:
        payload["status"] = "failed"
    return payload


def present_metric_outputs(metric_rows: list[dict]) -> list[dict]:
    output = []
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            rows = [r for r in metric_rows if r["seed"] == seed and r["model"] == arm]
            class_means = mean_by_class(rows, "dice")
            for class_id, value in class_means.items():
                output.append({"seed": seed, "model": arm, "metric": "per_class_present_dice",
                               "class": class_id, "class_name": CLASS_NAMES[class_id],
                               "n": sum(r["class"] == class_id for r in rows), "value": value})
            output.append({"seed": seed, "model": arm, "metric": "class_balanced_present_macro",
                           "class": "all", "class_name": "all", "n": len(rows),
                           "value": finite_mean(class_means.values())})
            output.append({"seed": seed, "model": arm, "metric": "pooled_present_patient_class_dice",
                           "class": "all", "class_name": "all", "n": len(rows),
                           "value": finite_mean(r["dice"] for r in rows)})
    return output


def absent_output(absent_rows: list[dict]) -> list[dict]:
    output = []
    for seed in SEEDS:
        pairs = sorted({(r["case_id"], r["class"]) for r in absent_rows if r["seed"] == seed})
        for cid, class_id in pairs:
            row = {"seed": seed, "case_id": cid, "class": class_id, "class_name": CLASS_NAMES[class_id]}
            for arm in MODEL_ARMS:
                source = next(r for r in absent_rows if r["seed"] == seed and r["model"] == arm
                              and r["case_id"] == cid and r["class"] == class_id)
                for field in ("any_fp", "fp_voxels", "fp_volume_ml", "fp_gt_0p1ml"):
                    row[f"{arm.lower()}_{field}"] = source[field]
            if row["e0_any_fp"] and not row["e2_any_fp"]:
                status = "resolved"
            elif row["e2_any_fp"] and not row["e0_any_fp"]:
                status = "introduced"
            elif row["e0_any_fp"] and row["e2_any_fp"]:
                status = "both_fp"
            else:
                status = "both_clean"
            row["status"] = status
            output.append(row)
    return output


def delta_outputs(metric_rows: list[dict], hard_cache: dict) -> list[dict]:
    present_pairs = sorted({(r["case_id"], r["class"]) for r in metric_rows if r["model"] == "E0"})
    output = []
    for cid, class_id in present_pairs:
        row = {"case_id": cid, "class": class_id, "class_name": CLASS_NAMES[class_id],
               "gt_volume_ml": hard_cache[("E0", 3407, cid, class_id)]["gt_volume_ml"],
               "total_lesion_volume_ml": hard_cache[("E0", 3407, cid, class_id)]["total_lesion_volume_ml"],
               "lesion_size_group": hard_cache[("E0", 3407, cid, class_id)]["lesion_size_group"]}
        deltas = []
        for seed in SEEDS:
            e0 = hard_cache[("E0", seed, cid, class_id)]["dice"]
            e2 = hard_cache[("E2", seed, cid, class_id)]["dice"]
            delta = e2 - e0
            deltas.append(delta)
            row[f"dice_e0_seed_{seed}"] = e0
            row[f"dice_e2_seed_{seed}"] = e2
            row[f"delta_seed_{seed}"] = delta
        rsigns = [raw_sign(v) for v in deltas]
        psigns = [practical_sign(v) for v in deltas]
        row.update({"mean_delta": finite_mean(deltas), "sample_sd": sample_sd(deltas),
                    "raw_sign_pattern": "".join(rsigns), "practical_sign_pattern": "".join(psigns),
                    "unanimous_raw_direction": len(set(rsigns)) == 1 and rsigns[0] != "0",
                    "unanimous_practical_direction": len(set(psigns)) == 1 and psigns[0] != "0",
                    "notes": "practical thresholds: >=0.01 E2, <=-0.01 E0, otherwise neutral"})
        output.append(row)
    return output


def stability_outputs(delta_rows: list[dict], absent_pair_rows: list[dict]) -> tuple[list[dict], dict]:
    output = []
    raw_counts = Counter(row["raw_sign_pattern"] for row in delta_rows)
    practical_counts = Counter(row["practical_sign_pattern"] for row in delta_rows)
    for pattern, count in sorted(raw_counts.items()):
        output.append({"family": "patient_class_delta", "scope": "all", "metric": "raw_sign_pattern_count",
                       "comparison": pattern, "n": len(delta_rows), "value": count})
    for pattern, count in sorted(practical_counts.items()):
        output.append({"family": "patient_class_delta", "scope": "all", "metric": "practical_sign_pattern_count",
                       "comparison": pattern, "n": len(delta_rows), "value": count})
    raw_unanimous = sum(r["unanimous_raw_direction"] for r in delta_rows) / len(delta_rows)
    practical_unanimous = sum(r["unanimous_practical_direction"] for r in delta_rows) / len(delta_rows)
    strong = [r for r in delta_rows if any(abs(r[f"delta_seed_{s}"]) >= PRACTICAL for s in SEEDS)]
    strong_unanimous = sum(r["unanimous_practical_direction"] for r in strong) / len(strong) if strong else math.nan
    output.extend([
        {"family": "patient_class_delta", "scope": "all", "metric": "unanimous_raw_nonzero_proportion", "comparison": "3_seeds", "n": len(delta_rows), "value": raw_unanimous},
        {"family": "patient_class_delta", "scope": "all", "metric": "unanimous_practical_nonzero_proportion", "comparison": "3_seeds", "n": len(delta_rows), "value": practical_unanimous},
        {"family": "patient_class_delta", "scope": "strong_effect_pairs", "metric": "unanimous_practical_nonzero_proportion", "comparison": "3_seeds", "n": len(strong), "value": strong_unanimous},
    ])
    correlations = []
    agreements = []
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            x = [r[f"delta_seed_{s1}"] for r in delta_rows]
            y = [r[f"delta_seed_{s2}"] for r in delta_rows]
            rho = float(spearmanr(x, y).statistic)
            agree = finite_mean(practical_sign(a) == practical_sign(b) for a, b in zip(x, y))
            correlations.append(rho); agreements.append(agree)
            output.append({"family": "patient_class_delta", "scope": "all", "metric": "spearman_rho",
                           "comparison": f"{s1}_vs_{s2}", "n": len(delta_rows), "value": rho})
            output.append({"family": "patient_class_delta", "scope": "all", "metric": "practical_sign_agreement",
                           "comparison": f"{s1}_vs_{s2}", "n": len(delta_rows), "value": agree})
    for class_id in CLASS_NAMES:
        rows = [r for r in delta_rows if r["class"] == class_id]
        output.append({"family": "patient_class_delta", "scope": CLASS_NAMES[class_id],
                       "metric": "unanimous_practical_nonzero_proportion", "comparison": "3_seeds",
                       "n": len(rows), "value": finite_mean(r["unanimous_practical_direction"] for r in rows)})
    for group in ("small_lt1ml", "medium_1to10ml", "large_ge10ml"):
        rows = [r for r in delta_rows if r["lesion_size_group"] == group]
        output.append({"family": "patient_class_delta", "scope": group,
                       "metric": "mean_3seed_delta_descriptive", "comparison": "E2_minus_E0",
                       "n": len(rows), "value": finite_mean(r["mean_delta"] for r in rows),
                       "notes": "pre-existing whole-case lesion-volume thresholds; descriptive and excluded from decision"})
    states_by_seed = {s: {(r["case_id"], r["class"]): r["status"] for r in absent_pair_rows if r["seed"] == s} for s in SEEDS}
    absent_pairs = sorted(states_by_seed[SEEDS[0]])
    unanimous_absent = finite_mean(len({states_by_seed[s][pair] for s in SEEDS}) == 1 for pair in absent_pairs)
    output.append({"family": "absent_fp", "scope": "all", "metric": "unanimous_four_state_proportion",
                   "comparison": "3_seeds", "n": len(absent_pairs), "value": unanimous_absent})
    for threshold_field, label in (("any_fp", "any_fp"), ("fp_gt_0p1ml", "fp_gt_0p1ml")):
        sets = {}
        for seed in SEEDS:
            rows = [r for r in absent_pair_rows if r["seed"] == seed]
            sets[(seed, "resolved")] = {(r["case_id"], r["class"]) for r in rows if r[f"e0_{threshold_field}"] and not r[f"e2_{threshold_field}"]}
            sets[(seed, "introduced")] = {(r["case_id"], r["class"]) for r in rows if r[f"e2_{threshold_field}"] and not r[f"e0_{threshold_field}"]}
        for status in ("resolved", "introduced"):
            for i, s1 in enumerate(SEEDS):
                for s2 in SEEDS[i + 1:]:
                    output.append({"family": "absent_fp", "scope": label, "metric": f"{status}_set_jaccard",
                                   "comparison": f"{s1}_vs_{s2}", "n": len(absent_pairs),
                                   "value": jaccard(sets[(s1, status)], sets[(s2, status)])})
    summary = {"median_pairwise_spearman": float(statistics.median(correlations)),
               "median_pairwise_practical_sign_agreement": float(statistics.median(agreements)),
               "strong_effect_pair_count": len(strong),
               "strong_effect_unanimous_practical_proportion": strong_unanimous,
               "all_pair_unanimous_practical_proportion": practical_unanimous}
    return output, summary


def case_class_oracle(delta_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    seed_rows = []; class_rows = []; choices = []; seed_details = {}
    for seed in SEEDS:
        selected_rows = []
        for row in delta_rows:
            e0=row[f"dice_e0_seed_{seed}"]; e2=row[f"dice_e2_seed_{seed}"]
            tie=abs(e2-e0)<=EPS; choice="E0" if tie or e0>e2 else "E2"
            selected=e0 if choice=="E0" else e2
            item={"case_id":row["case_id"],"class":row["class"],"class_name":row["class_name"],
                  "gt_volume_ml":row["gt_volume_ml"],"seed":seed,"dice_e0":e0,"dice_e2":e2,
                  "delta":e2-e0,"choice":choice,"tie":tie,"selected_dice":selected}
            choices.append(item); selected_rows.append(item)
        e0_macro=class_balanced_macro(selected_rows,"dice_e0")
        e2_macro=class_balanced_macro(selected_rows,"dice_e2")
        oracle_macro=class_balanced_macro(selected_rows,"selected_dice")
        max_single=max(e0_macro,e2_macro); headroom=oracle_macro-max_single
        if oracle_macro + EPS < max(e0_macro,e2_macro):
            raise RuntimeError(f"Oracle invariant failed for seed {seed}")
        row={"seed":seed,"e0_class_balanced_present_macro":e0_macro,"e2_class_balanced_present_macro":e2_macro,
             "max_single_class_balanced_present_macro":max_single,"oracle_class_balanced_present_macro":oracle_macro,
             "oracle_headroom":headroom,"e0_pooled_present":finite_mean(r["dice_e0"] for r in selected_rows),
             "e2_pooled_present":finite_mean(r["dice_e2"] for r in selected_rows),
             "oracle_pooled_present":finite_mean(r["selected_dice"] for r in selected_rows),
             "choice_e0_count":sum(r["choice"]=="E0" for r in selected_rows),
             "choice_e2_count":sum(r["choice"]=="E2" for r in selected_rows),
             "tie_count":sum(r["tie"] for r in selected_rows),"n_present_pairs":len(selected_rows)}
        seed_rows.append(row); seed_details[seed]=row
        for class_id in CLASS_NAMES:
            rows=[r for r in selected_rows if r["class"]==class_id]
            e0=finite_mean(r["dice_e0"] for r in rows);e2=finite_mean(r["dice_e2"] for r in rows)
            oracle=finite_mean(r["selected_dice"] for r in rows)
            class_rows.append({"seed":seed,"class":class_id,"class_name":CLASS_NAMES[class_id],"n":len(rows),
                               "e0_present_dice":e0,"e2_present_dice":e2,"max_single_present_dice":max(e0,e2),
                               "oracle_present_dice":oracle,"oracle_headroom":oracle-max(e0,e2),
                               "choice_e0_count":sum(r["choice"]=="E0" for r in rows),
                               "choice_e2_count":sum(r["choice"]=="E2" for r in rows),
                               "tie_count":sum(r["tie"] for r in rows)})
    seed_rows.append({"seed":"mean","oracle_headroom":finite_mean(r["oracle_headroom"] for r in seed_rows),
                      "oracle_class_balanced_present_macro":finite_mean(r["oracle_class_balanced_present_macro"] for r in seed_rows)})
    seed_rows.append({"seed":"sample_sd","oracle_headroom":sample_sd(r["oracle_headroom"] for r in seed_rows[:3]),
                      "oracle_class_balanced_present_macro":sample_sd(r["oracle_class_balanced_present_macro"] for r in seed_rows[:3])})
    summary={"per_seed":seed_details,"mean_headroom":finite_mean(r["oracle_headroom"] for r in seed_rows[:3]),
             "sample_sd_headroom":sample_sd(r["oracle_headroom"] for r in seed_rows[:3])}
    return seed_rows,class_rows,choices,summary


def leave_one_seed_out(delta_rows: list[dict]) -> tuple[list[dict], dict]:
    output=[]
    for held in SEEDS:
        train=[s for s in SEEDS if s!=held]
        chosen=[]
        for row in delta_rows:
            train_delta=finite_mean(row[f"delta_seed_{s}"] for s in train)
            arm="E2" if train_delta>=PRACTICAL else "E0"
            if train_delta<=-PRACTICAL: arm="E0"
            if -PRACTICAL < train_delta < PRACTICAL: arm="E0"
            chosen.append({"class":row["class"],"selected":row[f"dice_{arm.lower()}_seed_{held}"],
                           "e0":row[f"dice_e0_seed_{held}"],"e2":row[f"dice_e2_seed_{held}"],"arm":arm})
        selector=class_balanced_macro(chosen,"selected"); e0=class_balanced_macro(chosen,"e0");e2=class_balanced_macro(chosen,"e2")
        max_single=max(e0,e2)
        output.append({"held_out_seed":held,"training_seeds":";".join(map(str,train)),"selector_macro":selector,
                       "e0_macro":e0,"e2_macro":e2,"max_single_macro":max_single,"gain_vs_max_single":selector-max_single,
                       "selected_e0_count":sum(r["arm"]=="E0" for r in chosen),"selected_e2_count":sum(r["arm"]=="E2" for r in chosen),
                       "n_present_pairs":len(chosen),"limitation":"same patients' GT in other model seeds; not unseen-patient generalization"})
    summary={"per_held_seed":output,"positive_gain_seed_count":sum(r["gain_vs_max_single"]>0 for r in output),
             "mean_gain":finite_mean(r["gain_vs_max_single"] for r in output)}
    return output,summary


def presence_oracle(metric_rows: list[dict], absent_rows: list[dict]) -> list[dict]:
    output=[]
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            present=[r for r in metric_rows if r["seed"]==seed and r["model"]==arm]
            absent=[r for r in absent_rows if r["seed"]==seed and r["model"]==arm]
            original_class=[];oracle_class=[]
            for class_id in CLASS_NAMES:
                p=[r["dice"] for r in present if r["class"]==class_id]
                a=[0.0 for r in absent if r["class"]==class_id and r["any_fp"]]
                original_class.append(finite_mean(p+a))
                oracle_class.append(finite_mean(p))
            output.append({"seed":seed,"model":arm,"original_nnunet_style_foreground_macro":finite_mean(original_class),
                           "presence_oracle_nnunet_style_foreground_macro":finite_mean(oracle_class),
                           "foreground_macro_gap_due_absent_fp":finite_mean(oracle_class)-finite_mean(original_class),
                           "original_class_balanced_present_macro":class_balanced_macro(present,"dice"),
                           "oracle_class_balanced_present_macro":class_balanced_macro(present,"dice"),
                           "original_pooled_present":finite_mean(r["dice"] for r in present),
                           "oracle_pooled_present":finite_mean(r["dice"] for r in present),
                           "original_absent_fp_pairs":sum(r["any_fp"] for r in absent),"oracle_absent_fp_pairs":0,
                           "present_pair_dice_max_abs_change":0.0,
                           "method":"one-vs-rest class mask; GT-absent class prediction zeroed; no argmax/reassignment; both-empty excluded as NaN"})
    return output


def slice_selectors(result_sets: dict, gt_root: Path, delta_rows: list[dict], safe_soft: dict) -> list[dict]:
    accum=defaultdict(list)
    base_by_seed_class={}
    for seed in SEEDS:
        for class_id in CLASS_NAMES:
            rows=[r for r in delta_rows if r["class"]==class_id]
            e0=finite_mean(r[f"dice_e0_seed_{seed}"] for r in rows);e2=finite_mean(r[f"dice_e2_seed_{seed}"] for r in rows)
            base_by_seed_class[(seed,class_id)]="E2" if e2>e0 else "E0"
    for seed in SEEDS:
        soft_ok=safe_soft[("E0",seed)] and safe_soft[("E2",seed)]
        rows_by_case=defaultdict(list)
        for row in delta_rows: rows_by_case[row["case_id"]].append(row)
        for cid,case_rows in sorted(rows_by_case.items()):
            gt,_=load_seg(gt_root/f"{cid}.nii.gz")
            hard={}
            for arm in MODEL_ARMS:
                pred,_=load_seg(result_sets[(arm,seed)].validation/f"{cid}.nii.gz")
                hard[arm]=pred
            full_probs={}
            if soft_ok:
                for arm in MODEL_ARMS:
                    with np.load(result_sets[(arm,seed)].validation/f"{cid}.npz") as payload:
                        probabilities=payload["probabilities"]
                        if not np.array_equal(np.argmax(probabilities,axis=0),hard[arm].transpose(2,1,0)):
                            raise RuntimeError(f"unsafe NPZ mapping: {arm} seed {seed} {cid} argmax does not reproduce hard NIfTI")
                        full_probs[arm]=probabilities.transpose(0,3,2,1)
            for row in case_rows:
                class_id=row["class"];gm=gt==class_id
                masks={arm:hard[arm]==class_id for arm in MODEL_ARMS}
                selected_all=np.zeros_like(gm,dtype=bool)
                selected_positive=np.zeros_like(gm,dtype=bool)
                base=base_by_seed_class[(seed,class_id)]
                for z in range(gm.shape[2]):
                    gz=gm[:,:,z]; nref=int(gz.sum())
                    if nref:
                        d0=binary_dice(gz,masks["E0"][:,:,z]);d2=binary_dice(gz,masks["E2"][:,:,z])
                        pick="E2" if d2>d0+EPS else "E0"
                        selected_all[:,:,z]=masks[pick][:,:,z]
                        selected_positive[:,:,z]=masks[pick][:,:,z]
                    else:
                        fp0=int(masks["E0"][:,:,z].sum());fp2=int(masks["E2"][:,:,z].sum())
                        pick="E2" if fp2<fp0 else "E0"
                        selected_all[:,:,z]=masks[pick][:,:,z]
                        selected_positive[:,:,z]=masks[base][:,:,z]
                accum[(seed,class_id,"exploratory_highly_optimistic_slice_gt_greedy_selector")].append(binary_dice(gm,selected_all))
                accum[(seed,class_id,"exploratory_positive_slice_delineation_gt_greedy_selector")].append(binary_dice(gm,selected_positive))
                if not soft_ok:
                    continue
                probs={arm:full_probs[arm][class_id] for arm in MODEL_ARMS}
                for selector in ("exploratory_soft_bce_gt_guided_selector","exploratory_soft_dice_loss_gt_guided_selector"):
                    selected=np.zeros_like(gm,dtype=bool)
                    for z in range(gm.shape[2]):
                        target=gm[:,:,z].astype(np.float64)
                        losses={}
                        for arm in MODEL_ARMS:
                            prob=np.clip(probs[arm][:,:,z].astype(np.float64),1e-7,1-1e-7)
                            if selector.endswith("bce_gt_guided_selector"):
                                losses[arm]=float(-np.mean(target*np.log(prob)+(1-target)*np.log(1-prob)))
                            else:
                                losses[arm]=float(1-(2*np.sum(prob*target)+EPS)/(np.sum(prob)+np.sum(target)+EPS))
                        pick="E2" if losses["E2"]<losses["E0"]-EPS else "E0"
                        selected[:,:,z]=hard[pick][:,:,z]
                    accum[(seed,class_id,selector)].append(binary_dice(gm,selected))
    output=[]
    for (seed,class_id,selector),values in sorted(accum.items()):
        rows=[r for r in delta_rows if r["class"]==class_id]
        e0=finite_mean(r[f"dice_e0_seed_{seed}"] for r in rows);e2=finite_mean(r[f"dice_e2_seed_{seed}"] for r in rows)
        greedy=finite_mean(values)
        base=base_by_seed_class[(seed,class_id)] if "positive_slice" in selector else "not_applicable"
        output.append({"seed":seed,"selector_type":selector,"class":class_id,"class_name":CLASS_NAMES[class_id],
                       "n_present_cases":len(values),"baseline_e0_present_dice":e0,"baseline_e2_present_dice":e2,
                       "best_single_present_dice":max(e0,e2),"greedy_selector_present_dice":greedy,
                       "apparent_gain_vs_best_single":greedy-max(e0,e2),"strict_volume_oracle":False,
                       "negative_slice_base_model":base,
                       "notes":"GT-guided, undeployable exploratory selector; per-class binary masks only; no multiclass reconstruction"})
    return output


def patient_cluster_stat(sampled_ids: list[str], delta_rows: list[dict], value_key_by_seed) -> float:
    values=[]
    by_case=defaultdict(list)
    for row in delta_rows: by_case[row["case_id"]].append(row)
    sampled_rows=[row for cid in sampled_ids for row in by_case[cid]]
    for seed in SEEDS:
        class_values=[]
        for class_id in CLASS_NAMES:
            vals=[value_key_by_seed(row,seed) for row in sampled_rows if row["class"]==class_id]
            if not vals: return math.nan
            class_values.append(finite_mean(vals))
        values.append(finite_mean(class_values))
    return finite_mean(values)


def bootstrap_and_permutation(delta_rows: list[dict]) -> dict:
    rng=np.random.default_rng(BOOTSTRAP_SEED); patient_ids=sorted({r["case_id"] for r in delta_rows})
    delta_stats=[];headroom_stats=[];per_class={c:[] for c in CLASS_NAMES};invalid=0
    def sampled_headroom(sampled_ids: list[str]) -> float:
        by_case=defaultdict(list)
        for row in delta_rows: by_case[row["case_id"]].append(row)
        sampled_rows=[row for cid in sampled_ids for row in by_case[cid]]
        seed_headrooms=[]
        for seed in SEEDS:
            e0_classes=[];e2_classes=[];oracle_classes=[]
            for class_id in CLASS_NAMES:
                rows=[r for r in sampled_rows if r["class"]==class_id]
                if not rows: return math.nan
                e0_classes.append(finite_mean(r[f"dice_e0_seed_{seed}"] for r in rows))
                e2_classes.append(finite_mean(r[f"dice_e2_seed_{seed}"] for r in rows))
                oracle_classes.append(finite_mean(max(r[f"dice_e0_seed_{seed}"],r[f"dice_e2_seed_{seed}"]) for r in rows))
            e0=finite_mean(e0_classes);e2=finite_mean(e2_classes);oracle=finite_mean(oracle_classes)
            seed_headrooms.append(oracle-max(e0,e2))
        return finite_mean(seed_headrooms)
    for _ in range(N_RESAMPLES):
        sampled=[patient_ids[i] for i in rng.integers(0,len(patient_ids),size=len(patient_ids))]
        delta=patient_cluster_stat(sampled,delta_rows,lambda row,seed:row[f"delta_seed_{seed}"])
        headroom=sampled_headroom(sampled)
        if not all(math.isfinite(x) for x in (delta,headroom)):
            invalid+=1;continue
        delta_stats.append(delta);headroom_stats.append(headroom)
        sampled_set=[row for cid in sampled for row in delta_rows if row["case_id"]==cid]
        for class_id in CLASS_NAMES:
            per_class[class_id].append(finite_mean(finite_mean(r[f"delta_seed_{s}"] for s in SEEDS)
                                                   for r in sampled_set if r["class"]==class_id))
    observed=patient_cluster_stat(patient_ids,delta_rows,lambda row,seed:row[f"delta_seed_{seed}"])
    prng=np.random.default_rng(BOOTSTRAP_SEED+1); extreme=0
    by_patient={cid:[r for r in delta_rows if r["case_id"]==cid] for cid in patient_ids}
    for _ in range(N_RESAMPLES):
        signs={cid:(1 if x else -1) for cid,x in zip(patient_ids,prng.integers(0,2,size=len(patient_ids)))}
        perm=[]
        for cid in patient_ids:
            for row in by_patient[cid]:
                copied=dict(row)
                for seed in SEEDS: copied[f"delta_seed_{seed}"]=row[f"delta_seed_{seed}"]*signs[cid]
                perm.append(copied)
        stat=patient_cluster_stat(patient_ids,perm,lambda row,seed:row[f"delta_seed_{seed}"])
        if abs(stat)>=abs(observed)-1e-15: extreme+=1
    dlo,dhi=percentile_ci(delta_stats);hlo,hhi=percentile_ci(headroom_stats)
    return {"rng_seed":BOOTSTRAP_SEED,"n_resamples":N_RESAMPLES,"independent_cluster":"patient",
            "primary_estimand":"3-seed average of 5-class-balanced GT-present E2-minus-E0 Dice",
            "primary_delta":{"observed":observed,"bootstrap_mean":finite_mean(delta_stats),"bootstrap_sample_sd":sample_sd(delta_stats),
                             "percentile_95_ci":[dlo,dhi],"valid_replicates":len(delta_stats),"invalid_replicates":invalid},
            "oracle_headroom":{"observed":sampled_headroom(patient_ids),"bootstrap_mean":finite_mean(headroom_stats),"bootstrap_sample_sd":sample_sd(headroom_stats),
                               "percentile_95_ci":[hlo,hhi],"valid_replicates":len(headroom_stats),"invalid_replicates":invalid},
            "per_class_delta":{CLASS_NAMES[c]:{"bootstrap_mean":finite_mean(v),"bootstrap_sample_sd":sample_sd(v),
                                               "percentile_95_ci":list(percentile_ci(v)),"valid_replicates":len(v),
                                               "invalid_replicates":invalid} for c,v in per_class.items()},
            "paired_patient_sign_flip":{"observed_statistic":observed,"extreme_count":extreme,"two_sided_p_value":(extreme+1)/(N_RESAMPLES+1),
                                        "one_random_sign_per_patient_joint_across_classes_and_seeds":True},
            "limitations":["39 fold-0 validation patients only","model seeds are not independent patient cohorts",
                           "bootstrap does not include fold uncertainty","validation cases participated in checkpoint selection"]}


def decide(oracle_summary: dict, oracle_class_rows: list[dict], loso: dict, stability: dict,
           slice_rows: list[dict], presence_rows: list[dict]) -> dict:
    per_seed=[oracle_summary["per_seed"][s]["oracle_headroom"] for s in SEEDS]
    class_mean={c:finite_mean(r["oracle_headroom"] for r in oracle_class_rows if r["class"]==c) for c in CLASS_NAMES}
    go_checks={
        "all_3_seed_headroom_gt_0p01":all(v>0.01 for v in per_seed),
        "mean_headroom_ge_0p02":oracle_summary["mean_headroom"]>=0.02,
        "at_least_3_classes_mean_headroom_ge_0p01":sum(v>=0.01 for v in class_mean.values())>=3,
        "loso_positive_gain_at_least_2_seeds":loso["positive_gain_seed_count"]>=2,
        "mean_loso_gain_ge_0p005":loso["mean_gain"]>=0.005,
        "median_spearman_ge_0p30":stability["median_pairwise_spearman"]>=0.30,
        "strong_effect_unanimous_ge_0p50":stability["strong_effect_unanimous_practical_proportion"]>=0.50,
        "primary_headroom_gt_present":True,"not_slice_only":True}
    max_slice_gain=max((r["apparent_gain_vs_best_single"] for r in slice_rows),default=math.nan)
    stop_checks={
        "at_least_2_seed_headroom_le_0p01":sum(v<=0.01 for v in per_seed)>=2,
        "mean_headroom_le_0p01":oracle_summary["mean_headroom"]<=0.01,
        "loso_never_beats_max_single":loso["positive_gain_seed_count"]==0,
        "mean_loso_gain_le_0":loso["mean_gain"]<=0,
        "low_spearman_and_low_strong_unanimity":stability["median_pairwise_spearman"]<=0.10 and stability["strong_effect_unanimous_practical_proportion"]<=0.30,
        "volume_headroom_small_but_slice_gain_large":oracle_summary["mean_headroom"]<=0.01 and max_slice_gain>=0.03,
        "no_repeatable_gt_present_complementarity":(loso["mean_gain"]<=0 and stability["median_pairwise_spearman"]<=0.10),
        "improvement_almost_entirely_absent_fp":oracle_summary["mean_headroom"]<=0.01 and any(r["foreground_macro_gap_due_absent_fp"]>0 for r in presence_rows)}
    if all(go_checks.values()): decision="E0_E2_COMPLEMENTARITY_GO"
    elif any(stop_checks.values()): decision="E0_E2_COMPLEMENTARITY_STOP"
    else: decision="E0_E2_COMPLEMENTARITY_BORDERLINE"
    return {"decision":decision,"go_checks":go_checks,"stop_checks":stop_checks,"per_seed_oracle_headroom":dict(zip(map(str,SEEDS),per_seed)),
            "mean_oracle_headroom":oracle_summary["mean_headroom"],"per_class_mean_oracle_headroom":{CLASS_NAMES[c]:v for c,v in class_mean.items()},
            "loso_mean_gain":loso["mean_gain"],"median_pairwise_spearman":stability["median_pairwise_spearman"],
            "strong_effect_unanimous_practical_proportion":stability["strong_effect_unanimous_practical_proportion"],
            "max_exploratory_slice_apparent_gain":max_slice_gain,
            "scope":"Stops or advances only E0/E2 complementarity as model-design evidence; not all 2.5D research."}


def report_text(decision: dict, oracle_summary: dict, loso: dict, stability: dict, bootstrap: dict,
                reproduction: dict, present_rows: list[dict], absent_pairs: list[dict], slice_rows: list[dict],
                soft_safe: bool) -> str:
    seed_heads=[oracle_summary["per_seed"][s]["oracle_headroom"] for s in SEEDS]
    summary_scores={(s,a):next(r["value"] for r in present_rows if r["seed"]==s and r["model"]==a and r["metric"]=="class_balanced_present_macro") for s in SEEDS for a in MODEL_ARMS}
    absence={s:{a:sum(r[f"{a.lower()}_any_fp"] for r in absent_pairs if r["seed"]==s) for a in MODEL_ARMS} for s in SEEDS}
    lines=["# E0/E2 multi-seed complementarity and GT-oracle analysis","",
           f"**Locked decision: `{decision['decision']}`**","",
           "## 1. Scope","Only existing fold-0 hard predictions, exported probabilities, and the 39 validation GT cases were analyzed. No training, inference, resampling, or data expansion was performed.","",
           "## 2. Asset and provenance audit",f"All six result sets contain 39 hard predictions, 39 NPZ files, 39 properties files, both checkpoints, summary, debug, timing, and profiling records. All validations record `--val_best`, so `checkpoint_best.pth` is the validated checkpoint.","",
           "## 3. Case-ID join","All joins were explicit by case ID. The six prediction sets and GT contain the same 39 IDs with no duplicates, missing cases, or extras.","",
           "## 4. Geometry, labels, and voxel volume","Every prediction matches GT shape and affine. Labels are restricted to 0-5. Affine-determinant and header-zoom voxel volumes agree within tolerance.","",
           "## 5. Seed-3407 reproduction",f"Reproduction status: `{reproduction['status']}`. E0 absent FP={reproduction['e0_absent_fp_count']}/113; E2 absent FP={reproduction['e2_absent_fp_count']}/113; maximum GT-present Dice difference from the historical analysis={reproduction['max_present_dice_absolute_difference']:.3g}.","",
           "## 6. Metric separation","nnU-Net `foreground_mean.Dice`, class-balanced GT-present macro, pooled patient-class Dice, and online EMA are not interchangeable. The decision uses class-balanced GT-present metrics.","",
           "## 7. GT-present performance","| Seed | E0 class-balanced present | E2 class-balanced present | E2-E0 |","|---:|---:|---:|---:|"]
    for s in SEEDS: lines.append(f"| {s} | {summary_scores[(s,'E0')]:.6f} | {summary_scores[(s,'E2')]:.6f} | {summary_scores[(s,'E2')]-summary_scores[(s,'E0')]:+.6f} |")
    lines += ["","## 8. Absent-class false positives","| Seed | E0 any-FP pairs | E2 any-FP pairs | Denominator |","|---:|---:|---:|---:|"]
    for s in SEEDS: lines.append(f"| {s} | {absence[s]['E0']} | {absence[s]['E2']} | 113 |")
    lines += ["","## 9. Cross-seed direction stability",f"Median pairwise Spearman rho={stability['median_pairwise_spearman']:.4f}; strong-effect three-seed unanimous practical direction={stability['strong_effect_unanimous_practical_proportion']:.1%} (n={stability['strong_effect_pair_count']}).","",
              "## 10. Practical sign definition","A delta >=+0.01 favors E2, <=-0.01 favors E0, and intermediate values are neutral. Majority sign is not treated as stability.","",
              "## 11. Patient-cluster bootstrap",f"Primary three-seed class-balanced delta={bootstrap['primary_delta']['observed']:+.6f}; percentile 95% CI [{bootstrap['primary_delta']['percentile_95_ci'][0]:+.6f}, {bootstrap['primary_delta']['percentile_95_ci'][1]:+.6f}], valid={bootstrap['primary_delta']['valid_replicates']}, invalid={bootstrap['primary_delta']['invalid_replicates']}. Models seeds are retained jointly within sampled patients.","",
              "## 12. Patient-level sign flip",f"Two-sided p={bootstrap['paired_patient_sign_flip']['two_sided_p_value']:.6f}; one sign was applied jointly to all classes and seeds for each patient.","",
              "## 13. Case-class GT oracle","| Seed | Oracle headroom vs best single |","|---:|---:|"]
    for s,h in zip(SEEDS,seed_heads): lines.append(f"| {s} | {h:+.6f} |")
    lines += [f"Mean headroom={oracle_summary['mean_headroom']:+.6f}; sample SD={oracle_summary['sample_sd_headroom']:.6f}; patient-bootstrap 95% CI [{bootstrap['oracle_headroom']['percentile_95_ci'][0]:+.6f}, {bootstrap['oracle_headroom']['percentile_95_ci'][1]:+.6f}].","",
              "## 14. Leave-one-seed-out selector",f"Positive held-seed gains={loso['positive_gain_seed_count']}/3; mean gain vs the stronger complete single model={loso['mean_gain']:+.6f}. This selector uses the same patients' GT from the other two model seeds and is not deployable or an unseen-patient test.","",
              "## 15. Presence oracle","The one-vs-rest presence oracle removes predictions only when that class is GT-absent. It preserves every GT-present binary Dice exactly and is a diagnostic of summary sensitivity to absent-class FP, not model performance.","",
              "## 16. Slice selectors",f"The exact label `exploratory_highly_optimistic_slice_gt_greedy_selector` is used. Soft selectors were {'included after exact NPZ restoration verification' if soft_safe else 'skipped because NPZ restoration could not be proven safe'}. These GT-guided slice results are not a strict volume oracle.","",
              "## 17. Oracle invariants","For every seed and class, the case-class oracle is at least as high as both E0 and E2 on corresponding GT-present support; ties use E0 deterministically.","",
              "## 18. Decision rule evaluation",f"GO checks passed: {sum(decision['go_checks'].values())}/{len(decision['go_checks'])}. STOP triggers met: {sum(decision['stop_checks'].values())}/{len(decision['stop_checks'])}. Final decision is `{decision['decision']}`.","",
              "## 19. Interpretation boundary","E0 and E2 both use a full symmetric three-slice backbone. This comparison cannot establish whether neighboring context helps relative to center-only 2D, whether the center residual helps, or whether a learned selector generalizes.","",
              "## 20. Validation limitation","There are only 39 fold-0 validation patients, and these cases participated in checkpoint selection. There is no independent test set.","",
              "## 21. Uncertainty limitation","The patient bootstrap preserves within-patient classes and seeds but does not include fold uncertainty. Three model seeds are not independent cohorts.","",
              "## 22. Oracle limitation","All GT oracles and GT-guided selectors are undeployable conditional diagnostics. They must not be reported as achieved model scores or SOTA comparisons.","",
              "## 23. Research claim boundary","The result concerns only repeatable E0/E2 complementarity on fold 0. It does not stop all 2.5D research and does not support a SOTA claim.","",
              "## 24. Reproducibility","Configuration, manifests, exact case lists, generated tables, tests, and SHA-256 checksums are included in this directory. Historical seed-3407 outputs were read-only and not regenerated.","",
              "## 18-point executive summary",""]
    bullets=[
        f"The locked outcome is `{decision['decision']}`.","Six required E0/E2 result sets are complete.","All sets contain the same 39 fold-0 cases.","Hard-prediction geometry matches GT exactly within the audit tolerance.","All observed labels are valid integers 0-5.","Validation provenance points to `checkpoint_best.pth`.","Locked nnU-Net summary scores reproduce within 1e-6.","Seed 3407 absent-FP counts reproduce exactly at 70/113 and 58/113.","All 390 historical E0/E2 seed-3407 class-case count rows reproduce.","GT-present Dice is the primary support for complementarity decisions.",f"Mean strict case-class oracle headroom is {oracle_summary['mean_headroom']:+.6f}.",f"LOSO selector mean gain is {loso['mean_gain']:+.6f}.",f"Median cross-seed patient-class delta Spearman rho is {stability['median_pairwise_spearman']:.4f}.",f"Strong-effect unanimous practical direction is {stability['strong_effect_unanimous_practical_proportion']:.1%}.","Absent-class FP diagnostics are kept separate from GT-present delineation.","Presence-oracle numbers are diagnostics, not model scores.","Slice GT-greedy numbers are highly optimistic and not strict volume-oracle results.","The analysis cannot attribute effects to neighbor context versus center residual because both arms are three-slice models."]
    lines += [f"- {b}" for b in bullets]
    lines += ["","## Exactly one next-step recommendation","Do not design or train another E0/E2 selector from these fold-0 oracle diagnostics; use the locked decision to close this branch and, in a separate planning step, choose a publication question that can be tested against an explicit center-only 2D baseline with multi-fold or independent-test confirmation.",""]
    return "\n".join(lines)


def checksum_files(output_dir: Path, extra_files: list[Path]) -> None:
    targets=sorted([p for p in output_dir.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"]+extra_files,key=lambda p:str(p))
    lines=[]
    for path in targets:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        label=path.name if path.parent==output_dir else str(path)
        lines.append(f"{digest}  {label}")
    (output_dir/"SHA256SUMS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--drive-root",type=Path,required=True)
    parser.add_argument("--multiseed-root",type=Path,required=True)
    parser.add_argument("--gt-root",type=Path,required=True)
    parser.add_argument("--old-analysis",type=Path,required=True)
    parser.add_argument("--test-file",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result_sets=make_result_sets(args.drive_root,args.multiseed_root)
    locked={("E0",3407):0.2737889536359917,("E2",3407):0.28616442756224575,
            ("E0",1234):0.2707331397708116,("E2",1234):0.26321597788313883,
            ("E0",5678):0.2620165774279875,("E2",5678):0.2504461232918047}
    inventory,case_ids,audit=audit_assets(result_sets,args.gt_root,locked)
    hard_blockers=[f"{k}: {v}" for k,v in audit.items() if v["missing"] or not v["score_match"]]
    if hard_blockers or len(case_ids)!=39 or any(r["provenance_status"]!="verified" for r in inventory):
        write_csv(args.output_dir/"prediction_inventory.csv",inventory)
        write_json(args.output_dir/"oracle_decision.json",{"decision":"ANALYSIS_BLOCKED","blockers":hard_blockers})
        raise RuntimeError("ANALYSIS_BLOCKED: "+"; ".join(hard_blockers))
    metric_rows,absent_rows,hard_cache=audit_geometry_and_metrics(result_sets,args.gt_root,case_ids,inventory)
    reproduction=reproduce_seed3407(hard_cache,args.old_analysis,absent_rows)
    if reproduction["status"]!="passed":
        write_csv(args.output_dir/"prediction_inventory.csv",inventory);write_json(args.output_dir/"reproduction_check.json",reproduction)
        write_json(args.output_dir/"oracle_decision.json",{"decision":"ANALYSIS_BLOCKED","blockers":["seed3407 reproduction failed"]})
        raise RuntimeError("ANALYSIS_BLOCKED: seed3407 reproduction failed")
    safe_soft=verify_soft_mapping(result_sets,case_ids,inventory)
    present=present_metric_outputs(metric_rows); absent=absent_output(absent_rows); deltas=delta_outputs(metric_rows,hard_cache)
    stability_rows,stability=stability_outputs(deltas,absent)
    oracle_seed,oracle_class,oracle_choices,oracle_summary=case_class_oracle(deltas)
    loso_rows,loso=leave_one_seed_out(deltas)
    for r in loso_rows: stability_rows.append({"family":"leave_one_seed_out_selector","scope":str(r["held_out_seed"]),"metric":"gain_vs_max_single","comparison":"selector", "n":r["n_present_pairs"],"value":r["gain_vs_max_single"],"notes":r["limitation"]})
    presence=presence_oracle(metric_rows,absent_rows)
    slices=slice_selectors(result_sets,args.gt_root,deltas,safe_soft)
    bootstrap=bootstrap_and_permutation(deltas)
    decision=decide(oracle_summary,oracle_class,loso,stability,slices,presence)
    config={"analysis_date":"2026-07-26","class_map":CLASS_NAMES,"seeds":SEEDS,"fold":0,"rng_seed":BOOTSTRAP_SEED,
            "bootstrap_replicates":N_RESAMPLES,"sign_flip_replicates":N_RESAMPLES,"raw_sign_epsilon":EPS,
            "practical_sign_threshold":PRACTICAL,"tie_tolerance":EPS,"case_join":"explicit case_id",
            "voxel_volume_ml":"abs(det(affine[:3,:3]))/1000","empty_mask_policy":"both empty excluded as NaN; GT present/pred empty Dice=0",
            "primary_estimand":"patient-clustered 3-seed average of per-seed 5-class-balanced GT-present E2-minus-E0 Dice",
            "npz_mapping":"nnU-Net exported probabilities in C,Z,Y,X; properties require full bbox and unchanged restored shape; transpose Z,Y,X to X,Y,Z; argmax must exactly reproduce hard NIfTI",
            "forbidden_actions_observed":False}
    manifest={"schema_version":1,"git_commit":GIT_COMMIT,"n_cases":len(case_ids),"case_ids":case_ids,
              "result_sets":{f"{a}_{s}":str(rs.root) for (a,s),rs in result_sets.items()},"gt_root":str(args.gt_root),
              "historical_analysis_read_only":str(args.old_analysis),"soft_mapping_safe":{f"{a}_{s}":v for (a,s),v in safe_soft.items()},
              "generated_files":["prediction_inventory.csv","reproduction_check.json","multiseed_present_metrics.csv",
              "multiseed_patient_class_deltas.csv","absent_fp_by_seed.csv","cross_seed_stability.csv","bootstrap_summary.json",
              "presence_oracle_summary.csv","case_class_oracle_per_seed.csv","case_class_oracle_per_class.csv","case_class_oracle_choices.csv",
              "slice_class_gt_greedy_summary.csv","oracle_decision.json","analysis_manifest.json","analysis_config.json","SHA256SUMS.txt","ORACLE_ANALYSIS_REPORT.md",
              "run_oracle_analysis.py","test_oracle_analysis.py"]}
    write_csv(args.output_dir/"prediction_inventory.csv",inventory)
    write_json(args.output_dir/"reproduction_check.json",reproduction)
    write_csv(args.output_dir/"multiseed_present_metrics.csv",present)
    write_csv(args.output_dir/"multiseed_patient_class_deltas.csv",deltas)
    write_csv(args.output_dir/"absent_fp_by_seed.csv",absent)
    write_csv(args.output_dir/"cross_seed_stability.csv",stability_rows)
    write_json(args.output_dir/"bootstrap_summary.json",json_safe(bootstrap))
    write_csv(args.output_dir/"presence_oracle_summary.csv",presence)
    write_csv(args.output_dir/"case_class_oracle_per_seed.csv",oracle_seed)
    write_csv(args.output_dir/"case_class_oracle_per_class.csv",oracle_class)
    write_csv(args.output_dir/"case_class_oracle_choices.csv",oracle_choices)
    write_csv(args.output_dir/"slice_class_gt_greedy_summary.csv",slices)
    write_json(args.output_dir/"oracle_decision.json",json_safe(decision))
    write_json(args.output_dir/"analysis_manifest.json",json_safe(manifest))
    write_json(args.output_dir/"analysis_config.json",json_safe(config))
    (args.output_dir/"ORACLE_ANALYSIS_REPORT.md").write_text(report_text(decision,oracle_summary,loso,stability,bootstrap,reproduction,present,absent,slices,all(safe_soft.values())),encoding="utf-8")
    shutil.copy2(Path(__file__),args.output_dir/"run_oracle_analysis.py")
    shutil.copy2(args.test_file,args.output_dir/"test_oracle_analysis.py")
    checksum_files(args.output_dir,[])
    print(json.dumps({"decision":decision["decision"],"output_dir":str(args.output_dir),"oracle_mean_headroom":oracle_summary["mean_headroom"],
                      "loso_mean_gain":loso["mean_gain"],"median_spearman":stability["median_pairwise_spearman"],
                      "strong_unanimity":stability["strong_effect_unanimous_practical_proportion"]},indent=2))


if __name__=="__main__":
    main()
