from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
PREPROCESSED = REPO / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset001_BHSD"
RAW = REPO / "nnUNet_data" / "nnUNet_raw" / "Dataset001_BHSD"
GT = PREPROCESSED / "gt_segmentations"
PREDICTION_DIR = Path(
    r"D:\BHSD_server_backups\SRG_SF_direction_results_2026-07-24"
    r"\nnUNetTrainer_25D_SymmetricE0Control__nnUNetPlans__2d\fold_0\validation"
)

SUMMARY_ROOTS = [
    Path(r"D:\BHSD_server_backups\multiclass_2d_min300_patience100"),
    Path(r"D:\BHSD_server_backups\multiclass_3d_min300_patience100"),
    Path(r"D:\BHSD_server_backups\binary_2d_min300_patience100"),
    Path(r"D:\BHSD_server_backups\binary_3d_min300_patience100"),
    Path(r"D:\BHSD_server_backups\multiclass_25d_3slice_fold0_min300_patience100"),
    Path(r"D:\BHSD_server_backups\binary_25d_3slice_fold0_min300_patience100"),
    Path(
        r"C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups"
        r"\multiclass_25d_attention_screen_partial_2026-07-22"
    ),
    Path(r"D:\BHSD_server_backups\multiclass_25d_controlled_screens_2026-07-22"),
    Path(r"D:\BHSD_server_backups\SRG_SF_direction_results_2026-07-24"),
    Path(
        r"C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups"
        r"\controlled_multiseed_fold0_2026-07-24"
    ),
]


def case_id(path_value: str) -> str:
    name = Path(path_value).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def image_geometry(image: sitk.Image) -> dict:
    return {
        "size_xyz": tuple(int(v) for v in image.GetSize()),
        "spacing_xyz": tuple(float(v) for v in image.GetSpacing()),
        "origin_xyz": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    ga, gb = image_geometry(a), image_geometry(b)
    return (
        ga["size_xyz"] == gb["size_xyz"]
        and np.allclose(ga["spacing_xyz"], gb["spacing_xyz"], atol=1e-6, rtol=0)
        and np.allclose(ga["origin_xyz"], gb["origin_xyz"], atol=1e-5, rtol=0)
        and np.allclose(ga["direction"], gb["direction"], atol=1e-6, rtol=0)
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_splits() -> tuple[list[dict], dict, list[dict]]:
    splits = json.loads((PREPROCESSED / "splits_final.json").read_text(encoding="utf-8"))
    raw_images = {p.name[:-12] for p in (RAW / "imagesTr").glob("*_0000.nii.gz")}
    raw_labels = {p.name[:-7] for p in (RAW / "labelsTr").glob("*.nii.gz")}
    assert raw_images == raw_labels and len(raw_images) == 192
    val_counter: Counter[str] = Counter()
    rows = []
    for fold, split in enumerate(splits):
        train, val = set(split["train"]), set(split["val"])
        val_counter.update(val)
        rows.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_val": len(val),
                "train_val_overlap": len(train & val),
                "union_matches_192_raw_cases": train | val == raw_images,
                "status": "PASS" if not (train & val) and train | val == raw_images else "FAIL",
            }
        )
    exactly_once = set(val_counter) == raw_images and set(val_counter.values()) == {1}
    assert len(splits) == 5 and exactly_once and all(r["status"] == "PASS" for r in rows)

    properties = []
    for path in sorted((PREPROCESSED / "nnUNetPlans_2d").glob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        present_classes = sorted(
            int(label)
            for label, locations in payload.get("class_locations", {}).items()
            if isinstance(label, int) and label > 0 and len(locations) > 0
        )
        properties.append(
            {
                "case_id": path.stem,
                "z_slices": int(payload["shape_before_cropping"][0]),
                "present_classes": present_classes,
                "n_present_classes": len(present_classes),
            }
        )
    smallest = sorted(properties, key=lambda r: (r["z_slices"], r["case_id"]))[:10]
    multiclass = sorted(properties, key=lambda r: (-r["n_present_classes"], r["case_id"]))
    selected = list(smallest)
    for row in multiclass:
        if row["case_id"] not in {v["case_id"] for v in selected}:
            selected.append(row)
        if len(selected) == 20:
            break
    assert len(selected) == 20
    split_summary = {
        "n_cases": len(raw_images),
        "n_folds": len(splits),
        "every_case_is_validation_exactly_once": exactly_once,
        "raw_image_and_label_case_ids_identical": raw_images == raw_labels,
        "selected_real_case_count": len(selected),
    }
    return rows, split_summary, selected


def audit_real_cases(selected: list[dict], splits: list[dict]) -> list[dict]:
    fold_by_case = {case: fold for fold, split in enumerate(splits) for case in split["val"]}
    rows = []
    for item in selected:
        cid = item["case_id"]
        image = sitk.ReadImage(str(RAW / "imagesTr" / f"{cid}_0000.nii.gz"))
        label = sitk.ReadImage(str(RAW / "labelsTr" / f"{cid}.nii.gz"))
        label_array = sitk.GetArrayFromImage(label)
        labels = sorted(int(v) for v in np.unique(label_array))
        geometry_ok = same_geometry(image, label)
        labels_ok = set(labels) <= set(range(6))
        assert geometry_ok and labels_ok
        rows.append(
            {
                "case_id": cid,
                "selection_reason": "small_z" if item in selected[:10] else "many_present_classes",
                "validation_fold": fold_by_case[cid],
                "z_slices": int(label_array.shape[0]),
                "present_classes_from_properties": "|".join(map(str, item["present_classes"])),
                "observed_labels": "|".join(map(str, labels)),
                "image_label_geometry_match": geometry_ok,
                "labels_subset_0_5": labels_ok,
                "boundary_indices_for_3slice": "0->0|0|1;last->last-1|last|last",
                "status": "PASS",
            }
        )
    return rows


def audit_prediction_geometry(expected_fold0: set[str]) -> list[dict]:
    prediction_ids = {p.name[:-7] for p in PREDICTION_DIR.glob("*.nii.gz")}
    assert prediction_ids == expected_fold0
    rows = []
    for cid in sorted(prediction_ids)[:20]:
        pred = sitk.ReadImage(str(PREDICTION_DIR / f"{cid}.nii.gz"))
        gt = sitk.ReadImage(str(GT / f"{cid}.nii.gz"))
        pred_values = sorted(int(v) for v in np.unique(sitk.GetArrayFromImage(pred)))
        gt_values = sorted(int(v) for v in np.unique(sitk.GetArrayFromImage(gt)))
        geometry_ok = same_geometry(pred, gt)
        labels_ok = set(pred_values) <= set(range(6)) and set(gt_values) <= set(range(6))
        assert geometry_ok and labels_ok
        rows.append(
            {
                "case_id": cid,
                "prediction_gt_geometry_match": geometry_ok,
                "prediction_labels": "|".join(map(str, pred_values)),
                "gt_labels": "|".join(map(str, gt_values)),
                "labels_subset_0_5": labels_ok,
                "case_id_in_fold0_validation": cid in expected_fold0,
                "status": "PASS",
            }
        )
    return rows


def audit_summaries(splits: list[dict]) -> list[dict]:
    expected = {fold: set(split["val"]) for fold, split in enumerate(splits)}
    rows = []
    for root in SUMMARY_ROOTS:
        assert root.exists(), root
        for path in sorted(root.rglob("summary.json")):
            if path.parent.name != "validation":
                continue
            match = re.search(r"fold_(\d+)", str(path))
            assert match, path
            fold = int(match.group(1))
            payload = json.loads(path.read_text(encoding="utf-8"))
            ids = [case_id(row["prediction_file"]) for row in payload.get("metric_per_case", [])]
            class_means = [float(v["Dice"]) for v in payload.get("mean", {}).values()]
            calculated = float(np.nanmean(class_means))
            reported = float(payload["foreground_mean"]["Dice"])
            split_match = set(ids) == expected[fold] and len(ids) == len(set(ids))
            metric_match = math.isclose(calculated, reported, abs_tol=1e-12, rel_tol=0)
            fold_dir = path.parent.parent
            best_exists = (fold_dir / "checkpoint_best.pth").is_file()
            final_exists = (fold_dir / "checkpoint_final.pth").is_file()
            status = "PASS" if split_match and metric_match and best_exists and final_exists else "FAIL"
            assert status == "PASS", path
            rows.append(
                {
                    "result_root": root.name,
                    "relative_summary": str(path.relative_to(root)),
                    "fold": fold,
                    "n_cases": len(ids),
                    "case_ids_match_locked_validation_split": split_match,
                    "foreground_mean_recomputed_from_class_means": calculated,
                    "foreground_mean_reported": reported,
                    "metric_internal_consistency": metric_match,
                    "checkpoint_best_present": best_exists,
                    "checkpoint_final_present": final_exists,
                    "status": status,
                }
            )
    assert len(rows) == 52, f"expected 52 validation summaries, found {len(rows)}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_rows, split_summary, selected = audit_splits()
    splits = json.loads((PREPROCESSED / "splits_final.json").read_text(encoding="utf-8"))
    real_rows = audit_real_cases(selected, splits)
    prediction_rows = audit_prediction_geometry(set(splits[0]["val"]))
    summary_rows = audit_summaries(splits)

    write_csv(args.output_dir / "split_fold_checks.csv", split_rows)
    write_csv(args.output_dir / "real_case_sample_audit.csv", real_rows)
    write_csv(args.output_dir / "prediction_geometry_sample_audit.csv", prediction_rows)
    write_csv(args.output_dir / "historical_summary_inventory.csv", summary_rows)
    result = {
        **split_summary,
        "prediction_geometry_cases_checked": len(prediction_rows),
        "historical_validation_summaries_checked": len(summary_rows),
        "all_historical_summary_case_sets_match_locked_splits": all(
            row["case_ids_match_locked_validation_split"] for row in summary_rows
        ),
        "all_historical_summaries_have_best_and_final_checkpoints": all(
            row["checkpoint_best_present"] and row["checkpoint_final_present"] for row in summary_rows
        ),
        "overall_status": "PASS",
    }
    (args.output_dir / "real_data_and_summary_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log = [
        "PASS five patient-level folds cover all 192 cases exactly once as validation",
        "PASS raw image/label IDs match",
        "PASS 20 selected real cases: geometry and labels 0-5",
        "PASS 20 E0 predictions: case IDs, geometry, spacing, and labels",
        "PASS 52 historical validation summaries: locked case sets and internal Dice consistency",
        "PASS every audited result tree has checkpoint_best.pth and checkpoint_final.pth",
        "OVERALL PASS",
    ]
    (args.output_dir / "real_data_and_summary_audit.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
