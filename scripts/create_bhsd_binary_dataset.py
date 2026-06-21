from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_SOURCE_LABELS = {0, 1, 2, 3, 4, 5}


def _copy_tree_files(src_dir: Path, dst_dir: Path, pattern: str = "*.nii.gz") -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_file in sorted(src_dir.glob(pattern)):
        shutil.copy2(src_file, dst_dir / src_file.name)
        count += 1
    return count


def _case_id_from_image(image_path: Path) -> str:
    stem = image_path.stem.replace(".nii", "")
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def _read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def _image_shape_zyx(image: sitk.Image) -> tuple[int, ...]:
    return tuple(int(i) for i in sitk.GetArrayFromImage(image).shape)


def _unique_int_values(array: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(array))


def _assert_subset(values: Iterable[int], allowed: set[int], path: Path, label_type: str) -> None:
    unexpected = sorted(set(int(v) for v in values) - allowed)
    if unexpected:
        raise ValueError(f"{label_type} at {path} contains unexpected labels: {unexpected}")


def _build_binary_dataset_json(src_dataset_json: dict, num_training: int) -> dict:
    return {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "hemorrhage": 1},
        "numTraining": int(num_training),
        "file_ending": src_dataset_json.get("file_ending", ".nii.gz"),
        "name": "BHSD_Binary",
        "dataset_name": "BHSD_Binary",
    }


def create_binary_dataset(src_dataset: Path, dst_dataset: Path, output_csv: Path) -> pd.DataFrame:
    if not src_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dataset}")
    if dst_dataset.exists():
        raise FileExistsError(
            f"Destination dataset already exists: {dst_dataset}. Refusing to overwrite the binary dataset."
        )

    src_images_tr = src_dataset / "imagesTr"
    src_labels_tr = src_dataset / "labelsTr"
    src_images_ts = src_dataset / "imagesTs"
    src_dataset_json_path = src_dataset / "dataset.json"
    if not src_dataset_json_path.exists():
        raise FileNotFoundError(f"Missing source dataset.json: {src_dataset_json_path}")

    with src_dataset_json_path.open("r", encoding="utf-8") as f:
        src_dataset_json = json.load(f)

    dst_images_tr = dst_dataset / "imagesTr"
    dst_labels_tr = dst_dataset / "labelsTr"
    dst_images_ts = dst_dataset / "imagesTs"
    dst_dataset.mkdir(parents=True, exist_ok=False)
    dst_images_tr.mkdir(parents=True, exist_ok=True)
    dst_labels_tr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_images_tr.glob("*.nii.gz"))
    label_files = sorted(src_labels_tr.glob("*.nii.gz"))
    if not image_files or not label_files:
        raise RuntimeError("Source dataset must contain imagesTr and labelsTr .nii.gz files")

    image_case_ids = {}
    for image_file in image_files:
        case_id = _case_id_from_image(image_file)
        image_case_ids.setdefault(case_id, []).append(image_file)
    label_case_ids = {label_file.stem.replace(".nii", ""): label_file for label_file in label_files}

    missing_labels = sorted(set(image_case_ids) - set(label_case_ids))
    if missing_labels:
        raise RuntimeError(f"Missing labelsTr files for case IDs: {missing_labels[:10]}")

    _copy_tree_files(src_images_tr, dst_images_tr)
    if src_images_ts.exists():
        _copy_tree_files(src_images_ts, dst_images_ts)

    rows = []
    foreground_voxels_per_positive_case: list[int] = []
    total_foreground_voxels = 0
    binary_positive_cases = 0

    for case_id, image_group in sorted(image_case_ids.items()):
        label_path = label_case_ids[case_id]
        label_image = _read_image(label_path)
        label_array = sitk.GetArrayFromImage(label_image)
        original_unique = _unique_int_values(label_array)
        _assert_subset(original_unique, VALID_SOURCE_LABELS, label_path, "Source label")

        reference_image = _read_image(image_group[0])
        image_shape = _image_shape_zyx(reference_image)
        label_shape = tuple(int(i) for i in label_array.shape)
        if image_shape != label_shape:
            raise ValueError(
                f"Shape mismatch for case {case_id}: image shape {image_shape} vs label shape {label_shape}"
            )

        binary_array = (label_array > 0).astype(np.uint8)
        binary_unique = _unique_int_values(binary_array)
        _assert_subset(binary_unique, {0, 1}, label_path, "Binary label")

        binary_foreground_voxels = int(binary_array.sum())
        is_positive_case = binary_foreground_voxels > 0
        if is_positive_case:
            binary_positive_cases += 1
            foreground_voxels_per_positive_case.append(binary_foreground_voxels)
        total_foreground_voxels += binary_foreground_voxels

        binary_label_image = sitk.GetImageFromArray(binary_array)
        binary_label_image.CopyInformation(label_image)
        sitk.WriteImage(binary_label_image, str(dst_labels_tr / label_path.name))

        rows.append(
            {
                "case_id": case_id,
                "original_unique_labels": "|".join(str(v) for v in original_unique),
                "binary_unique_labels": "|".join(str(v) for v in binary_unique),
                "image_shape": "x".join(str(v) for v in image_shape),
                "label_shape": "x".join(str(v) for v in label_shape),
                "binary_foreground_voxels": binary_foreground_voxels,
                "is_positive_case": bool(is_positive_case),
            }
        )

    binary_dataset_json = _build_binary_dataset_json(src_dataset_json, num_training=len(label_case_ids))
    with (dst_dataset / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(binary_dataset_json, f, indent=2)
        f.write("\n")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    min_fg = min(foreground_voxels_per_positive_case) if foreground_voxels_per_positive_case else 0
    median_fg = int(np.median(foreground_voxels_per_positive_case)) if foreground_voxels_per_positive_case else 0
    max_fg = max(foreground_voxels_per_positive_case) if foreground_voxels_per_positive_case else 0

    print(f"training_cases: {len(label_case_ids)}")
    print(f"binary_positive_cases: {binary_positive_cases}")
    print(f"total_binary_foreground_voxels: {total_foreground_voxels}")
    print(f"binary_foreground_voxels_per_positive_case_min: {min_fg}")
    print(f"binary_foreground_voxels_per_positive_case_median: {median_fg}")
    print(f"binary_foreground_voxels_per_positive_case_max: {max_fg}")
    print(f"dataset_check_csv: {output_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dataset", required=True)
    parser.add_argument("--dst-dataset", required=True)
    args = parser.parse_args()

    create_binary_dataset(
        src_dataset=Path(args.src_dataset),
        dst_dataset=Path(args.dst_dataset),
        output_csv=PROJECT_ROOT / "outputs" / "bhsd_binary_dataset_check.csv",
    )


if __name__ == "__main__":
    main()
