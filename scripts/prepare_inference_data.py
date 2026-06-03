from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_fold_validation_data(
    raw_dataset_dir: Path,
    split_json: Path,
    fold: int,
    output_root: Path,
) -> tuple[Path, Path]:
    with split_json.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    val_cases = splits[fold]["val"]

    images_src = raw_dataset_dir / "imagesTr"
    labels_src = raw_dataset_dir / "labelsTr"
    images_out = output_root / f"fold_{fold}" / "images"
    labels_out = output_root / f"fold_{fold}" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for case_id in val_cases:
        for image_file in sorted(images_src.glob(f"{case_id}_*.nii.gz")):
            _safe_link_or_copy(image_file, images_out / image_file.name)
        label_file = labels_src / f"{case_id}.nii.gz"
        if label_file.exists():
            _safe_link_or_copy(label_file, labels_out / label_file.name)

    return images_out, labels_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    prepare_fold_validation_data(
        raw_dataset_dir=Path(args.raw_dataset_dir),
        split_json=Path(args.split_json),
        fold=args.fold,
        output_root=Path(args.output_root),
    )


if __name__ == "__main__":
    main()
