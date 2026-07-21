from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

# When invoked as ``python scripts/check_gadi_ready.py``, Python otherwise puts
# only ``scripts/`` at the front of sys.path. Prefer this checkout over a stale
# installed copy while validating the server environment.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTIVE_CONFIGS = (
    "baseline_2d",
    "baseline_3d",
    "baseline_25d_3slide",
    "spacing_aware_25d",
    "baseline_2d_binary",
    "baseline_3d_binary",
    "baseline_25d_3slide_binary",
    "baseline_25d_5slide_binary",
    "csam_official_3slice",
    "csam_official_3slice_binary",
    "csam_official_volume32_fold0",
    "csa_net_official_3slice_fold0",
)

REQUIRED_FILES = (
    "nnunet25d/install_extension.py",
    "nnunet25d/common/early_stopping.py",
    "nnunet25d/trainer_bhsd.py",
    "nnunet25d/trainer_25d.py",
    "nnunet25d/trainer_csam_volume_official.py",
    "nnunet25d/trainer_csa_net_official.py",
    "scripts/run_experiment.py",
    "hpc/gadi/train_2d_folds.pbs",
    "hpc/gadi/train_3d_folds.pbs",
    "hpc/gadi/train_25d_3slice_fold0.pbs",
    "hpc/gadi/train_binary_25d_3slice_fold0.pbs",
    "hpc/gadi/train_csam_volume_fold0.pbs",
    "hpc/gadi/train_csa_net_fold0.pbs",
    "source_faithful/bhsd_data.py",
    "source_faithful/train_attention.py",
    "hpc/gadi/smoke_source_faithful_attention.pbs",
    "hpc/gadi/train_csam_source_faithful_fold0.pbs",
    "hpc/gadi/train_csa_net_source_faithful_fold0.pbs",
    "hpc/gadi/train_csam_source_faithful_binary_fold0.pbs",
    "hpc/gadi/train_csa_net_source_faithful_binary_fold0.pbs",
)

EXPECTED_EARLY_STOP = {
    "max_num_epochs": 1000,
    "early_stop_min_epochs": 300,
    "early_stop_patience": 100,
    "early_stop_min_delta": 0.0001,
    "early_stop_metric": "ema_fg_dice",
}

ATTENTION_ADAPTATION_CONFIGS = {
    "csam_official_3slice",
    "csam_official_3slice_binary",
    "csam_official_volume32_fold0",
    "csa_net_official_3slice_fold0",
}

SOURCE_FAITHFUL_CONFIGS = {
    "csam_source_faithful_bhsd_fold0": {
        "method": "csam",
        "dataset_name": "Dataset001_BHSD",
        "num_classes": 6,
        "epochs": 150,
        "batch_size": 2,
        "sequence_length": 20,
        "input_size": 128,
        "learning_rate": 0.0001,
    },
    "csa_net_source_faithful_bhsd_fold0": {
        "method": "csa_net",
        "dataset_name": "Dataset001_BHSD",
        "num_classes": 6,
        "epochs": 40,
        "batch_size": 16,
        "input_size": 224,
        "learning_rate": 0.001,
        "seed": 1234,
        "deterministic": True,
    },
    "csam_source_faithful_bhsd_binary_fold0": {
        "method": "csam",
        "epochs": 150,
        "batch_size": 2,
        "sequence_length": 20,
        "input_size": 128,
        "learning_rate": 0.0001,
        "num_classes": 2,
        "dataset_name": "Dataset002_BHSD_Binary",
    },
    "csa_net_source_faithful_bhsd_binary_fold0": {
        "method": "csa_net",
        "epochs": 40,
        "batch_size": 16,
        "input_size": 224,
        "learning_rate": 0.001,
        "seed": 1234,
        "deterministic": True,
        "num_classes": 2,
        "dataset_name": "Dataset002_BHSD_Binary",
    },
}


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return config


def check_repository() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    conflict_markers = ("<<<<<<<", "=======", ">>>>>>>")
    conflict_scan = (
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "index.html",
        PROJECT_ROOT / "hpc" / "gadi" / "README.md",
    )
    for path in conflict_scan:
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in conflict_markers):
            errors.append(f"{path.relative_to(PROJECT_ROOT)} contains unresolved merge-conflict markers")

    for relative in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    for name in ACTIVE_CONFIGS:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            errors.append(f"Missing active config: {path.relative_to(PROJECT_ROOT)}")
            continue
        try:
            config = load_config(name)
        except Exception as exc:
            errors.append(f"Cannot parse {path.name}: {exc}")
            continue

        for key, expected in EXPECTED_EARLY_STOP.items():
            if config.get(key) != expected:
                errors.append(f"{path.name}: {key}={config.get(key)!r}, expected {expected!r}")
        if config.get("validation_checkpoint") != "best":
            errors.append(f"{path.name}: validation_checkpoint must be 'best'")
        if config.get("inference_checkpoint") != "checkpoint_best.pth":
            errors.append(f"{path.name}: inference_checkpoint must be checkpoint_best.pth")
        if config.get("save_npz") is not True:
            errors.append(f"{path.name}: save_npz must be true for reproducible validation")
        if name in ATTENTION_ADAPTATION_CONFIGS:
            if config.get("protocol_tier") != "harmonized_nnunet_adaptation":
                errors.append(f"{path.name}: attention protocol tier is not declared accurately")
            if config.get("source_faithful") is not False:
                errors.append(f"{path.name}: must declare source_faithful: false")
            if not config.get("upstream_repository") or not config.get("upstream_commit"):
                errors.append(f"{path.name}: missing pinned upstream provenance")

    for name, expected_fields in SOURCE_FAITHFUL_CONFIGS.items():
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            errors.append(f"Missing source-faithful config: {path.relative_to(PROJECT_ROOT)}")
            continue
        try:
            config = load_config(name)
        except Exception as exc:
            errors.append(f"Cannot parse {path.name}: {exc}")
            continue
        if config.get("protocol_tier") != "source_faithful_bhsd_port" or config.get("source_faithful") is not True:
            errors.append(f"{path.name}: source-faithful protocol tier is not declared accurately")
        if not config.get("upstream_repository") or not config.get("upstream_commit"):
            errors.append(f"{path.name}: missing pinned upstream provenance")
        if not config.get("unavoidable_deviations"):
            errors.append(f"{path.name}: unavoidable BHSD deviations are not documented")
        for key, expected in expected_fields.items():
            if config.get(key) != expected:
                errors.append(f"{path.name}: {key}={config.get(key)!r}, expected {expected!r}")

    trainer_source = (PROJECT_ROOT / "nnunet25d" / "trainer_bhsd.py").read_text(encoding="utf-8")
    baseline_25d_source = (PROJECT_ROOT / "nnunet25d" / "baseline" / "trainer_25d.py").read_text(
        encoding="utf-8"
    )
    if '[256, 256]' not in trainer_source:
        errors.append("2D BHSD trainer no longer records the [256, 256] patch override")
    if '[256, 256]' not in baseline_25d_source:
        errors.append("2.5D trainer no longer records the [256, 256] patch override")

    referenced_configs: set[str] = set()
    for pbs_path in (PROJECT_ROOT / "hpc" / "gadi").glob("*.pbs"):
        text = pbs_path.read_text(encoding="utf-8")
        referenced_configs.update(re.findall(r"--config\s+([A-Za-z0-9_]+)", text))
    for name in sorted(referenced_configs):
        if not (CONFIG_DIR / f"{name}.yaml").is_file():
            errors.append(f"PBS script references missing config: {name}")

    notes.append(f"Validated {len(ACTIVE_CONFIGS)} active experiment configs")
    notes.append(f"Validated {len(SOURCE_FAITHFUL_CONFIGS)} source-faithful protocol configs")
    notes.append(f"Validated {len(referenced_configs)} config references from Gadi PBS scripts")
    return errors, notes


def check_server(require_binary: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    resolved: dict[str, Path] = {}
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        value = os.environ.get(key)
        if not value:
            errors.append(f"Server environment variable is not set: {key}")
            continue
        path = Path(value)
        resolved[key] = path
        if not path.is_dir():
            errors.append(f"{key} does not exist: {path}")

    preprocessed = resolved.get("nnUNet_preprocessed")
    if preprocessed is not None and preprocessed.is_dir():
        dataset1 = preprocessed / "Dataset001_BHSD"
        for filename in ("nnUNetPlans.json", "splits_final.json"):
            if not (dataset1 / filename).is_file():
                errors.append(f"Missing Dataset001 preprocessed file: {dataset1 / filename}")

        plans_path = dataset1 / "nnUNetPlans.json"
        if plans_path.is_file():
            plans = json.loads(plans_path.read_text(encoding="utf-8"))
            patch_3d = plans["configurations"]["3d_fullres"]["patch_size"]
            notes.append(f"Dataset001 planned 3D patch: {patch_3d}")
            if list(patch_3d[-2:]) != [256, 256]:
                errors.append(f"Dataset001 3D in-plane patch changed unexpectedly: {patch_3d}")

        if require_binary:
            dataset2 = preprocessed / "Dataset002_BHSD_Binary"
            for filename in ("nnUNetPlans.json", "splits_final.json"):
                if not (dataset2 / filename).is_file():
                    errors.append(f"Missing Dataset002 preprocessed file: {dataset2 / filename}")

    try:
        import nnunetv2  # noqa: F401
        import torch  # noqa: F401

        from nnunet25d.trainer_25d import (  # noqa: F401
            nnUNetTrainer_25D_HarmonizedMin300Patience100,
        )
        from nnunet25d.trainer_bhsd import nnUNetTrainer_BHSDEarlyStop  # noqa: F401
        from nnunet25d.trainer_csa_net_official import nnUNetTrainer25DCSANetOfficial  # noqa: F401
        from nnunet25d.trainer_csam_volume_official import nnUNetTrainerCSAMVolumeOfficial  # noqa: F401

        notes.append("Python and active custom trainer imports succeeded")
    except Exception as exc:
        errors.append(f"Python environment or trainer import failed: {exc}")

    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the BHSD repository is ready for Gadi execution")
    parser.add_argument("--server", action="store_true", help="also check Gadi paths, plans, and trainer imports")
    parser.add_argument(
        "--require-binary",
        action="store_true",
        help="with --server, also require prepared Dataset002_BHSD_Binary",
    )
    args = parser.parse_args()

    errors, notes = check_repository()
    if args.server:
        server_errors, server_notes = check_server(args.require_binary)
        errors.extend(server_errors)
        notes.extend(server_notes)

    for note in notes:
        print(f"OK: {note}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Gadi readiness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
