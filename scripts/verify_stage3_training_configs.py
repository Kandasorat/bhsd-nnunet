from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d.stage3.provenance import validate_locked_checkpoint, validate_split_file
from scripts.run_experiment import _train_command


CONFIG_NAMES = (
    "stage3_r0_fold1_seed3407",
    "stage3_r1_fold1_seed3407",
    "stage3_r0_fold1_seed1234",
    "stage3_r1_fold1_seed1234",
    "stage3_r0_fold1_seed5678",
    "stage3_r1_fold1_seed5678",
)

EXPECTED_TRAINERS = {
    ("R0", 3407): "nnUNetTrainer_Stage3_R0",
    ("R1", 3407): "nnUNetTrainer_Stage3_R1",
    ("R0", 1234): "nnUNetTrainer_Stage3_R0Seed1234",
    ("R1", 1234): "nnUNetTrainer_Stage3_R1Seed1234",
    ("R0", 5678): "nnUNetTrainer_Stage3_R0Seed5678",
    ("R1", 5678): "nnUNetTrainer_Stage3_R1Seed5678",
}

LOCKED_COMMON = {
    "protocol_tier": "preregistered_stage3_fold1_gate",
    "dataset_name": "Dataset001_BHSD",
    "dataset_id": 1,
    "configuration": "2d",
    "folds": [1],
    "device": "cuda",
    "save_npz": True,
    "validation_checkpoint": "best",
    "inference_checkpoint": "checkpoint_best.pth",
    "resume": False,
    "disable_checkpointing": False,
    "data_seed": 1_003_410,
    "deterministic": True,
    "nnunet_n_proc_da": 0,
    "max_num_epochs": 1000,
    "early_stop_min_epochs": 300,
    "early_stop_patience": 100,
    "early_stop_min_delta": 0.0001,
    "early_stop_metric": "ema_fg_dice",
    "plans": "nnUNetPlans",
    "num_input_slices": 3,
    "residual_parameters": 18_342,
    "backbone_passes_per_prediction": 1,
    "primary_model_metric": "nnunet_foreground_mean_dice",
    "primary_mechanism_endpoint": "class_balanced_gt_present_macro_dice",
    "training_approved": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_and_validate_configs(config_dir: Path) -> list[dict]:
    rows = []
    seen = set()
    for name in CONFIG_NAMES:
        path = config_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, expected in LOCKED_COMMON.items():
            if config.get(key) != expected:
                raise RuntimeError(f"{path.name}: {key}={config.get(key)!r}, expected {expected!r}")
        arm = str(config["stage3_arm"])
        seed = int(config["seed"])
        if config["trainer"] != EXPECTED_TRAINERS[(arm, seed)]:
            raise RuntimeError(f"{path.name}: trainer does not match arm/seed")
        if (arm, seed) in seen:
            raise RuntimeError(f"Duplicate arm/seed: {(arm, seed)}")
        seen.add((arm, seed))
        command = _train_command(config, 1)
        if "--npz" not in command or "--val_best" not in command:
            raise RuntimeError(f"{path.name}: generated command lacks --npz/--val_best")
        if "--disable_checkpointing" in command or "--c" in command:
            raise RuntimeError(f"{path.name}: initial command unexpectedly disables checkpoints or resumes")
        rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
                "arm": arm,
                "seed": seed,
                "trainer": config["trainer"],
                "command": command,
            }
        )
    if seen != set(EXPECTED_TRAINERS):
        raise RuntimeError(f"Arm/seed grid mismatch: {sorted(seen)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the locked six-job Stage3 fold1 configuration set")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split_file = args.split_file
    if split_file is None:
        preprocessed = os.environ.get("nnUNet_preprocessed")
        if not preprocessed:
            raise EnvironmentError("nnUNet_preprocessed is required when --split-file is omitted")
        split_file = Path(preprocessed) / "Dataset001_BHSD" / "splits_final.json"

    configs = load_and_validate_configs(args.config_dir)
    split = validate_split_file(split_file)
    checkpoints = []
    if args.checkpoint_root is not None:
        for fold in (1, 2, 3, 4):
            path = args.checkpoint_root / f"fold_{fold}" / "checkpoint_best.pth"
            checkpoints.append(
                {"fold": fold, "path": str(path), "sha256": validate_locked_checkpoint(path, fold)}
            )
    payload = {
        "status": "PASS",
        "configs": configs,
        "split_sha256": split["sha256"],
        "split_cases": split["num_cases"],
        "checkpoints": checkpoints,
        "explicit_validation_flags": ["--val_best", "--npz"],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
