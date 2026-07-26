from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCKED_SPLIT_SHA256 = "A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def validate_config(config: dict) -> None:
    expected = {
        "dataset_name": "Dataset001_BHSD",
        "num_epochs": 1000,
        "num_iterations_per_epoch": 250,
        "num_val_iterations_per_epoch": 50,
        "initial_lr": 0.01,
        "optimizer": "SGD",
        "momentum": 0.99,
        "nesterov": True,
        "weight_decay": 3e-5,
        "scheduler": "PolyLR",
        "scheduler_horizon": 1000,
        "poly_exponent": 0.9,
        "performance_early_stopping": False,
        "primary_checkpoint": "checkpoint_final.pth",
        "sensitivity_checkpoint": "checkpoint_best.pth",
        "save_npz": True,
        "validation_final_dir": "validation_final",
        "validation_best_sensitivity_dir": "validation_best_sensitivity",
        "allow_resume": False,
        "overwrite_existing": False,
        "nnunet_n_proc_da": 0,
        "data_seed": 1_003_410,
        "split_sha256": LOCKED_SPLIT_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Locked config mismatch for {key}: {config.get(key)!r} != {value!r}")
    if config.get("array_name") not in {"fixed1000_core_multiclass_15", "fixed1000_fold0_diagnostic_12"}:
        raise RuntimeError("Unauthorized array name")
    if str(config.get("model")) not in {"2D", "A0", "3D", "C1", "C2", "D0", "D1"}:
        raise RuntimeError("Unauthorized model")
    if len(config.get("folds", [])) != 1:
        raise RuntimeError("Each fixed1000 config must contain exactly one fold")


def command(config: dict, *, validation_only: bool, best: bool) -> list[str]:
    cmd = [
        shutil.which("nnUNetv2_train") or "nnUNetv2_train",
        str(config["dataset_name"]),
        str(config["configuration"]),
        str(config["folds"][0]),
        "-tr",
        str(config["trainer"]),
        "-p",
        str(config["plans"]),
        "--npz",
        "-device",
        "cuda",
    ]
    if validation_only:
        cmd.append("--val")
    if best:
        cmd.append("--val_best")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / "configs" / "fixed1000" / config_path
        if config_path.suffix != ".yaml":
            config_path = config_path.with_suffix(".yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)

    expected_commit = os.environ.get("FIXED1000_EXPECTED_COMMIT", "")
    if not expected_commit or git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("Git HEAD does not match FIXED1000_EXPECTED_COMMIT")
    if git("status", "--porcelain"):
        raise RuntimeError("Training requires a clean Git working tree")

    preprocessed = Path(os.environ["nnUNet_preprocessed"])
    split_path = preprocessed / "Dataset001_BHSD" / "splits_final.json"
    if sha256(split_path) != LOCKED_SPLIT_SHA256:
        raise RuntimeError("splits_final.json SHA-256 mismatch")

    os.environ.update(
        {
            "BHSD_SEED": str(config["seed"]),
            "BHSD_DATA_SEED": str(config["data_seed"]),
            "PYTHONHASHSEED": str(config["seed"]),
            "BHSD_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "nnUNet_n_proc_DA": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    results = Path(os.environ["nnUNet_results"])
    namespace = str(config["result_namespace"])
    result_folder = results / str(config["dataset_name"]) / namespace / f"fold_{config['folds'][0]}"
    if result_folder.exists():
        raise RuntimeError(f"Fail-closed: result folder already exists: {result_folder}")

    started = time.time()
    final_cmd = command(config, validation_only=False, best=False)
    subprocess.run(final_cmd, cwd=ROOT, check=True)
    final_validation = result_folder / "validation"
    final_target = result_folder / "validation_final"
    if not (final_validation / "summary.json").is_file() or not list(final_validation.glob("*.npz")):
        raise RuntimeError("Final-checkpoint validation or NPZ export is incomplete")
    final_validation.rename(final_target)

    best_cmd = command(config, validation_only=True, best=True)
    subprocess.run(best_cmd, cwd=ROOT, check=True)
    best_validation = result_folder / "validation"
    best_target = result_folder / "validation_best_sensitivity"
    if not (best_validation / "summary.json").is_file() or not list(best_validation.glob("*.npz")):
        raise RuntimeError("Best-checkpoint sensitivity validation or NPZ export is incomplete")
    best_validation.rename(best_target)

    for checkpoint in ("checkpoint_final.pth", "checkpoint_best.pth"):
        if not (result_folder / checkpoint).is_file():
            raise RuntimeError(f"Missing required checkpoint: {checkpoint}")
    provenance = {
        "schema_version": 1,
        "status": "COMPLETE",
        "git_commit": expected_commit,
        "git_clean_at_start": True,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "split_sha256": LOCKED_SPLIT_SHA256,
        "array_name": config["array_name"],
        "array_index": config["array_index"],
        "model": config["model"],
        "fold": config["folds"][0],
        "model_seed": config["seed"],
        "data_seed": config["data_seed"],
        "epochs_required": 1000,
        "primary_checkpoint": "checkpoint_final.pth",
        "sensitivity_checkpoint": "checkpoint_best.pth",
        "commands": {"train_and_final_validation": final_cmd, "best_sensitivity_validation": best_cmd},
        "wall_seconds": time.time() - started,
    }
    (result_folder / "fixed1000_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FIXED1000_TASK_COMPLETE={result_folder}")


if __name__ == "__main__":
    main()

