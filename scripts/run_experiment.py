from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOCAL_NNUNET_DATA_ROOT = PROJECT_ROOT / "nnUNet_data"
LOCAL_NNUNET_PATHS = {
    "nnUNet_raw": LOCAL_NNUNET_DATA_ROOT / "nnUNet_raw",
    "nnUNet_preprocessed": LOCAL_NNUNET_DATA_ROOT / "nnUNet_preprocessed",
    "nnUNet_results": LOCAL_NNUNET_DATA_ROOT / "nnUNet_results",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(name_or_path: str) -> Dict[str, Any]:
    config_path = Path(name_or_path)
    if not config_path.exists():
        config_path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config: {name_or_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    return config


def ensure_required_env() -> None:
    resolved = {}
    for key, local_path in LOCAL_NNUNET_PATHS.items():
        configured = os.environ.get(key)
        configured_path = Path(configured) if configured else None
        if configured_path is not None and configured_path.exists():
            resolved[key] = configured_path
            continue
        if local_path.exists():
            resolved[key] = local_path
            os.environ[key] = str(local_path)
            continue
        if configured_path is not None:
            raise EnvironmentError(
                f"{key} points to a missing path: {configured_path}. "
                f"Also could not find the project-local fallback at {local_path}."
            )
        raise EnvironmentError(
            f"Missing {key}. Also could not find the project-local fallback at {local_path}."
        )
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def results_dir_for_config(config: Dict[str, Any]) -> Path:
    return RESULTS_DIR / config["experiment_name"]


def _dataset_root(config: Dict[str, Any]) -> Path:
    return ensure_required_env()["nnUNet_raw"] / config["dataset_name"]


def _split_json_path(config: Dict[str, Any]) -> Path:
    return ensure_required_env()["nnUNet_preprocessed"] / config["dataset_name"] / "splits_final.json"


def write_metadata(config: Dict[str, Any], stage: str) -> Path:
    exp_dir = results_dir_for_config(config)
    exp_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": stage,
        "config_path": config["_config_path"],
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "cwd": str(PROJECT_ROOT),
        "python": sys.executable,
    }
    metadata_path = exp_dir / f"{stage}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def append_row_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        frame.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(csv_path, index=False)


def gpu_index_for_config(config: Dict[str, Any]) -> int:
    device = str(config.get("device", "cuda"))
    if ":" in device:
        try:
            return int(device.split(":", maxsplit=1)[1])
        except ValueError:
            return 0
    return 0


class NvidiaSmiMonitor:
    def __init__(self, gpu_index: int, sample_interval_s: float = 30.0):
        self.gpu_index = int(gpu_index)
        self.sample_interval_s = float(sample_interval_s)
        self.samples: List[Dict[str, Any]] = []
        self.command = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _query_once(self) -> Dict[str, Any] | None:
        if self.command is None:
            return None
        result = subprocess.run(
            [
                self.command,
                f"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                gpu_index = int(parts[0])
            except ValueError:
                continue
            if gpu_index != self.gpu_index:
                continue
            return {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "gpu_index": gpu_index,
                "gpu_name": parts[1],
                "utilization_gpu_pct": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_total_mb": float(parts[4]),
                "temperature_gpu_c": float(parts[5]),
            }
        return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._query_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> None:
        if self.command is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s + 1.0))

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "gpu_monitor_available": self.command is not None,
                "gpu_sample_count": 0,
                "gpu_name": None,
                "mean_gpu_utilization_pct": None,
                "max_gpu_utilization_pct": None,
                "mean_gpu_memory_used_mb": None,
                "max_gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "max_gpu_temperature_c": None,
            }
        frame = pd.DataFrame(self.samples)
        return {
            "gpu_monitor_available": True,
            "gpu_sample_count": int(len(frame)),
            "gpu_name": frame["gpu_name"].iloc[-1],
            "mean_gpu_utilization_pct": float(frame["utilization_gpu_pct"].mean()),
            "max_gpu_utilization_pct": float(frame["utilization_gpu_pct"].max()),
            "mean_gpu_memory_used_mb": float(frame["memory_used_mb"].mean()),
            "max_gpu_memory_used_mb": float(frame["memory_used_mb"].max()),
            "gpu_memory_total_mb": float(frame["memory_total_mb"].max()),
            "max_gpu_temperature_c": float(frame["temperature_gpu_c"].max()),
        }


def resolve_cli_command(command: List[str]) -> List[str]:
    if not command:
        return command
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved is None and os.name == "nt":
        python_dir = Path(sys.executable).resolve().parent
        candidate_dirs = [python_dir / "Scripts", python_dir]
        for candidate_dir in candidate_dirs:
            candidate = candidate_dir / f"{executable}.exe"
            if candidate.exists():
                resolved = str(candidate)
                break
    if resolved is not None:
        return [resolved, *command[1:]]
    return command


def run_command(command: List[str], config: Dict[str, Any], stage: str) -> None:
    resolved_paths = ensure_required_env()
    exp_dir = results_dir_for_config(config)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(config, stage)
    env = os.environ.copy()
    env["nnUNet_n_proc_DA"] = str(config.get("nnunet_n_proc_da", 4))
    for key, path in resolved_paths.items():
        env[key] = str(path)

    resolved_command = resolve_cli_command(command)
    start_wall_time = pd.Timestamp.utcnow()
    start_perf = time.perf_counter()
    monitor = NvidiaSmiMonitor(
        gpu_index=gpu_index_for_config(config),
        sample_interval_s=float(config.get("resource_monitor_interval_s", 30)),
    )
    monitor.start()

    exit_code = None
    try:
        completed = subprocess.run(resolved_command, check=True, cwd=str(PROJECT_ROOT), env=env)
        exit_code = completed.returncode
    except subprocess.CalledProcessError as exc:
        exit_code = exc.returncode
        raise
    finally:
        monitor.stop()
        end_wall_time = pd.Timestamp.utcnow()
        duration_seconds = time.perf_counter() - start_perf

        resource_samples_csv = exp_dir / f"{stage}_resource_samples.csv"
        if monitor.samples:
            pd.DataFrame(monitor.samples).to_csv(resource_samples_csv, index=False)

        stage_metrics_row = {
            "experiment_name": config["experiment_name"],
            "stage": stage,
            "command": " ".join(resolved_command),
            "start_time_utc": start_wall_time.isoformat(),
            "end_time_utc": end_wall_time.isoformat(),
            "duration_seconds": round(duration_seconds, 3),
            "exit_code": exit_code,
            "device": config.get("device", "cuda"),
            "nnunet_n_proc_da": env["nnUNet_n_proc_DA"],
            "resume": bool(config.get("resume", False)),
        }
        stage_metrics_row.update(monitor.summary())
        append_row_to_csv(exp_dir / "stage_metrics.csv", stage_metrics_row)


def maybe_install_25d(config: Dict[str, Any]) -> None:
    trainer = config.get("trainer", "")
    if "25D" in trainer or "SpacingAware25D" in trainer:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "nnunet25d" / "install_extension.py")], check=True)


def is_custom_25d_config(config: Dict[str, Any]) -> bool:
    trainer = str(config.get("trainer", ""))
    return "25D" in trainer or "SpacingAware25D" in trainer


def preprocess(config: Dict[str, Any]) -> None:
    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(config["dataset_id"]),
        "--verify_dataset_integrity",
    ]
    run_command(command, config, "preprocess")


def _train_command(config: Dict[str, Any], fold: int) -> List[str]:
    command = [
        "nnUNetv2_train",
        str(config["dataset_name"]),
        str(config["configuration"]),
        str(fold),
    ]
    trainer = config.get("trainer", "nnUNetTrainer")
    if trainer != "nnUNetTrainer":
        command.extend(["-tr", trainer])
    plans = config.get("plans", "nnUNetPlans")
    if plans != "nnUNetPlans":
        command.extend(["-p", plans])
    if config.get("save_npz", False):
        command.append("--npz")
    if config.get("resume", False):
        command.append("--c")
    if config.get("disable_checkpointing", False):
        command.append("--disable_checkpointing")
    if config.get("device"):
        command.extend(["-device", str(config["device"])])
    return command


def train(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 3407)))
    maybe_install_25d(config)
    for fold in config.get("folds", [0]):
        run_command(_train_command(config, int(fold)), config, f"train_fold_{fold}")


def _infer_command(config: Dict[str, Any], fold: int) -> List[str]:
    resolved_paths = ensure_required_env()
    input_folder = config.get(f"_prepared_inference_input_fold_{fold}") or config.get("inference_input") or str(
        resolved_paths["nnUNet_raw"] / config["dataset_name"] / "imagesTs"
    )
    output_folder = str(results_dir_for_config(config) / f"inference_fold_{fold}")
    command = [
        "nnUNetv2_predict",
        "-i",
        input_folder,
        "-o",
        output_folder,
        "-d",
        str(config["dataset_name"]),
        "-c",
        str(config["configuration"]),
        "-f",
        str(fold),
        "-tr",
        str(config.get("trainer", "nnUNetTrainer")),
    ]
    command.extend(["-p", str(config.get("plans", "nnUNetPlans"))])
    return command


def infer(config: Dict[str, Any]) -> None:
    from scripts.prepare_inference_data import prepare_fold_validation_data

    if is_custom_25d_config(config):
        raise NotImplementedError(
            "The custom 2.5D trainers currently support training only. "
            "Their training-time dataloader stacks neighbouring slices into "
            "multi-channel inputs, but this repository does not yet provide a "
            "matching nnUNetPredictor/nnUNetv2_predict inference path. "
            "Use the saved checkpoints for training comparisons, or implement "
            "a dedicated 2.5D predictor before running infer/evaluate."
        )

    maybe_install_25d(config)
    for fold in config.get("folds", [0]):
        if not config.get("inference_input"):
            staging_root = results_dir_for_config(config) / "prepared_inference"
            images_dir, labels_dir = prepare_fold_validation_data(
                raw_dataset_dir=_dataset_root(config),
                split_json=_split_json_path(config),
                fold=int(fold),
                output_root=staging_root,
            )
            config[f"_prepared_inference_input_fold_{fold}"] = str(images_dir)
            config[f"_prepared_ground_truth_fold_{fold}"] = str(labels_dir)
        run_command(_infer_command(config, int(fold)), config, f"infer_fold_{fold}")


def evaluate(config: Dict[str, Any]) -> None:
    from evaluation.aggregate_results import aggregate_case_metrics
    from evaluation.run_evaluation import evaluate_folder

    if is_custom_25d_config(config):
        raise NotImplementedError(
            "The custom 2.5D pipeline currently has no dedicated inference "
            "implementation, so standalone evaluate is also unsupported. "
            "Training metrics and checkpoints are still produced correctly."
        )

    model_name = config["experiment_name"]
    all_case_csvs = []
    for fold in config.get("folds", [0]):
        prediction_dir = results_dir_for_config(config) / f"inference_fold_{fold}"
        ground_truth_dir = config.get(f"_prepared_ground_truth_fold_{fold}") or config.get("ground_truth_dir")
        if not ground_truth_dir:
            raise ValueError(
                "Evaluation requires 'ground_truth_dir' in the config or a prepared validation label folder. "
                "Point it to the folder containing reference label .nii.gz files."
            )
        case_csv = results_dir_for_config(config) / f"{model_name}_fold_{fold}_case_metrics.csv"
        evaluate_folder(
            prediction_dir=prediction_dir,
            ground_truth_dir=Path(ground_truth_dir),
            output_csv=case_csv,
            model_name=model_name,
        )
        all_case_csvs.append(case_csv)

    merged_csv = results_dir_for_config(config) / f"{model_name}_case_metrics.csv"
    frames = []
    for fold, csv_path in zip(config.get("folds", [0]), all_case_csvs):
        frame = pd.read_csv(csv_path)
        frame["fold"] = int(fold)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(merged_csv, index=False)

    summary_csv = results_dir_for_config(config) / f"{model_name}_summary.csv"
    aggregate_case_metrics(merged_csv, summary_csv)
    write_metadata(config, "evaluate")


def run_all(config: Dict[str, Any]) -> None:
    preprocess(config)
    train(config)
    infer(config)
    evaluate(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preprocess", "train", "infer", "evaluate", "run_all"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Override config and resume training from checkpoints.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config["resume"] = True

    if args.stage == "preprocess":
        preprocess(config)
    elif args.stage == "train":
        train(config)
    elif args.stage == "infer":
        infer(config)
    elif args.stage == "evaluate":
        evaluate(config)
    elif args.stage == "run_all":
        run_all(config)


if __name__ == "__main__":
    main()
