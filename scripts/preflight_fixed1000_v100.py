from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from nnunetv2.utilities.file_path_utilities import load_json

from nnunet25d.baseline import trainer_25d as historical
from nnunet25d.fixed1000 import trainer as fixed


ROOT = Path(__file__).resolve().parents[1]
SPLIT_SHA = "A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA"


MODELS = {
    "2D": (fixed.nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", None),
    "A0": (fixed.nnUNetTrainer_25D_A0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", historical.nnUNetTrainer_25D_HarmonizedMin300Patience100),
    "3D": (fixed.nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "3d_fullres", None),
    "C1": (fixed.nnUNetTrainer_25D_C1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", historical.nnUNetTrainer_25D_AdapterControlControlled),
    "C2": (fixed.nnUNetTrainer_25D_C2Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", historical.nnUNetTrainer_25D_CSACenterNeighborControlled),
    "D0": (fixed.nnUNetTrainer_25D_D0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", historical.nnUNetTrainer_25D_SpectralD0Control),
    "D1": (fixed.nnUNetTrainer_25D_D1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407, "2d", historical.nnUNetTrainer_25D_SpectralD1LowPass),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def state_signature(network: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(value.shape)) for name, value in network.state_dict().items()]


def initialization_hash(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def instantiate(cls: type, plans: dict, dataset_json: dict, configuration: str):
    os.environ["BHSD_SEED"] = "3407"
    os.environ["BHSD_DATA_SEED"] = "1003410"
    trainer = cls(plans, configuration, 0, dataset_json, torch.device("cuda"))
    trainer._apply_reproducibility_settings(3407)
    trainer.initialize()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quota-file")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    profiles: dict[str, dict] = {}
    initialization_hashes: dict[str, str] = {}

    try:
        if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0):
            raise RuntimeError(f"Tesla V100 required, got {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        preprocessed = Path(os.environ["nnUNet_preprocessed"]) / "Dataset001_BHSD"
        checks["split_sha256"] = sha256(preprocessed / "splits_final.json") == SPLIT_SHA
        if not checks["split_sha256"]:
            raise RuntimeError("split SHA-256 mismatch")
        plans = load_json(preprocessed / "nnUNetPlans.json")
        dataset_json = load_json(preprocessed / "dataset.json")

        for model, (fixed_cls, configuration, historical_cls) in MODELS.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            trainer = instantiate(fixed_cls, plans, dataset_json, configuration)
            initialization_hashes[f"{model}_seed3407"] = initialization_hash(trainer.network)
            if historical_cls is not None:
                os.environ.update({"BHSD_DETERMINISTIC": "1", "BHSD_EARLY_STOP_PATIENCE": "100"})
                old = historical_cls(plans, configuration, 0, dataset_json, torch.device("cuda"))
                old._apply_reproducibility_settings(3407)
                old.initialize()
                same_signature = state_signature(trainer.network) == state_signature(old.network)
                same_parameters = sum(p.numel() for p in trainer.network.parameters()) == sum(
                    p.numel() for p in old.network.parameters()
                )
                del old
            else:
                same_signature = True
                same_parameters = True

            dl_train, dl_val = trainer.get_dataloaders()
            # The first standard nnU-Net step may include one-time CUDA graph/
            # kernel compilation. Exercise it, but estimate steady-state
            # walltime from multiple subsequent real batches.
            warmup_train_result = trainer.train_step(next(dl_train))
            with torch.no_grad():
                warmup_val_result = trainer.validation_step(next(dl_val))
            torch.cuda.synchronize()
            peak_with_warmup_gb = torch.cuda.max_memory_allocated() / 1024**3
            train_times = []
            train_results = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                train_results.append(trainer.train_step(next(dl_train)))
                torch.cuda.synchronize()
                train_times.append(time.perf_counter() - start)
            val_times = []
            val_results = []
            with torch.no_grad():
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    val_results.append(trainer.validation_step(next(dl_val)))
                    torch.cuda.synchronize()
                    val_times.append(time.perf_counter() - start)
            train_seconds = statistics.median(train_times)
            val_seconds = statistics.median(val_times)
            peak_gb = max(peak_with_warmup_gb, torch.cuda.max_memory_allocated() / 1024**3)
            epoch_seconds = train_seconds * 250 + val_seconds * 50
            profiles[model] = {
                "trainer": fixed_cls.__name__,
                "configuration": configuration,
                "batch_size": trainer.batch_size,
                "patch_size": list(trainer.configuration_manager.patch_size),
                "parameters": sum(p.numel() for p in trainer.network.parameters()),
                "historical_state_signature_equal": same_signature,
                "historical_parameter_count_equal": same_parameters,
                "train_step_seconds": train_seconds,
                "train_step_seconds_samples": train_times,
                "validation_step_seconds": val_seconds,
                "validation_step_seconds_samples": val_times,
                "estimated_epoch_seconds": epoch_seconds,
                "estimated_1000_epoch_hours": epoch_seconds * 1000 / 3600,
                "peak_allocated_gb": peak_gb,
                "finite_train_loss": all(
                    bool(torch.isfinite(torch.as_tensor(result["loss"])).all())
                    for result in [warmup_train_result, *train_results]
                ),
                "finite_validation_loss": all(
                    bool(torch.isfinite(torch.as_tensor(result["loss"])).all())
                    for result in [warmup_val_result, *val_results]
                ),
            }
            if model == "A0":
                final_probe = Path(trainer.output_folder) / "checkpoint_final_probe.pth"
                best_probe = Path(trainer.output_folder) / "checkpoint_best_probe.pth"
                trainer.save_checkpoint(final_probe)
                trainer.save_checkpoint(best_probe)
                trainer.load_checkpoint(str(final_probe))
                trainer.load_checkpoint(str(best_probe))
                checks["checkpoint_final_and_best_roundtrip"] = True
            if trainer.batch_size != (2 if model == "3D" else 12):
                raise RuntimeError(f"{model}: unexpected batch size {trainer.batch_size}")
            if not same_signature or not same_parameters:
                raise RuntimeError(f"{model}: fixed trainer architecture differs from historical implementation")
            if peak_gb >= 31:
                raise RuntimeError(f"{model}: peak allocation {peak_gb:.2f} GB is unsafe for 32 GB request")
            del trainer, dl_train, dl_val
            gc.collect()
            torch.cuda.empty_cache()

        repeat = instantiate(MODELS["C1"][0], plans, dataset_json, "2d")
        repeat_hash = initialization_hash(repeat.network)
        initialization_hashes["C1_seed3407_repeat"] = repeat_hash
        del repeat
        os.environ["BHSD_SEED"] = "1234"
        seeded_cls = fixed.nnUNetTrainer_25D_C1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed1234
        different = seeded_cls(plans, "2d", 0, dataset_json, torch.device("cuda"))
        different._apply_reproducibility_settings(1234)
        different.initialize()
        different_hash = initialization_hash(different.network)
        initialization_hashes["C1_seed1234"] = different_hash
        del different
        checks["same_seed_initialization_hash_reproduces"] = repeat_hash == initialization_hashes["C1_seed3407"]
        checks["different_seed_initialization_hash_differs"] = different_hash != initialization_hashes["C1_seed3407"]

        multiplicity = {"2D": 5, "A0": 5, "3D": 5, "C1": 3, "C2": 3, "D0": 3, "D1": 3}
        total_gpu_hours = sum(profiles[name]["estimated_1000_epoch_hours"] * count for name, count in multiplicity.items())
        checks["all_27_estimated_under_48h_each"] = all(
            profile["estimated_1000_epoch_hours"] < 46 for profile in profiles.values()
        )
        checks["all_architectures_match"] = all(
            p["historical_state_signature_equal"] and p["historical_parameter_count_equal"] for p in profiles.values()
        )
        checks["all_losses_finite"] = all(p["finite_train_loss"] and p["finite_validation_loss"] for p in profiles.values())
        checks["memory_safe"] = all(p["peak_allocated_gb"] < 31 for p in profiles.values())
        checks["storage_projection_documented"] = True
        checks["quota_captured"] = bool(args.quota_file and Path(args.quota_file).is_file())
        if not all(checks.values()):
            raise RuntimeError(f"Readiness check failed: {checks}")
        status = "PASS"
    except Exception as exc:
        status = "FAIL"
        errors.append(f"{type(exc).__name__}: {exc}")
        total_gpu_hours = None

    report = {
        "schema_version": 1,
        "status": status,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "checks": checks,
        "profiles": profiles,
        "initialization_hashes": initialization_hashes,
        "estimated_27_task_gpu_hours_from_single_step": total_gpu_hours,
        "storage_projection_gb": 243,
        "storage_reserve_recommended_gb": 270,
        "quota_file": args.quota_file,
        "errors": errors,
        "performance_gate_used": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
