from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import ensure_required_env, load_config  # noqa: E402


def parameter_counts(network: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    backbone = getattr(network, "backbone", None)
    adapter = total - sum(parameter.numel() for parameter in backbone.parameters()) if backbone is not None else 0
    return total, trainable, adapter


def timed_forward(network: torch.nn.Module, data: torch.Tensor, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            network(data)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))

    start_time = time.perf_counter()
    with torch.inference_mode():
        network(data)
    return (time.perf_counter() - start_time) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one configured 2.5D trainer before full training")
    parser.add_argument("--config", required=True, help="config name or YAML path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.set_num_threads(1)

    config = load_config(args.config)
    paths = ensure_required_env()
    os.environ["BHSD_SEED"] = str(int(config.get("seed", 3407)))
    os.environ["BHSD_DATA_SEED"] = str(int(config.get("data_seed", int(os.environ["BHSD_SEED"]) + 1_000_003)))
    os.environ["BHSD_DETERMINISTIC"] = "1" if bool(config.get("deterministic", False)) else "0"
    os.environ["nnUNet_n_proc_DA"] = str(config.get("nnunet_n_proc_da", 0))

    from nnunetv2.run.run_training import get_trainer_from_args

    trainer = get_trainer_from_args(
        str(config["dataset_name"]),
        str(config["configuration"]),
        int(config.get("folds", [0])[0]),
        str(config["trainer"]),
        str(config.get("plans", "nnUNetPlans")),
        device=device,
    )
    if hasattr(trainer, "_apply_reproducibility_settings"):
        trainer._apply_reproducibility_settings(int(config.get("seed", 3407)))
    trainer.initialize()
    trainer.set_deep_supervision_enabled(False)
    network = trainer.network.eval()

    patch_size = tuple(int(value) for value in trainer.configuration_manager.patch_size)
    profile_seed = 20260722
    torch.manual_seed(profile_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(profile_seed)
    data = torch.randn((1, int(trainer.num_input_channels), *patch_size), device=device)
    total_parameters, trainable_parameters, adapter_parameters = parameter_counts(network)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline_memory_mb = torch.cuda.memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory_mb = None

    for _ in range(args.warmup):
        timed_forward(network, data, device)
    latencies_ms = [timed_forward(network, data, device) for _ in range(args.iterations)]

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    method_name = getattr(network, "method_name", "plain_backbone")
    backbone_passes = 2 if method_name == "d6_adaptive_invariant" else 1
    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, max(0, int(0.95 * len(sorted_latencies)) - 1))
    profile = {
        "experiment_name": config["experiment_name"],
        "config": str(config["_config_path"]),
        "trainer": config["trainer"],
        "method_name": method_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": list(data.shape),
        "profile_batch_size": 1,
        "profile_seed": profile_seed,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "adapter_parameters": adapter_parameters,
        "backbone_passes_per_prediction": backbone_passes,
        "forward_latency_mean_ms": statistics.fmean(latencies_ms),
        "forward_latency_median_ms": statistics.median(latencies_ms),
        "forward_latency_p95_ms": sorted_latencies[p95_index],
        "baseline_allocated_memory_mb": baseline_memory_mb,
        "peak_allocated_memory_mb": peak_memory_mb,
        "incremental_peak_memory_mb": (
            peak_memory_mb - baseline_memory_mb
            if peak_memory_mb is not None and baseline_memory_mb is not None
            else None
        ),
        "profile_scope": "inference forward, batch 1, deep supervision disabled; training cost comes from stage_metrics.csv",
        "torch_version": torch.__version__,
    }

    output = args.output
    if output is None:
        metadata_root = Path(os.environ.get("BHSD_RESULTS_DIR", PROJECT_ROOT / "results"))
        output = metadata_root / str(config["experiment_name"]) / "compute_profile.json"
    fold_copy = Path(trainer.output_folder) / "compute_profile.json"
    profile["fold_result_copy"] = str(fold_copy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    fold_copy.parent.mkdir(parents=True, exist_ok=True)
    fold_copy.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(f"Compute profile written to {output} and {fold_copy}")


if __name__ == "__main__":
    main()
