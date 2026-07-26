from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d.stage3.compute import count_complete_backbone_passes, measure_model
from nnunet25d.stage3.model import FrozenCenterResidualWrapper
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from scripts.verify_stage3_training_configs import load_and_validate_configs, sha256


WARMUP = 20
REPETITIONS = 100
LATENCY_LIMIT = 1.25
MEMORY_LIMIT = 1.30


def build_random_center(plans_path: Path) -> torch.nn.Module:
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    architecture = plans["configurations"]["2d"]["architecture"]
    return get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture.get("_kw_requires_import", []),
        1,
        6,
        allow_init=True,
        deep_supervision=False,
    )


def git_state() -> tuple[str, list[str]]:
    head = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
    ).splitlines()
    return head, status


def code_hashes() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "nnunet25d" / "stage3").glob("*.py"))
    paths += [
        PROJECT_ROOT / "nnunet25d" / "trainer_stage3.py",
        PROJECT_ROOT / "scripts" / "verify_stage3_training_configs.py",
        Path(__file__),
        PROJECT_ROOT / "hpc" / "gadi" / "train_stage3_fold1_r0_r1.pbs",
    ]
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked Stage3 V100 pre-training compute gate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--plans", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage3 V100 gate requires CUDA")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    plans_path = args.plans
    if plans_path is None:
        preprocessed = os.environ.get("nnUNet_preprocessed")
        if not preprocessed:
            raise EnvironmentError("nnUNet_preprocessed is required when --plans is omitted")
        plans_path = Path(preprocessed) / "Dataset001_BHSD" / "nnUNetPlans.json"
    configs = load_and_validate_configs(args.config_dir)
    head, dirty = git_state()

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    center = build_random_center(plans_path).to(device).eval()
    center.decoder.deep_supervision = False
    center_input = torch.randn(1, 1, 256, 256, device=device)
    previous = torch.randn_like(center_input)
    following = torch.randn_like(center_input)
    r1_input = torch.cat((previous, center_input, following), dim=1)
    r0_input = torch.cat((center_input, center_input, center_input), dim=1)

    baseline = measure_model(
        center, center_input, warmup=WARMUP, repetitions=REPETITIONS, use_autocast=True
    )
    wrapper = FrozenCenterResidualWrapper(center, "R0").to(device).eval()
    r0 = measure_model(wrapper, r0_input, warmup=WARMUP, repetitions=REPETITIONS, use_autocast=True)
    r0["complete_backbone_passes"] = count_complete_backbone_passes(wrapper, wrapper.center, r0_input)
    wrapper.arm = "R1"
    r1 = measure_model(wrapper, r1_input, warmup=WARMUP, repetitions=REPETITIONS, use_autocast=True)
    r1["complete_backbone_passes"] = count_complete_backbone_passes(wrapper, wrapper.center, r1_input)

    latency_ratio = r1["median_latency_ms"] / baseline["median_latency_ms"]
    memory_ratio = r1["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
    v100 = "V100" in device_name.upper()
    capacity_match = (
        r0["parameters"] == r1["parameters"]
        and r0["trainable_parameters"] == r1["trainable_parameters"] == 18_342
        and r0["conv2d_flops_batch"] == r1["conv2d_flops_batch"]
        and r0["complete_backbone_passes"] == r1["complete_backbone_passes"] == 1
    )
    gate_pass = (
        v100
        and not dirty
        and capacity_match
        and latency_ratio <= LATENCY_LIMIT
        and memory_ratio <= MEMORY_LIMIT
    )
    payload = {
        "status": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "scope": "randomly initialized locked architecture; no historical checkpoint loaded and no patient inference",
        "git_head": head,
        "git_dirty": dirty,
        "device_name": device_name,
        "requires_v100": True,
        "plans_path": str(plans_path),
        "plans_sha256": sha256(plans_path),
        "config_hashes": {row["name"]: row["sha256"] for row in configs},
        "code_hashes": code_hashes(),
        "procedure": {
            "warmup": WARMUP,
            "repetitions": REPETITIONS,
            "batch": 1,
            "patch": [256, 256],
            "autocast": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "B": baseline,
        "R0": r0,
        "R1": r1,
        "checks": {
            "v100": v100,
            "git_clean": not dirty,
            "capacity_match": capacity_match,
            "latency_ratio": latency_ratio,
            "latency_limit": LATENCY_LIMIT,
            "latency_pass": latency_ratio <= LATENCY_LIMIT,
            "memory_ratio": memory_ratio,
            "memory_limit": MEMORY_LIMIT,
            "memory_pass": memory_ratio <= MEMORY_LIMIT,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if gate_pass else 3)


if __name__ == "__main__":
    main()
