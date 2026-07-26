from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunet25d.stage3.compute import count_complete_backbone_passes, measure_model
from nnunet25d.stage3.model import FrozenCenterResidualWrapper
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def build_random_center() -> torch.nn.Module:
    plans_path = REPO / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset001_BHSD" / "nnUNetPlans.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.warmup != 20 or args.repetitions != 100:
        raise ValueError("Stage3 preflight compute procedure is locked to warmup=20 and repetitions=100")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the locked memory/latency preflight")

    torch.manual_seed(3407)
    device = torch.device("cuda:0")
    center = build_random_center().to(device).eval()
    center.decoder.deep_supervision = False
    center_input = torch.randn(1, 1, 256, 256, device=device)
    triplet_input = torch.cat((center_input, center_input, center_input), dim=1)
    baseline = measure_model(center, center_input, warmup=args.warmup, repetitions=args.repetitions)

    wrapper = FrozenCenterResidualWrapper(center, "R0").to(device).eval()
    r0 = measure_model(wrapper, triplet_input, warmup=args.warmup, repetitions=args.repetitions)
    r0["complete_backbone_passes"] = count_complete_backbone_passes(
        wrapper, wrapper.center, triplet_input
    )
    wrapper.arm = "R1"
    r1_input = torch.randn_like(triplet_input)
    r1 = measure_model(wrapper, r1_input, warmup=args.warmup, repetitions=args.repetitions)
    r1["complete_backbone_passes"] = count_complete_backbone_passes(
        wrapper, wrapper.center, r1_input
    )

    if r0["parameters"] != r1["parameters"] or r0["conv2d_flops_batch"] != r1["conv2d_flops_batch"]:
        raise RuntimeError("R0/R1 compute or capacity differs")
    if r0["complete_backbone_passes"] != 1 or r1["complete_backbone_passes"] != 1:
        raise RuntimeError("Stage3 wrapper did not use exactly one center backbone pass")
    payload = {
        "scope": "randomly initialized locked architecture; no historical checkpoint loaded and no dataset inference",
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "B": baseline,
        "R0": r0,
        "R1": r1,
        "ratios": {
            "R1_over_B_median_latency": r1["median_latency_ms"] / baseline["median_latency_ms"],
            "R1_over_B_peak_allocated": r1["peak_allocated_bytes"] / baseline["peak_allocated_bytes"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
