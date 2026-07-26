from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["nnUNet_preprocessed"] = str(REPO / "nnUNet_data" / "nnUNet_preprocessed")
os.environ["nnUNet_raw"] = str(REPO / "nnUNet_data" / "nnUNet_raw")
os.environ["nnUNet_results"] = str(Path(r"C:\Users\92127\OneDrive\文档\New project\stage3_init_results"))

from nnunet25d.stage3.model import count_parameters
from nnunet25d.stage3.provenance import deterministic_state_dict_sha256
from nnunet25d.stage3.trainer import nnUNetTrainer_Stage3_R0, nnUNetTrainer_Stage3_R1


def initialize(trainer_class):
    preprocessed = REPO / "nnUNet_data" / "nnUNet_preprocessed"
    raw = REPO / "nnUNet_data" / "nnUNet_raw"
    results = REPO / "nnUNet_data" / "nnUNet_results"
    plans = json.loads((preprocessed / "Dataset001_BHSD" / "nnUNetPlans.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((preprocessed / "Dataset001_BHSD" / "dataset.json").read_text(encoding="utf-8"))
    trainer = trainer_class(plans, "2d", 1, dataset_json, torch.device("cpu"))
    trainer._apply_reproducibility_settings(trainer.bhsd_seed)
    trainer.initialize()
    wrapper = trainer._stage3_wrapper()
    optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    residual_ids = {id(parameter) for parameter in wrapper.residual.parameters() if parameter.requires_grad}
    center_ids = {id(parameter) for parameter in wrapper.center.parameters()}
    if optimizer_ids != residual_ids or optimizer_ids & center_ids:
        raise RuntimeError("Optimizer membership is not residual-only")
    if any(parameter.requires_grad for parameter in wrapper.center.parameters()):
        raise RuntimeError("Center parameter was not frozen")
    return trainer, {
        "trainer": trainer_class.__name__,
        "arm": wrapper.arm,
        "checkpoint_path": str(trainer.b_checkpoint_path),
        "checkpoint_sha256": trainer.b_checkpoint_sha256,
        "center_parameters": count_parameters(wrapper.center),
        "residual_parameters": count_parameters(wrapper.residual),
        "trainable_parameters": count_parameters(wrapper, trainable_only=True),
        "optimizer_parameter_tensors": len(optimizer_ids),
        "residual_state_sha256": deterministic_state_dict_sha256(wrapper.residual),
        "center_training": wrapper.center.training,
        "forward_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trainer_r0, r0 = initialize(nnUNetTrainer_Stage3_R0)
    del trainer_r0
    gc.collect()
    trainer_r1, r1 = initialize(nnUNetTrainer_Stage3_R1)
    if r0["checkpoint_sha256"] != r1["checkpoint_sha256"]:
        raise RuntimeError("R0/R1 checkpoint hashes differ")
    if r0["residual_state_sha256"] != r1["residual_state_sha256"]:
        raise RuntimeError("R0/R1 residual initialization differs")
    payload = {"status": "PASS", "R0": r0, "R1": r1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
