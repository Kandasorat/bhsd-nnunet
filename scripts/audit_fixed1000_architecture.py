from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def hash_state(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def signature(network: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(tensor.shape)) for name, tensor in network.state_dict().items()]


def build(cls: type, plans: dict, dataset_json: dict, seed: int = 3407):
    os.environ.update(
        {
            "BHSD_SEED": str(seed),
            "BHSD_DATA_SEED": "1003410",
            "BHSD_DETERMINISTIC": "1",
            "nnUNet_n_proc_DA": "0",
        }
    )
    trainer = cls(plans, "2d", 0, dataset_json, torch.device("cpu"))
    trainer._apply_reproducibility_settings(seed)
    trainer.initialize()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = Path(args.preprocessed) / "Dataset001_BHSD"
    os.environ["nnUNet_preprocessed"] = str(Path(args.preprocessed))
    os.environ.setdefault("nnUNet_raw", str(dataset.parent / "unused_raw"))
    os.environ["nnUNet_results"] = tempfile.mkdtemp(prefix="fixed1000_architecture_")
    plans = json.loads((dataset / "nnUNetPlans.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))

    from nnunet25d.baseline import trainer_25d as historical
    from nnunet25d.fixed1000 import trainer as fixed

    pairs = {
        "A0": (
            fixed.nnUNetTrainer_25D_A0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407,
            historical.nnUNetTrainer_25D_HarmonizedMin300Patience100,
        ),
        "C1": (
            fixed.nnUNetTrainer_25D_C1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407,
            historical.nnUNetTrainer_25D_AdapterControlControlled,
        ),
        "C2": (
            fixed.nnUNetTrainer_25D_C2Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407,
            historical.nnUNetTrainer_25D_CSACenterNeighborControlled,
        ),
        "D0": (
            fixed.nnUNetTrainer_25D_D0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407,
            historical.nnUNetTrainer_25D_SpectralD0Control,
        ),
        "D1": (
            fixed.nnUNetTrainer_25D_D1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407,
            historical.nnUNetTrainer_25D_SpectralD1LowPass,
        ),
    }

    comparisons = {}
    c1_seed3407_hash = None
    for model, (new_cls, old_cls) in pairs.items():
        new = build(new_cls, plans, dataset_json)
        new_hash = hash_state(new.network)
        new_signature = signature(new.network)
        new_parameters = sum(p.numel() for p in new.network.parameters())
        del new
        gc.collect()
        old = build(old_cls, plans, dataset_json)
        old_hash = hash_state(old.network)
        old_signature = signature(old.network)
        old_parameters = sum(p.numel() for p in old.network.parameters())
        del old
        gc.collect()
        comparisons[model] = {
            "state_dict_signature_equal": new_signature == old_signature,
            "parameter_count_equal": new_parameters == old_parameters,
            "initialization_hash_equal_at_seed3407": new_hash == old_hash,
            "fixed_parameters": new_parameters,
            "historical_parameters": old_parameters,
            "fixed_initialization_sha256": new_hash,
            "historical_initialization_sha256": old_hash,
        }
        if model == "C1":
            c1_seed3407_hash = new_hash

    repeat = build(pairs["C1"][0], plans, dataset_json)
    repeat_hash = hash_state(repeat.network)
    del repeat
    different_cls = fixed.nnUNetTrainer_25D_C1Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed1234
    different = build(different_cls, plans, dataset_json, seed=1234)
    different_hash = hash_state(different.network)
    del different

    os.environ["BHSD_SEED"] = "3407"
    os.environ["BHSD_DATA_SEED"] = "1003410"
    core2d = fixed.nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407(
        plans, "2d", 0, dataset_json, torch.device("cpu")
    )
    core3d = fixed.nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407(
        plans, "3d_fullres", 0, dataset_json, torch.device("cpu")
    )
    checks = {
        "all_state_dict_signatures_equal": all(v["state_dict_signature_equal"] for v in comparisons.values()),
        "all_parameter_counts_equal": all(v["parameter_count_equal"] for v in comparisons.values()),
        "all_seed3407_initialization_hashes_equal": all(
            v["initialization_hash_equal_at_seed3407"] for v in comparisons.values()
        ),
        "same_seed_reproduces": repeat_hash == c1_seed3407_hash,
        "different_seed_differs": different_hash != c1_seed3407_hash,
        "2d_batch_size_12": core2d.configuration_manager.batch_size == 12,
        "3d_batch_size_2": core3d.configuration_manager.batch_size == 2,
        "2d_patch_256": list(core2d.configuration_manager.patch_size) == [256, 256],
        "3d_patch_unchanged": list(core3d.configuration_manager.patch_size) == [28, 256, 256],
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "comparisons": comparisons,
        "seed_hashes": {
            "c1_seed3407": c1_seed3407_hash,
            "c1_seed3407_repeat": repeat_hash,
            "c1_seed1234": different_hash,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
