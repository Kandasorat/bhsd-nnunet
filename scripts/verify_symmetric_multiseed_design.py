from __future__ import annotations

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d import trainer_25d as trainer_module  # noqa: E402


SPECS = {
    ("E0", 1234): (
        "symmetric_25d_e0_control_fold0_seed1234",
        "nnUNetTrainer_25D_SymmetricE0ControlSeed1234",
        "paired_e0_control",
    ),
    ("E2", 1234): (
        "symmetric_25d_e2_reliability_gate_fold0_seed1234",
        "nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed1234",
        "paired_e2_candidate",
    ),
    ("E0", 5678): (
        "symmetric_25d_e0_control_fold0_seed5678",
        "nnUNetTrainer_25D_SymmetricE0ControlSeed5678",
        "paired_e0_control",
    ),
    ("E2", 5678): (
        "symmetric_25d_e2_reliability_gate_fold0_seed5678",
        "nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed5678",
        "paired_e2_candidate",
    ),
}

BASE_CONFIGS = {
    "E0": "symmetric_25d_e0_control_fold0",
    "E2": "symmetric_25d_e2_reliability_gate_fold0",
}

BASE_TRAINERS = {
    "E0": "nnUNetTrainer_25D_SymmetricE0Control",
    "E2": "nnUNetTrainer_25D_SymmetricE2ReliabilityGate",
}

MATCHED_FIELDS = (
    "dataset_name",
    "dataset_id",
    "configuration",
    "folds",
    "device",
    "save_npz",
    "validation_checkpoint",
    "inference_checkpoint",
    "resume",
    "disable_checkpointing",
    "data_seed",
    "deterministic",
    "nnunet_n_proc_da",
    "max_num_epochs",
    "early_stop_min_epochs",
    "early_stop_patience",
    "early_stop_min_delta",
    "early_stop_metric",
    "plans",
    "slice_mode",
    "num_input_slices",
    "descriptor_channels",
    "backbone_passes_per_prediction",
)


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"Config is not a mapping: {path}")
    return payload


def main() -> None:
    errors: list[str] = []
    trainers: set[str] = set()
    configs: set[str] = set()

    for (arm, seed), (config_name, trainer_name, role) in SPECS.items():
        config = load_config(config_name)
        base = load_config(BASE_CONFIGS[arm])
        configs.add(config_name)
        trainers.add(trainer_name)

        if config.get("seed") != seed:
            errors.append(f"{config_name}: seed={config.get('seed')!r}, expected {seed}")
        if config.get("data_seed") != 1003410:
            errors.append(f"{config_name}: data_seed must remain 1003410")
        if config.get("trainer") != trainer_name:
            errors.append(f"{config_name}: trainer namespace is not the pre-specified isolated namespace")
        if config.get("confirmation_role") != role:
            errors.append(f"{config_name}: confirmation_role={config.get('confirmation_role')!r}")
        if config.get("protocol_tier") != "controlled_nnunet_symmetric_reliability_multiseed_confirmation":
            errors.append(f"{config_name}: incorrect protocol tier")
        if config.get("source_faithful") is not False:
            errors.append(f"{config_name}: must declare source_faithful: false")

        for field in MATCHED_FIELDS:
            if config.get(field) != base.get(field):
                errors.append(
                    f"{config_name}: {field}={config.get(field)!r} differs from {BASE_CONFIGS[arm]} "
                    f"({base.get(field)!r})"
                )

        trainer_class = getattr(trainer_module, trainer_name, None)
        base_class = getattr(trainer_module, BASE_TRAINERS[arm])
        if trainer_class is None or not issubclass(trainer_class, base_class):
            errors.append(f"{trainer_name}: is not an isolated subclass of {BASE_TRAINERS[arm]}")

    if len(configs) != 4 or len(trainers) != 4:
        errors.append("The four confirmation jobs do not have unique config and trainer namespaces")

    pbs_path = PROJECT_ROOT / "hpc" / "gadi" / "train_25d_symmetric_multiseed_fold0.pbs"
    pbs_text = pbs_path.read_text(encoding="utf-8")
    if "#PBS -J 0-3" not in pbs_text:
        errors.append("Confirmation PBS array must contain exactly four indices (0-3)")
    if "symmetric_25d_e1" in pbs_text.lower():
        errors.append("E1 must not appear in the E0/E2 multi-seed confirmation array")
    for config_name in configs:
        if config_name not in pbs_text:
            errors.append(f"Confirmation PBS does not reference {config_name}")

    if errors:
        raise AssertionError("Symmetric multi-seed design verification failed:\n- " + "\n- ".join(errors))

    print("Symmetric E0/E2 multi-seed design verification passed.")
    print("Additional model seeds: 1234, 5678; fixed data seed: 1003410; fold: 0")
    print("Validated: E0/E2 only, paired policy fields, isolated output namespaces, four-job PBS array")


if __name__ == "__main__":
    main()
