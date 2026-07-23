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
    1234: (
        "axial_25d_a8_controlled_fold0_seed1234",
        "nnUNetTrainer_25D_AxialSliceConvControlledSeed1234",
    ),
    5678: (
        "axial_25d_a8_controlled_fold0_seed5678",
        "nnUNetTrainer_25D_AxialSliceConvControlledSeed5678",
    ),
}
BASE_CONFIG = "fusion_25d_c3_axial_fold0"
BASE_TRAINER = "nnUNetTrainer_25D_AxialSliceConvControlled"
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
    "axial_kernel",
)


def load_config(name: str) -> dict:
    payload = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {name}")
    return payload


def main() -> None:
    errors: list[str] = []
    base = load_config(BASE_CONFIG)
    base_class = getattr(trainer_module, BASE_TRAINER)
    old_a8_class = getattr(trainer_module, "nnUNetTrainer_25D_AxialSliceConv")

    if base.get("seed") != 3407 or base.get("data_seed") != 1003410:
        errors.append("Completed C3 reference is not the locked model/data seed 3407/1003410")
    if base.get("deterministic") is not True or base.get("nnunet_n_proc_da") != 0:
        errors.append("Completed C3 reference is not under the controlled data policy")
    if old_a8_class.adapter_method != base_class.adapter_method or base_class.adapter_method != "axial_slice_conv":
        errors.append("C3 no longer implements the same axial_slice_conv mechanism as A8")

    trainers: set[str] = set()
    configs: set[str] = set()
    for seed, (config_name, trainer_name) in SPECS.items():
        config = load_config(config_name)
        configs.add(config_name)
        trainers.add(trainer_name)

        if config.get("seed") != seed:
            errors.append(f"{config_name}: seed={config.get('seed')!r}, expected {seed}")
        if config.get("data_seed") != 1003410:
            errors.append(f"{config_name}: data_seed must remain 1003410")
        if config.get("trainer") != trainer_name:
            errors.append(f"{config_name}: trainer namespace is not pre-specified")
        if config.get("confirmation_role") != "a8_controlled_comparator":
            errors.append(f"{config_name}: incorrect confirmation role")
        if config.get("protocol_tier") != "controlled_nnunet_axial_multiseed_comparator":
            errors.append(f"{config_name}: incorrect protocol tier")
        if config.get("source_faithful") is not False:
            errors.append(f"{config_name}: must declare source_faithful: false")
        if config.get("backbone_passes_per_prediction") != 1:
            errors.append(f"{config_name}: must remain a one-backbone-pass comparator")

        for field in MATCHED_FIELDS:
            if config.get(field) != base.get(field):
                errors.append(
                    f"{config_name}: {field}={config.get(field)!r} differs from {BASE_CONFIG} "
                    f"({base.get(field)!r})"
                )

        trainer_class = getattr(trainer_module, trainer_name, None)
        if trainer_class is None or not issubclass(trainer_class, base_class):
            errors.append(f"{trainer_name}: is not an isolated subclass of {BASE_TRAINER}")

    if len(configs) != 2 or len(trainers) != 2:
        errors.append("The two additional A8 comparator jobs do not have unique namespaces")

    pbs_path = PROJECT_ROOT / "hpc" / "gadi" / "train_25d_axial_multiseed_fold0.pbs"
    pbs_text = pbs_path.read_text(encoding="utf-8")
    if "#PBS -J 0-1" not in pbs_text:
        errors.append("Controlled A8 PBS array must contain exactly two indices (0-1)")
    if "screen_25d_a8_axial_slice_conv_fold0" in pbs_text:
        errors.append("Controlled A8 PBS must not resubmit the completed A8 screen")
    if BASE_CONFIG in pbs_text:
        errors.append("Controlled A8 PBS must not rerun the completed C3 seed 3407 reference")
    for config_name in configs:
        if config_name not in pbs_text:
            errors.append(f"Controlled A8 PBS does not reference {config_name}")

    if errors:
        raise AssertionError("Controlled A8 multi-seed design verification failed:\n- " + "\n- ".join(errors))

    print("Controlled A8 multi-seed comparator verification passed.")
    print("Completed reference: C3 seed 3407; additional model seeds: 1234, 5678")
    print("Fixed data seed: 1003410; fold: 0; one backbone pass; two-job PBS array")
    print("Validated: same A8/C3 axial mechanism, matched policy, isolated output namespaces")


if __name__ == "__main__":
    main()
