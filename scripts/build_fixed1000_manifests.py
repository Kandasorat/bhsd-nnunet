from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "fixed1000"
MATRIX_PATH = ROOT / "preregistered_fixed1000_run_matrix.csv"
DATA_SEED = 1_003_410

CORE_TRAINERS = {
    "2D": "nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407",
    "A0": "nnUNetTrainer_25D_A0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407",
    "3D": "nnUNetTrainer_BHSDFixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407",
}


def diagnostic_trainer(model: str, seed: int) -> str:
    return f"nnUNetTrainer_25D_{model}Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed{seed}"


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    configurations = {"2D": "2d", "A0": "2d", "3D": "3d_fullres"}
    for model_index, model in enumerate(("2D", "A0", "3D")):
        for fold in range(5):
            trainer = CORE_TRAINERS[model]
            configuration = configurations[model]
            result.append(
                matrix_row(
                    array_name="fixed1000_core_multiclass_15",
                    array_index=model_index * 5 + fold,
                    model=model,
                    fold=fold,
                    model_seed=3407,
                    trainer=trainer,
                    configuration=configuration,
                    estimated_walltime="48:00:00",
                    estimated_storage="9GB",
                )
            )
    for model_index, model in enumerate(("C1", "C2", "D0", "D1")):
        for seed_index, seed in enumerate((3407, 1234, 5678)):
            result.append(
                matrix_row(
                    array_name="fixed1000_fold0_diagnostic_12",
                    array_index=model_index * 3 + seed_index,
                    model=model,
                    fold=0,
                    model_seed=seed,
                    trainer=diagnostic_trainer(model, seed),
                    configuration="2d",
                    estimated_walltime="48:00:00",
                    estimated_storage="9GB",
                )
            )
    return result


def matrix_row(**values: object) -> dict[str, object]:
    trainer = str(values["trainer"])
    configuration = str(values["configuration"])
    return {
        **values,
        "data_seed": DATA_SEED,
        "epochs": 1000,
        "iterations_per_epoch": 250,
        "val_iterations_per_epoch": 50,
        "optimizer": "SGD(lr=0.01,momentum=0.99,nesterov=true,weight_decay=3e-5)",
        "scheduler": "PolyLR(exponent=0.9)",
        "scheduler_horizon": 1000,
        "early_stopping": "disabled",
        "primary_checkpoint": "checkpoint_final.pth",
        "sensitivity_checkpoint": "checkpoint_best.pth",
        "result_namespace": f"{trainer}__nnUNetPlans__{configuration}",
    }


def config_name(row: dict[str, object]) -> str:
    model = str(row["model"]).lower()
    return f"{row['array_name']}_{int(row['array_index']):02d}_{model}_fold{row['fold']}_seed{row['model_seed']}.yaml"


def render_config(row: dict[str, object]) -> str:
    return "\n".join(
        [
            f"experiment_name: {row['array_name']}_{str(row['model']).lower()}_fold{row['fold']}_seed{row['model_seed']}",
            "protocol_tier: fixed1000_preregistered_confirmatory",
            f"array_name: {row['array_name']}",
            f"array_index: {row['array_index']}",
            f"model: {row['model']}",
            "dataset_name: Dataset001_BHSD",
            "dataset_id: 1",
            f"configuration: {row['configuration']}",
            f"trainer: {row['trainer']}",
            f"folds: [{row['fold']}]",
            "plans: nnUNetPlans",
            "device: cuda",
            f"seed: {row['model_seed']}",
            f"data_seed: {row['data_seed']}",
            "deterministic: true",
            "nnunet_n_proc_da: 0",
            "num_epochs: 1000",
            "num_iterations_per_epoch: 250",
            "num_val_iterations_per_epoch: 50",
            "initial_lr: 0.01",
            "optimizer: SGD",
            "momentum: 0.99",
            "nesterov: true",
            "weight_decay: 0.00003",
            "scheduler: PolyLR",
            "scheduler_horizon: 1000",
            "poly_exponent: 0.9",
            "performance_early_stopping: false",
            "primary_checkpoint: checkpoint_final.pth",
            "sensitivity_checkpoint: checkpoint_best.pth",
            "save_npz: true",
            "validation_final_dir: validation_final",
            "validation_best_sensitivity_dir: validation_best_sensitivity",
            f"result_namespace: {row['result_namespace']}",
            "split_sha256: A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA",
            "allow_resume: false",
            "overwrite_existing: false",
            "num_input_slices: 3" if row["model"] != "2D" and row["model"] != "3D" else "core_dimensionality_from_configuration: true",
            "slice_order: previous-centre-next" if row["model"] != "2D" and row["model"] != "3D" else "slice_order: not_applicable",
            "supervision: centre" if row["model"] != "2D" and row["model"] != "3D" else "supervision: standard_nnunet",
            "z_boundary: replicate" if row["model"] != "2D" and row["model"] != "3D" else "z_boundary: not_applicable",
            "",
        ]
    )


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = rows()
    expected_names = {config_name(row) for row in run_rows}
    for existing in CONFIG_DIR.glob("*.yaml"):
        if existing.name not in expected_names:
            raise RuntimeError(f"Unexpected fixed1000 config already exists: {existing}")
    for row in run_rows:
        (CONFIG_DIR / config_name(row)).write_text(render_config(row), encoding="utf-8", newline="\n")
    fieldnames = list(run_rows[0])
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)
    print(f"Wrote {len(run_rows)} configs and {MATRIX_PATH}")


if __name__ == "__main__":
    main()

