from __future__ import annotations

import csv
import hashlib
import inspect
from pathlib import Path
import unittest

import yaml

from nnunet25d.common.early_stopping import BHSDEarlyStoppingMixin
from nnunet25d.fixed1000 import trainer as fixed


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "fixed1000"
MATRIX = ROOT / "preregistered_fixed1000_run_matrix.csv"
LOCKED_SPLIT = "A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA"


def matrix_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configs() -> list[dict]:
    return [yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(CONFIG_DIR.glob("*.yaml"))]


def trainer_classes() -> list[type]:
    names = sorted({row["trainer"] for row in matrix_rows()})
    return [getattr(fixed, name) for name in names]


class Fixed1000ProtocolTests(unittest.TestCase):
    def test_01_all_27_configs_parse(self) -> None:
        self.assertEqual(len(configs()), 27)


    def test_02_array_mapping_is_unique_and_complete(self) -> None:
        rows = matrix_rows()
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({(r["array_name"], r["array_index"]) for r in rows}), 27)
        core = sorted(int(r["array_index"]) for r in rows if r["array_name"].endswith("_15"))
        diagnostic = sorted(int(r["array_index"]) for r in rows if r["array_name"].endswith("_12"))
        self.assertEqual(core, list(range(15)))
        self.assertEqual(diagnostic, list(range(12)))


    def test_03_split_sha_is_locked(self) -> None:
        self.assertEqual({c["split_sha256"] for c in configs()}, {LOCKED_SPLIT})


    def test_04_no_trainer_mro_reaches_early_stopping(self) -> None:
        for cls in trainer_classes():
            self.assertNotIn(BHSDEarlyStoppingMixin, cls.mro())
            self.assertNotIn("early_stopping", " ".join(base.__module__ for base in cls.mro()))


    def test_05_polylr_values_match_formula(self) -> None:
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        import torch

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        scheduler = PolyLRScheduler(optimizer, 0.01, 1000, exponent=0.9)
        for epoch in (0, 500, 900, 999):
            scheduler.step(epoch)
            expected = 0.01 * (1 - epoch / 1000) ** 0.9
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected, places=15)


    def test_06_locked_schedule_and_optimizer_fields(self) -> None:
        for c in configs():
            self.assertEqual((c["num_epochs"], c["num_iterations_per_epoch"], c["num_val_iterations_per_epoch"]), (
            1000,
            250,
            50,
            ))
            self.assertEqual((c["optimizer"], c["initial_lr"], c["momentum"], c["nesterov"], c["weight_decay"]), (
            "SGD",
            0.01,
            0.99,
            True,
            3e-5,
            ))


    def test_07_c1_c2_protocol_only_differs_by_mechanism(self) -> None:
        by_key = {(c["model"], c["seed"]): c for c in configs()}
        ignored = {"experiment_name", "array_index", "model", "trainer", "result_namespace", "slice_order"}
        for seed in (3407, 1234, 5678):
            c1 = {k: v for k, v in by_key[("C1", seed)].items() if k not in ignored}
            c2 = {k: v for k, v in by_key[("C2", seed)].items() if k not in ignored}
            self.assertEqual(c1, c2)


    def test_08_d0_d1_protocol_only_differs_by_mechanism(self) -> None:
        by_key = {(c["model"], c["seed"]): c for c in configs()}
        ignored = {"experiment_name", "array_index", "model", "trainer", "result_namespace", "slice_order"}
        for seed in (3407, 1234, 5678):
            d0 = {k: v for k, v in by_key[("D0", seed)].items() if k not in ignored}
            d1 = {k: v for k, v in by_key[("D1", seed)].items() if k not in ignored}
            self.assertEqual(d0, d1)


    def test_09_seed_matrix_is_locked(self) -> None:
        rows = matrix_rows()
        self.assertEqual({int(r["model_seed"]) for r in rows if r["array_name"].endswith("_15")}, {3407})
        for model in ("C1", "C2", "D0", "D1"):
            self.assertEqual({int(r["model_seed"]) for r in rows if r["model"] == model}, {3407, 1234, 5678})
        self.assertEqual({int(r["data_seed"]) for r in rows}, {1_003_410})


    def test_10_a0_slice_order_and_boundary_replication(self) -> None:
        cls = fixed.nnUNetTrainer_25D_A0Fixed1000NoEarlyStoppingFinalCheckpointPrimarySeed3407
        obj = object.__new__(cls)
        obj.num_input_slices = 3
        self.assertEqual(obj._get_slice_indices(0, 4), [0, 0, 1])
        self.assertEqual(obj._get_slice_indices(2, 4), [1, 2, 3])
        self.assertEqual(obj._get_slice_indices(3, 4), [2, 3, 3])


    def test_11_final_checkpoint_is_primary_and_npz_is_required(self) -> None:
        for c in configs():
            self.assertEqual(c["primary_checkpoint"], "checkpoint_final.pth")
            self.assertEqual(c["sensitivity_checkpoint"], "checkpoint_best.pth")
            self.assertIs(c["save_npz"], True)


    def test_12_validation_namespaces_do_not_collide(self) -> None:
        for c in configs():
            self.assertEqual(c["validation_final_dir"], "validation_final")
            self.assertEqual(c["validation_best_sensitivity_dir"], "validation_best_sensitivity")
            self.assertNotEqual(c["validation_final_dir"], c["validation_best_sensitivity_dir"])


    def test_13_result_namespaces_are_unique_for_model_configuration_seed(self) -> None:
        rows = matrix_rows()
        keys = {(r["model"], r["configuration"], r["model_seed"], r["result_namespace"]) for r in rows}
        self.assertEqual(len(keys), 15)
        self.assertTrue(all("Fixed1000" in r["result_namespace"] for r in rows))
        self.assertTrue(all("NoEarlyStopping" in r["result_namespace"] for r in rows))
        self.assertTrue(all("FinalCheckpointPrimary" in r["result_namespace"] for r in rows))


    def test_14_runner_has_fail_closed_existing_result_guard(self) -> None:
        source = inspect.getsource(__import__("scripts.run_fixed1000_task", fromlist=["main"]))
        self.assertIn("if result_folder.exists()", source)
        self.assertIn("Fail-closed", source)
        self.assertNotIn('cmd.append("--c")', source)


    def test_15_final_validation_command_does_not_select_best(self) -> None:
        from scripts.run_fixed1000_task import command

        c = configs()[0]
        self.assertNotIn("--val_best", command(c, validation_only=False, best=False))
        self.assertIn("--val_best", command(c, validation_only=True, best=True))
        self.assertIn("--val", command(c, validation_only=True, best=True))


    def test_16_no_unauthorized_models_or_arrays(self) -> None:
        self.assertEqual({c["model"] for c in configs()}, {"2D", "A0", "3D", "C1", "C2", "D0", "D1"})
        self.assertEqual({c["array_name"] for c in configs()}, {
        "fixed1000_core_multiclass_15",
        "fixed1000_fold0_diagnostic_12",
        })


    def test_17_no_active_csa_csam_namespace_is_referenced(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in CONFIG_DIR.glob("*.yaml")).lower()
        self.assertNotIn("csa_net_official", text)
        self.assertNotIn("csam_official", text)
        self.assertNotIn("publication_v3", text)


    def test_18_config_hashes_are_stable_and_nonempty(self) -> None:
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in CONFIG_DIR.glob("*.yaml")}
        self.assertEqual(len(hashes), 27)


if __name__ == "__main__":
    unittest.main()
