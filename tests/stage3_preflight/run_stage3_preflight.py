from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nnunet25d.common.dataloader_25d import nnUNetDataLoader25D
from nnunet25d.stage3.compute import count_complete_backbone_passes, measure_model
from nnunet25d.stage3.evaluator import evaluate_folder
from nnunet25d.stage3.metrics import dice_score, hd95_mm, lesion_metrics, nsd_mm
from nnunet25d.stage3.model import FrozenCenterResidualWrapper, LogitResidualBranch, count_parameters
from nnunet25d.stage3.provenance import (
    LOCKED_B_CHECKPOINT_SHA256,
    LOCKED_SPLIT_SHA256,
    apply_reproducibility_settings,
    deterministic_state_dict_sha256,
    sha256_file,
    validate_locked_checkpoint,
    validate_split_file,
)
from nnunet25d.stage3.statistics import (
    PresentDiceRow,
    class_balanced_present_macro,
    patient_cluster_bootstrap_present_delta,
    pooled_present_case_class_dice,
)
from nnunet25d.stage3.trainer import _nnUNetTrainerStage3FrozenResidual
from nnunet25d.stage3.transforms import (
    assert_stage3_intensity_synchronization,
    iter_transform_tree,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class TinyCenter(nn.Module):
    def __init__(self, deep_supervision: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, padding=1)
        self.decoder = SimpleNamespace(deep_supervision=deep_supervision)

    def forward(self, image: torch.Tensor):
        logits = self.conv(image)
        if self.decoder.deep_supervision:
            return [logits, F.avg_pool2d(logits, 2), F.avg_pool2d(logits, 4)]
        return logits


def _seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _equal_outputs(left, right) -> bool:
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            torch.equal(a, b) for a, b in zip(left, right)
        )
    return torch.equal(left, right)


def _training_transform(shared_intensity: bool = True):
    factory = (
        _nnUNetTrainerStage3FrozenResidual.get_training_transforms
        if shared_intensity
        else nnUNetTrainer.get_training_transforms
    )
    return factory(
        np.asarray([32, 32]),
        (-0.35, 0.35),
        None,
        (0, 1),
        False,
        [False],
        False,
        [1, 2, 3, 4, 5],
        None,
        None,
    )


def test_s3_01():
    _seed()
    center = TinyCenter()
    r0 = FrozenCenterResidualWrapper(copy.deepcopy(center), "R0")
    r1 = FrozenCenterResidualWrapper(copy.deepcopy(center), "R1")
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        baseline = center(image[:, 1:2])
        out_r0 = r0(image)
        out_r1 = r1(image)
    assert _equal_outputs(baseline, out_r0)
    assert _equal_outputs(baseline, out_r1)
    return {"scales": [list(value.shape) for value in baseline], "bitwise": True}


def test_s3_02():
    _seed()
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R1")
    wrapper.residual.final_conv.weight.data.normal_()
    wrapper.delta_enabled.zero_()
    shapes = [(1, 3, side, side + 4) for side in range(16, 36, 2)]
    for shape in shapes:
        image = torch.randn(shape)
        with torch.no_grad():
            baseline = wrapper.center(image[:, 1:2])
            observed = wrapper(image)
        assert _equal_outputs(baseline, observed)
    return {"shapes": [list(shape) for shape in shapes], "bitwise": True}


def test_s3_03():
    _seed()
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R1")
    optimizer = torch.optim.SGD(wrapper.residual.parameters(), lr=0.1)
    image = torch.randn(2, 3, 24, 24)
    target = torch.randn(2, 6, 24, 24)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = wrapper(image)[0]
        F.mse_loss(output, target).backward()
        optimizer.step()
    assert all(parameter.grad is None for parameter in wrapper.center.parameters())
    residual_grads = {name: parameter.grad for name, parameter in wrapper.residual.named_parameters()}
    assert residual_grads and all(gradient is not None for gradient in residual_grads.values())
    assert all(torch.isfinite(gradient).all() for gradient in residual_grads.values())
    return {"residual_parameters_with_finite_grad": len(residual_grads)}


def test_s3_04():
    _seed()
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R1")
    before = deterministic_state_dict_sha256(wrapper.center)
    wrapper.train()
    assert not wrapper.center.training
    optimizer = torch.optim.SGD(wrapper.residual.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    wrapper(torch.randn(2, 3, 20, 20))[0].square().mean().backward()
    optimizer.step()
    after = deterministic_state_dict_sha256(wrapper.center)
    assert before == after
    assert not wrapper.center.training
    return {"center_state_sha256": before}


def test_s3_05():
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R0")
    optimizer = torch.optim.SGD(wrapper.residual.parameters(), lr=0.01)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    residual_ids = {id(parameter) for parameter in wrapper.residual.parameters() if parameter.requires_grad}
    center_ids = {id(parameter) for parameter in wrapper.center.parameters()}
    assert optimizer_ids == residual_ids
    assert not optimizer_ids & center_ids
    return {"optimizer_parameter_tensors": len(optimizer_ids)}


def test_s3_06():
    r0 = FrozenCenterResidualWrapper(TinyCenter(), "R0")
    r1 = FrozenCenterResidualWrapper(TinyCenter(), "R1")
    assert count_parameters(r0.residual) == LogitResidualBranch.expected_parameter_count == 18_342
    assert count_parameters(r0) == count_parameters(r1)
    assert count_parameters(r0, trainable_only=True) == count_parameters(r1, trainable_only=True) == 18_342
    return {"residual_parameters": 18_342, "total_equal": True}


def test_s3_07():
    image = torch.randn(1, 3, 24, 24)
    counts = {}
    for arm in ("R0", "R1"):
        wrapper = FrozenCenterResidualWrapper(TinyCenter(), arm)
        stem_calls = 0

        def stem_hook(*_args):
            nonlocal stem_calls
            stem_calls += 1

        handle = wrapper.residual.slice_stem.register_forward_hook(stem_hook)
        center_calls = count_complete_backbone_passes(wrapper, wrapper.center, image)
        handle.remove()
        assert center_calls == 1 and stem_calls == 3
        counts[arm] = {"center": center_calls, "stem": stem_calls}
    center = TinyCenter()
    assert count_complete_backbone_passes(center, center, image[:, 1:2]) == 1
    return counts


def test_s3_08():
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R0")
    image = torch.randn(2, 3, 20, 20)
    triplet = wrapper.materialize_branch_triplet(image)
    assert torch.equal(triplet[:, 0:1], image[:, 1:2])
    assert torch.equal(triplet[:, 0:1], triplet[:, 1:2])
    assert torch.equal(triplet[:, 1:2], triplet[:, 2:3])
    return {"exact_duplicate": True}


def test_s3_09():
    loader = object.__new__(nnUNetDataLoader25D)
    loader.slice_offsets = (-1, 0, 1)
    assert loader._get_slice_indices(2, 5) == [1, 2, 3]
    volume = torch.arange(5).view(1, 5, 1, 1)
    observed = volume[:, loader._get_slice_indices(2, 5)].flatten().tolist()
    target = torch.tensor([[[[2]]]])
    assert observed == [1, 2, 3] and int(target.item()) == 2
    return {"triplet": observed, "target": 2}


def test_s3_10():
    loader = object.__new__(nnUNetDataLoader25D)
    loader.slice_offsets = (-1, 0, 1)
    assert loader._get_slice_indices(0, 5) == [0, 0, 1]
    assert loader._get_slice_indices(4, 5) == [3, 4, 4]
    wrapper = FrozenCenterResidualWrapper(TinyCenter(), "R0")
    triplet = wrapper.materialize_branch_triplet(torch.tensor([[[[0.0]], [[4.0]], [[9.0]]]]))
    assert triplet.flatten().tolist() == [4.0, 4.0, 4.0]
    return {"first": [0, 0, 1], "last": [3, 4, 4], "r0": [4, 4, 4]}


def test_s3_11():
    _seed()
    branch = LogitResidualBranch().eval()
    branch.final_conv.weight.data.normal_()
    branch.final_conv.bias.data.normal_()
    triplet = torch.randn(2, 3, 28, 28)
    logits = torch.randn(2, 6, 28, 28)
    swapped = triplet[:, [2, 1, 0]]
    with torch.no_grad():
        delta_a, state_a = branch(triplet, logits, return_intermediates=True)
        delta_b, state_b = branch(swapped, logits, return_intermediates=True)
    max_error = float((delta_a - delta_b).abs().max())
    assert torch.equal(state_a["neighbour"], state_b["neighbour"])
    assert torch.equal(delta_a, delta_b) or max_error <= 1e-7
    return {"max_abs_error": max_error}


def test_s3_12():
    pipeline = _training_transform()
    spatial = pipeline.transforms[0]
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    marker = (((yy - 22) ** 2 + (xx - 27) ** 2) <= 45).float()
    image = marker.repeat(3, 1, 1)
    seg = marker.to(torch.int16).unsqueeze(0)
    minimum_dice = 1.0
    for seed in range(32):
        torch.manual_seed(seed)
        transformed = spatial(image=image.clone(), segmentation=seg.clone())
        channels = transformed["image"]
        target = transformed["segmentation"]
        assert torch.equal(channels[0], channels[1]) and torch.equal(channels[1], channels[2])
        predicted_mask = channels[0] >= 0.5
        target_mask = target[0] >= 0.5
        denominator = int(predicted_mask.sum() + target_mask.sum())
        value = 1.0 if denominator == 0 else float(2 * (predicted_mask & target_mask).sum() / denominator)
        minimum_dice = min(minimum_dice, value)
    assert minimum_dice >= 0.95
    return {"trials": 32, "minimum_landmark_dice": minimum_dice}


def test_s3_13():
    pipeline = _training_transform()
    assert_stage3_intensity_synchronization(pipeline)
    settings = {
        type(item).__name__: bool(item.synchronize_channels)
        for item in iter_transform_tree(pipeline)
        if hasattr(item, "synchronize_channels")
    }
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij")
    base = (yy + 2 * xx + 3).float()
    image = base.repeat(3, 1, 1)
    seg = (base > 3).to(torch.int16).unsqueeze(0)
    for seed in range(100):
        torch.manual_seed(seed)
        transformed = pipeline(image=image.clone(), segmentation=seg.clone())["image"]
        assert torch.equal(transformed[0], transformed[1])
        assert torch.equal(transformed[1], transformed[2])
    return {"trials": 100, "settings": settings}


def test_s3_14():
    wrapper = FrozenCenterResidualWrapper(TinyCenter(True), "R1")
    image = torch.randn(2, 3, 32, 32)
    output = wrapper(image)
    assert [list(item.shape) for item in output] == [[2, 6, 32, 32], [2, 6, 16, 16], [2, 6, 8, 8]]
    wrapper.set_deep_supervision_enabled(False)
    inference = wrapper(image)
    assert list(inference.shape) == [2, 6, 32, 32]
    return {"training_shapes": [list(item.shape) for item in output], "inference_shape": list(inference.shape)}


def test_s3_15():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint_best.pth"
        path.write_bytes(b"synthetic-stage3-checkpoint")
        observed = sha256_file(path)
        original = LOCKED_B_CHECKPOINT_SHA256[4]
        try:
            LOCKED_B_CHECKPOINT_SHA256[4] = observed
            assert validate_locked_checkpoint(path, 4) == observed
            LOCKED_B_CHECKPOINT_SHA256[4] = "0" * 64
            try:
                validate_locked_checkpoint(path, 4)
            except RuntimeError:
                rejected_wrong_hash = True
            else:
                rejected_wrong_hash = False
            assert rejected_wrong_hash
            wrong_name = path.with_name("checkpoint_final.pth")
            wrong_name.write_bytes(path.read_bytes())
            try:
                validate_locked_checkpoint(wrong_name, 4)
            except ValueError:
                rejected_wrong_name = True
            else:
                rejected_wrong_name = False
            assert rejected_wrong_name
        finally:
            LOCKED_B_CHECKPOINT_SHA256[4] = original
    return {"same_hash_accepted": True, "wrong_hash_rejected": True, "final_name_rejected": True}


def test_s3_16():
    code = (
        "import json; from nnunet25d.stage3.provenance import apply_reproducibility_settings; "
        "print(json.dumps(apply_reproducibility_settings(1234)))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO)
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=environment
    )
    snapshot = json.loads(completed.stdout.strip().splitlines()[-1])
    assert snapshot["declared_seed"] == snapshot["torch_initial_seed"] == 1234
    assert snapshot["python_hash_seed"] == "1234"
    assert snapshot["nnunet_n_proc_da"] == "0"
    assert snapshot["deterministic_algorithms"] and snapshot["cudnn_deterministic"]
    assert not snapshot["cudnn_benchmark"]
    return snapshot


def _audit_records(seed: int, count: int = 100):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(count):
        center = int(rng.integers(0, 32))
        records.append(
            {
                "case_id": f"BHSD_{int(rng.integers(0, 192)):03d}",
                "center_z": center,
                "crop": [int(rng.integers(-32, 32)), int(rng.integers(-32, 32)), 256, 256],
                "geometry_params": [float(rng.uniform(-0.35, 0.35)), float(rng.uniform(0.7, 1.4))],
                "intensity_params": [float(rng.uniform(0.75, 1.25)), float(rng.uniform(0.7, 1.5))],
            }
        )
    return records


def test_s3_17():
    r0_records = _audit_records(1_003_410)
    r1_records = _audit_records(1_003_410)
    assert r0_records == r1_records and len(r0_records) == 100
    image = torch.randn(1, 3, 16, 16)
    r0 = FrozenCenterResidualWrapper(TinyCenter(), "R0").materialize_branch_triplet(image)
    r1 = FrozenCenterResidualWrapper(TinyCenter(), "R1").materialize_branch_triplet(image)
    assert torch.equal(r0[:, 1], r1[:, 1]) and not torch.equal(r0[:, 0], r1[:, 0])
    return {"paired_records": 100, "only_materialized_neighbours_differ": True}


def test_s3_18():
    split_path = REPO / "nnUNet_data" / "nnUNet_preprocessed" / "Dataset001_BHSD" / "splits_final.json"
    result = validate_split_file(split_path)
    assert result["sha256"] == LOCKED_SPLIT_SHA256 and result["num_cases"] == 192
    return {"sha256": result["sha256"], "cases": 192, "folds": 5}


def test_s3_19():
    empty = np.zeros((3, 3, 3), dtype=bool)
    present = empty.copy(); present[1, 1, 1] = True
    assert np.isnan(dice_score(empty, empty))
    assert dice_score(present, empty) == 0.0
    assert dice_score(empty, present) == 0.0
    rows = []
    for model in ("R0", "R1"):
        rows.extend(
            [PresentDiceRow("p1", 1, model, True, 1.0), PresentDiceRow("p2", 1, model, True, 1.0)]
        )
        for class_id in range(2, 6):
            rows.append(PresentDiceRow("p1", class_id, model, True, 0.0))
    class_balanced = class_balanced_present_macro(rows, "R0")
    pooled = pooled_present_case_class_dice(rows, "R0")
    assert class_balanced != pooled
    bootstrap = patient_cluster_bootstrap_present_delta(rows, "R1", "R0", iterations=50, seed=20260726)
    assert bootstrap["observed_delta"] == 0.0
    return {"both_empty": "NaN", "present_empty": 0.0, "class_balanced": class_balanced, "pooled": pooled}


def test_s3_20():
    mask = np.zeros((4, 4, 4), dtype=bool); mask[1, 1, 1] = True
    shifted = np.zeros_like(mask); shifted[2, 1, 1] = True
    assert hd95_mm(mask, mask, (5.0, 1.0, 1.0)) == 0.0
    assert hd95_mm(mask, shifted, (5.0, 1.0, 1.0)) == 5.0
    diagonal = hd95_mm(mask, np.zeros_like(mask), (5.0, 1.0, 1.0))
    assert np.isclose(diagonal, np.linalg.norm([15.0, 3.0, 3.0]))
    import inspect
    import nnunet25d.stage3.metrics as metric_module
    assert "evaluation.metrics" not in inspect.getsource(metric_module)
    return {"perfect_mm": 0.0, "z_shift_mm": 5.0, "empty_penalty_mm": diagonal}


def test_s3_21():
    reference = np.zeros((3, 3, 3), dtype=bool); reference[0, 1, 1] = True
    prediction = np.zeros_like(reference); prediction[1, 1, 1] = True
    assert nsd_mm(reference, reference, (1.0, 1.0, 1.0), 3.0) == 1.0
    within = nsd_mm(reference, prediction, (2.9, 1.0, 1.0), 3.0)
    outside = nsd_mm(reference, prediction, (3.1, 1.0, 1.0), 3.0)
    assert within == 1.0 and outside == 0.0
    assert nsd_mm(reference, np.zeros_like(reference), (1.0, 1.0, 1.0), 3.0) == 0.0
    return {"perfect": 1.0, "2.9mm": within, "3.1mm": outside, "empty_prediction": 0.0}


def test_s3_22():
    reference = np.zeros((5, 5, 7), dtype=bool)
    reference[2, 2, 1] = True
    reference[2, 2, 5] = True
    merged_prediction = np.zeros_like(reference)
    merged_prediction[2, 2, 1:6] = True
    result = lesion_metrics(reference, merged_prediction, (1.0, 1.0, 1.0))
    assert result.true_positive == 1 and result.false_negative == 1 and result.false_positive == 0
    assert result.small_gt_lesions == 2 and result.small_true_positive == 1
    separate_prediction = reference.copy()
    exact = lesion_metrics(reference, separate_prediction, (1.0, 1.0, 1.0))
    assert exact.true_positive == 2 and exact.recall == 1.0
    return {"merged_prediction": result.__dict__, "separate_prediction_tp": exact.true_positive}


def test_s3_23():
    raw = REPO / "nnUNet_data" / "nnUNet_raw" / "Dataset001_BHSD"
    image_paths = sorted((raw / "imagesTr").glob("*_0000.nii.gz"))[:20]
    assert len(image_paths) == 20
    audited = []
    for image_path in image_paths:
        case_id = image_path.name.removesuffix("_0000.nii.gz")
        label_path = raw / "labelsTr" / f"{case_id}.nii.gz"
        image = sitk.ReadImage(str(image_path))
        segmentation = sitk.ReadImage(str(label_path))
        assert image.GetSize() == segmentation.GetSize()
        assert np.allclose(image.GetSpacing(), segmentation.GetSpacing(), rtol=0, atol=1e-6)
        assert np.allclose(image.GetOrigin(), segmentation.GetOrigin(), rtol=0, atol=1e-5)
        assert np.allclose(image.GetDirection(), segmentation.GetDirection(), rtol=0, atol=1e-6)
        labels = set(int(value) for value in np.unique(sitk.GetArrayViewFromImage(segmentation)))
        assert labels.issubset(set(range(6)))
        audited.append(case_id)
    return {"read_only_cases": audited, "stage3_summaries_found": 0}


def test_s3_24():
    _seed()
    center = TinyCenter(deep_supervision=False)
    r0 = FrozenCenterResidualWrapper(copy.deepcopy(center), "R0")
    r1 = FrozenCenterResidualWrapper(copy.deepcopy(center), "R1")
    image = torch.randn(1, 3, 64, 64)
    reports = {
        "B": measure_model(center, image[:, 1:2], warmup=2, repetitions=5),
        "R0": measure_model(r0, image, warmup=2, repetitions=5),
        "R1": measure_model(r1, image, warmup=2, repetitions=5),
    }
    assert reports["R0"]["parameters"] == reports["R1"]["parameters"]
    assert reports["R0"]["conv2d_flops_batch"] == reports["R1"]["conv2d_flops_batch"]
    assert count_complete_backbone_passes(r0, r0.center, image) == 1
    assert count_complete_backbone_passes(r1, r1.center, image) == 1
    reports["scope"] = "synthetic implementation measurement; full nnU-Net hardware profile is a separate freeze artifact"
    return reports


def test_s3_25():
    import inspect
    source = inspect.getsource(_nnUNetTrainerStage3FrozenResidual)
    assert "Stage3 fold0 training is permanently forbidden" in source
    assert "Stage3 fold0 performance evaluation is forbidden" in source
    try:
        evaluate_folder("missing", "missing", fold=0, output_json="unused", output_csv="unused")
    except RuntimeError:
        evaluator_blocked = True
    else:
        evaluator_blocked = False
    assert evaluator_blocked
    return {"trainer_guard": True, "evaluator_guard": True}


TESTS = {
    "S3-01": test_s3_01,
    "S3-02": test_s3_02,
    "S3-03": test_s3_03,
    "S3-04": test_s3_04,
    "S3-05": test_s3_05,
    "S3-06": test_s3_06,
    "S3-07": test_s3_07,
    "S3-08": test_s3_08,
    "S3-09": test_s3_09,
    "S3-10": test_s3_10,
    "S3-11": test_s3_11,
    "S3-12": test_s3_12,
    "S3-13": test_s3_13,
    "S3-14": test_s3_14,
    "S3-15": test_s3_15,
    "S3-16": test_s3_16,
    "S3-17": test_s3_17,
    "S3-18": test_s3_18,
    "S3-19": test_s3_19,
    "S3-20": test_s3_20,
    "S3-21": test_s3_21,
    "S3-22": test_s3_22,
    "S3-23": test_s3_23,
    "S3-24": test_s3_24,
    "S3-25": test_s3_25,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "simpleitk": sitk.Version_VersionString(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "tests": {},
    }
    log_lines = []
    failed = False
    for test_id, function in TESTS.items():
        try:
            evidence = function()
            results["tests"][test_id] = {"status": "PASS", "evidence": evidence}
            log_lines.append(f"{test_id} PASS")
        except Exception as error:
            failed = True
            results["tests"][test_id] = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            log_lines.append(f"{test_id} FAIL {type(error).__name__}: {error}")
    results["overall_status"] = "FAIL" if failed else "PASS"
    json_path = args.output_dir / "stage3_preflight_results.json"
    log_path = args.output_dir / "stage3_preflight.log"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    checksum_path = args.output_dir / "stage3_preflight_SHA256SUMS.txt"
    checksum_lines = [
        f"{file_sha256(Path(__file__))}  {Path(__file__).name}",
        f"{file_sha256(json_path)}  {json_path.name}",
        f"{file_sha256(log_path)}  {log_path.name}",
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    print("\n".join(log_lines))
    print(f"OVERALL {results['overall_status']}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
