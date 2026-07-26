from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOCKED_SPLIT_SHA256 = "A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA"
LOCKED_B_CHECKPOINT_SHA256 = {
    0: "64401DF82911F37814052DBF3529752F02FA645595998E832BADD77A1C5B9A93",
    1: "AA88CEF38BF82C2B9880C02696C83A89099CCF653E4E763284D90953BD787AD1",
    2: "2BB596645A3DCAD78AA4156E66F1E0AAC133E563B19A9D91718C5D6A65044458",
    3: "78B3775392B633F09E9E892C2658A42819E040285D1055034DC113C59B326CF8",
    4: "E59B3FD75BA8A1D397564A8CBE61988A36646E29D848635774A943B135620628",
}
LOCKED_B_ROOT = Path(r"D:\BHSD_server_backups\multiclass_2d_min300_patience100")


@dataclass(frozen=True)
class Stage3RunConfig:
    arm: str
    fold: int
    model_seed: int
    data_seed: int = 1_003_410
    patch_size: tuple[int, int] = (256, 256)
    batch_size: int = 12
    initial_lr: float = 0.01
    momentum: float = 0.99
    nesterov: bool = True
    weight_decay: float = 3e-5
    max_epochs: int = 1000
    minimum_epochs: int = 300
    patience: int = 100
    minimum_improvement: float = 1e-4
    checkpoint_rule: str = "online_ema_fg_dice_best"
    augmentation_workers: int = 0

    def __post_init__(self) -> None:
        if self.arm.upper() not in {"R0", "R1"}:
            raise ValueError("arm must be R0 or R1")
        if self.fold not in {0, 1, 2, 3, 4}:
            raise ValueError("fold must be in 0..4")
        if self.model_seed not in {3407, 1234, 5678}:
            raise ValueError("model_seed is not preregistered")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_path(fold: int, root: str | Path = LOCKED_B_ROOT) -> Path:
    if fold not in LOCKED_B_CHECKPOINT_SHA256:
        raise ValueError(f"No locked B checkpoint for fold {fold}")
    return Path(root) / f"fold_{fold}" / "checkpoint_best.pth"


def validate_locked_checkpoint(path: str | Path, fold: int) -> str:
    candidate = Path(path)
    if candidate.name != "checkpoint_best.pth":
        raise ValueError("Stage3 rejects final/latest checkpoints; filename must be checkpoint_best.pth")
    expected = LOCKED_B_CHECKPOINT_SHA256.get(fold)
    if expected is None:
        raise ValueError(f"No preregistered checkpoint hash for fold {fold}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    observed = sha256_file(candidate)
    if observed != expected:
        raise RuntimeError(f"B checkpoint SHA-256 mismatch for fold {fold}: {observed} != {expected}")
    return observed


def deterministic_state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def apply_reproducibility_settings(model_seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(int(model_seed))
    os.environ["nnUNet_n_proc_DA"] = "0"
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return reproducibility_snapshot(model_seed)


def reproducibility_snapshot(declared_seed: int) -> dict[str, Any]:
    return {
        "declared_seed": int(declared_seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "nnunet_n_proc_da": os.environ.get("nnUNet_n_proc_DA"),
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def validate_split_file(path: str | Path) -> dict[str, Any]:
    split_path = Path(path)
    observed_hash = sha256_file(split_path)
    if observed_hash != LOCKED_SPLIT_SHA256:
        raise RuntimeError(f"Split SHA-256 mismatch: {observed_hash} != {LOCKED_SPLIT_SHA256}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if len(splits) != 5:
        raise RuntimeError(f"Expected five folds, found {len(splits)}")
    validation_counts: dict[str, int] = {}
    fold_rows = []
    all_cases = set()
    for fold, item in enumerate(splits):
        train = set(item["train"])
        val = set(item["val"])
        if train & val:
            raise RuntimeError(f"Fold {fold} has train/validation overlap")
        all_cases.update(train | val)
        for case_id in val:
            validation_counts[case_id] = validation_counts.get(case_id, 0) + 1
        fold_rows.append({"fold": fold, "train": sorted(train), "val": sorted(val)})
    if len(all_cases) != 192:
        raise RuntimeError(f"Expected 192 cases, found {len(all_cases)}")
    if set(validation_counts.values()) != {1} or set(validation_counts) != all_cases:
        raise RuntimeError("Each case must appear in validation exactly once")
    return {
        "sha256": observed_hash,
        "num_cases": len(all_cases),
        "folds": fold_rows,
    }


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_as_dict(config: Stage3RunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["arm"] = config.arm.upper()
    payload["patch_size"] = list(config.patch_size)
    return payload
