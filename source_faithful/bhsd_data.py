from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


def fold_keys(preprocessed_dataset: Path, fold: int) -> tuple[list[str], list[str]]:
    splits = json.loads((preprocessed_dataset / "splits_final.json").read_text(encoding="utf-8"))
    split = splits[int(fold)]
    return list(split["train"]), list(split["val"])


class BHSDCaseStore:
    """Read nnU-Net-preprocessed cases while preserving the frozen five-fold split."""

    def __init__(self, preprocessed_dataset: str | Path, keys: list[str], binary: bool = False):
        self.root = Path(preprocessed_dataset)
        self.binary = bool(binary)
        self.data_folder = self.root / "nnUNetPlans_2d"
        dataset_class = infer_dataset_class(str(self.data_folder))
        self.dataset = dataset_class(str(self.data_folder), keys)

    @lru_cache(maxsize=32)
    def load(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        data, seg, _, _ = self.dataset.load_case(key)
        target = np.asarray(seg[0], dtype=np.int64)
        # nnU-Net uses -1 outside the cropped valid image region. The released
        # standalone losses have no ignore-label handling, so this BHSD adapter
        # restores those voxels to semantic background.
        target = np.maximum(target, 0)
        if self.binary:
            target = (target > 0).astype(np.int64, copy=False)
        return np.asarray(data[0], dtype=np.float32), target


def resize_pair(image: np.ndarray, target: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if image.shape == (size, size):
        return image, target
    factors = (size / image.shape[0], size / image.shape[1])
    return zoom(image, factors, order=3), zoom(target, factors, order=0)


class CSAMSequenceDataset(Dataset):
    def __init__(self, store: BHSDCaseStore, keys: list[str], sequence_length: int = 20, size: int = 128):
        self.store = store
        self.keys = list(keys)
        self.sequence_length = int(sequence_length)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        key = self.keys[index]
        image, target = self.store.load(key)
        max_start = max(0, image.shape[0] - self.sequence_length)
        start = random.randint(0, max_start) if max_start else 0
        indices = np.clip(np.arange(start, start + self.sequence_length), 0, image.shape[0] - 1)
        image, target = image[indices], target[indices]
        y = random.randint(0, max(0, image.shape[1] - self.size))
        x = random.randint(0, max(0, image.shape[2] - self.size))
        image = image[:, y : y + self.size, x : x + self.size]
        target = target[:, y : y + self.size, x : x + self.size]
        if image.shape[-2:] != (self.size, self.size):
            resized = [resize_pair(i, t, self.size) for i, t in zip(image, target)]
            image = np.stack([item[0] for item in resized])
            target = np.stack([item[1] for item in resized])
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image[:, None])).float(),
            "label": torch.from_numpy(np.ascontiguousarray(target)).long(),
            "case": key,
        }


class CSANetSliceDataset(Dataset):
    def __init__(self, store: BHSDCaseStore, keys: list[str], size: int = 224, augment: bool = True):
        self.store = store
        self.size = int(size)
        self.augment = bool(augment)
        self.samples: list[tuple[str, int]] = []
        for key in keys:
            image, _ = store.load(key)
            self.samples.extend((key, z) for z in range(image.shape[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        key, z = self.samples[index]
        image, target = self.store.load(key)
        prev_z, next_z = max(0, z - 1), min(image.shape[0] - 1, z + 1)
        prev, center, following, label = image[prev_z], image[z], image[next_z], target[z]
        if self.augment and random.choice((False, True)):
            prev, center, following, label = [np.fliplr(a) for a in (prev, center, following, label)]
        prev, resized_label = resize_pair(prev, label, self.size)
        center, _ = resize_pair(center, label, self.size)
        following, _ = resize_pair(following, label, self.size)
        return {
            "prev": torch.from_numpy(np.ascontiguousarray(prev[None])).float(),
            "image": torch.from_numpy(np.ascontiguousarray(center[None])).float(),
            "next": torch.from_numpy(np.ascontiguousarray(following[None])).float(),
            "label": torch.from_numpy(np.ascontiguousarray(resized_label)).long(),
            "case": key,
            "slice": z,
        }
