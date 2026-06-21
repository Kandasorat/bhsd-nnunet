from __future__ import annotations

import torch

from nnunet25d.common.early_stopping import BHSDEarlyStoppingMixin
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_BHSDEarlyStop(BHSDEarlyStoppingMixin, nnUNetTrainer):
    """Standard nnU-Net trainer with validation-based early stopping for BHSD."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initialize_early_stopping()


__all__ = ["nnUNetTrainer_BHSDEarlyStop"]
