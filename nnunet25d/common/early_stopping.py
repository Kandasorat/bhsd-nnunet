from __future__ import annotations

import os
from typing import List

import numpy as np
import torch


class BHSDEarlyStoppingMixin:
    def initialize_early_stopping(self) -> None:
        self.early_stop_patience = int(os.environ.get("BHSD_EARLY_STOP_PATIENCE", "0"))
        self.early_stop_min_delta = float(os.environ.get("BHSD_EARLY_STOP_MIN_DELTA", "0.0"))
        self.early_stop_metric = os.environ.get("BHSD_EARLY_STOP_METRIC", "ema_fg_dice")
        self.num_epochs = int(os.environ.get("BHSD_MAX_EPOCHS", str(self.num_epochs)))

        self._early_stop_enabled = self.early_stop_patience > 0
        self._early_stop_best = None
        self._early_stop_bad_epochs = 0
        self._early_stop_triggered = False

    def _metric_history(self) -> List[float]:
        logging_state = getattr(self.logger, "my_fantastic_logging", None)
        if isinstance(logging_state, dict):
            return list(logging_state.get(self.early_stop_metric, []))

        if hasattr(self.logger, "get_checkpoint"):
            checkpoint_state = self.logger.get_checkpoint()
            if isinstance(checkpoint_state, dict):
                return list(checkpoint_state.get(self.early_stop_metric, []))

        return []

    def _reset_early_stopping_state(self) -> None:
        self._early_stop_best = None
        self._early_stop_bad_epochs = 0
        self._early_stop_triggered = False

        for score in self._metric_history():
            if score is None or not np.isfinite(score):
                continue
            if self._early_stop_best is None or score > (self._early_stop_best + self.early_stop_min_delta):
                self._early_stop_best = float(score)
                self._early_stop_bad_epochs = 0
            else:
                self._early_stop_bad_epochs += 1

    def load_checkpoint(self, filename_or_checkpoint):
        super().load_checkpoint(filename_or_checkpoint)
        if self._early_stop_enabled:
            self._reset_early_stopping_state()

    def on_epoch_end(self):
        super().on_epoch_end()
        if not self._early_stop_enabled:
            return

        metric_history = self._metric_history()
        if not metric_history:
            return

        current_score = metric_history[-1]
        if current_score is None or not np.isfinite(current_score):
            self.print_to_log_file(
                f"Early stopping metric {self.early_stop_metric} is not finite at epoch {self.current_epoch - 1}; ignoring."
            )
            return

        if self._early_stop_best is None or current_score > (self._early_stop_best + self.early_stop_min_delta):
            self._early_stop_best = float(current_score)
            self._early_stop_bad_epochs = 0
            return

        self._early_stop_bad_epochs += 1
        self.print_to_log_file(
            f"Early stopping monitor {self.early_stop_metric}: no improvement for "
            f"{self._early_stop_bad_epochs}/{self.early_stop_patience} epochs "
            f"(best={self._early_stop_best:.6f}, current={float(current_score):.6f}, min_delta={self.early_stop_min_delta})."
        )
        if self._early_stop_bad_epochs >= self.early_stop_patience:
            self._early_stop_triggered = True
            self.print_to_log_file(
                f"Early stopping triggered at epoch {self.current_epoch - 1} based on validation metric "
                f"{self.early_stop_metric}."
            )

    def run_training(self):
        self.on_train_start()

        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self._early_stop_triggered:
                break

        self.on_train_end()
