"""nnU-Net recursive class-discovery shim for preregistered Stage3 trainers."""

from nnunet25d.stage3.trainer import (
    nnUNetTrainer_Stage3_R0,
    nnUNetTrainer_Stage3_R0Seed1234,
    nnUNetTrainer_Stage3_R0Seed5678,
    nnUNetTrainer_Stage3_R1,
    nnUNetTrainer_Stage3_R1Seed1234,
    nnUNetTrainer_Stage3_R1Seed5678,
)

__all__ = [name for name in globals() if name.startswith("nnUNetTrainer_Stage3_")]
