"""Custom 2.5D nnU-Net extensions for BHSD research experiments."""

from nnunet25d.baseline.trainer_25d import (
    nnUNetTrainer_25D,
    nnUNetTrainer_25D_HarmonizedMin300Patience100,
    nnUNetTrainer_25D_5Slice,
    nnUNetTrainer_SpacingAware25D,
)
from nnunet25d.csam import (
    C2BAMUNet,
    CSAM,
    EncoderCSAM,
    OfficialCSAMCenterSliceWrapper,
    PositionalAttentionModule,
    SemanticAttentionModule,
    SliceAttentionModule,
    nnUNetTrainer25DCSAMOfficial,
    nnUNetTrainer25DCSAMOfficialNoUncertainty,
)

__all__ = [
    "C2BAMUNet",
    "CSAM",
    "EncoderCSAM",
    "OfficialCSAMCenterSliceWrapper",
    "PositionalAttentionModule",
    "SemanticAttentionModule",
    "SliceAttentionModule",
    "nnUNetTrainer_25D",
    "nnUNetTrainer_25D_HarmonizedMin300Patience100",
    "nnUNetTrainer_25D_5Slice",
    "nnUNetTrainer_SpacingAware25D",
    "nnUNetTrainer25DCSAMOfficial",
    "nnUNetTrainer25DCSAMOfficialNoUncertainty",
]
