from nnunet25d.csam.trainer_official import (
    nnUNetTrainer25DCSAMOfficial,
    nnUNetTrainer25DCSAMOfficialNoUncertainty,
)

# Compatibility aliases for environments that still import the old shim name.
nnUNetTrainer25DCSAM = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DCSAMBottleneck = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DCSAM_5Slide = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DFeatureFusion = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DFeatureFusionBottleneck = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DFeatureFusionMultiScale = nnUNetTrainer25DCSAMOfficial
nnUNetTrainer25DFeatureFusionMultiScale_5Slice = nnUNetTrainer25DCSAMOfficial

__all__ = [
    "nnUNetTrainer25DCSAM",
    "nnUNetTrainer25DCSAMBottleneck",
    "nnUNetTrainer25DCSAM_5Slide",
    "nnUNetTrainer25DFeatureFusion",
    "nnUNetTrainer25DFeatureFusionBottleneck",
    "nnUNetTrainer25DFeatureFusionMultiScale",
    "nnUNetTrainer25DFeatureFusionMultiScale_5Slice",
    "nnUNetTrainer25DCSAMOfficial",
    "nnUNetTrainer25DCSAMOfficialNoUncertainty",
]
