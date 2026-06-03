# nnU-Net 2.5D Extensions

This package stores custom nnU-Net v2 extensions used for BHSD experiments.

Supported trainers:

- `nnUNetTrainer_25D`
- `nnUNetTrainer_25D_5Slice`
- `nnUNetTrainer_SpacingAware25D`

These classes are installed into the active nnU-Net environment through `scripts/install_extension.py`.

The spacing-aware dataloader uses `bhsd_spacing_summary.csv` to choose a case-specific slice step in z so that adjacent context better reflects physical spacing.
