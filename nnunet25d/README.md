# nnU-Net 2.5D Extensions

This package stores custom nnU-Net v2 extensions used for BHSD experiments.

Supported trainers:

- `nnUNetTrainer_25D`
- `nnUNetTrainer_25D_5Slice`
- `nnUNetTrainer_SpacingAware25D`
- `nnUNetTrainer25DCSAMOfficial`
- `nnUNetTrainer25DCSAMOfficialNoUncertainty`
- `nnUNetTrainerCSAMVolumeOfficial` (ordered overlapping 32-slice windows)
- `nnUNetTrainer25DCSANetOfficial` (official previous/center/next CSA-Net)

These classes are installed into the active nnU-Net environment through `scripts/install_extension.py`.

The spacing-aware dataloader uses `bhsd_spacing_summary.csv` to choose a case-specific slice step in z so that adjacent context better reflects physical spacing.
The official CSAM integration vendors the published CSAM modules and networks, then uses a thin center-slice wrapper so the current BHSD 2.5D dataloader and inference path remain usable.

The two fold-0 paper-based attention pilots are documented in
`hpc/gadi/ATTENTION_FOLD0.md`. Their vendored-source provenance and deliberate
compatibility fixes are recorded in `csam/SOURCE_VOLUME.md` and
`csa_net/vendor/SOURCE.md`.
