# Upstream CSAM architecture integration

This repository vendors the CSAM architecture from the authors' repository:

- https://github.com/aL3x-O-o-Hung/CSAM

The vendored code lives in:

- `nnunet25d/csam/CSAM_modules.py`
- `nnunet25d/csam/CSAM_networks.py`
- `nnunet25d/csam/LICENSE`

The nnU-Net-facing integration layer is intentionally thin:

- `nnunet25d/csam/official_wrapper.py`
- `nnunet25d/csam/trainer_official.py`

## Current Path

- config: `configs/csam_official_3slice.yaml`
- trainer: `nnUNetTrainer25DCSAMOfficial`
- verify script: `scripts/verify_csam_official.py`

## Important Note

The official CSAM network itself predicts one logit map per input slice.
The current BHSD 2.5D wrapper keeps the existing center-slice supervision pipeline by:

1. reshaping one batch item into `[K, C, H, W]`
2. running the official CSAM network on that slice stack
3. returning only the center-slice logits to nnU-Net training/inference

This means the attention and backbone are upstream-code-based, while the final
center-slice reduction and nnU-Net training protocol are BHSD compatibility
adaptations. This is not a source-faithful reproduction of the paper's training
protocol. See `docs/ATTENTION_REPRODUCTION_POLICY.md`.
