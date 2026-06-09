# Verify `csam_3slide`

## Summary

`csam_3slide` is the new 2.5D multi-scale center-guided slice attention model for BHSD.

It implements:

- three adjacent slices as input
- a shared 2D encoder
- multi-scale cross-slice feature fusion with center-guided slice attention
- center-slice segmentation output only

It is separate from:

- standard 2D nnU-Net
- standard 3D full-resolution nnU-Net
- the legacy simple 3-slide 2.5D channel-stacking baseline

## Current Code Path

- config: `configs/csam_3slide.yaml`
- trainer alias: `nnUNetTrainer25DCSAM`
- trainer implementation: `nnunet25d/csam/trainer_25d_feature_fusion.py`
- network implementation: `nnunet25d/csam/feature_fusion_25d.py`
- shared dataloader path: `nnunet25d/common/dataloader_25d.py`

## Code Fixes Applied

The following CSAM-specific fixes were applied:

1. Weight initialization now uses recursive `module.apply(InitWeights_He(1e-2))`.
2. Unsafe `view()` calls in the feature-fusion path were replaced with `reshape()`.
3. Saved attention weights are detached before being cached in `last_attention_weights`.
4. Architecture argument resolution now raises a clear `ImportError` if `locate()` fails.
5. No softmax or sigmoid is applied in the network output path.

## Local Verification Result

`scripts/verify_csam_3slide.py` was run successfully in the local BHSD environment.

Observed results:

- direct import of `nnUNetTrainer25DCSAM`, `MultiScaleFeatureFusion25DUNet`, and `CenterGuidedSliceFusion` passed
- shim import through `nnunetv2.training.nnUNetTrainer.trainer_25d_feature_fusion` also passed locally
- `K=3` full-resolution output shape was `[2, 6, 256, 256]`
- `K=5` full-resolution output shape was `[2, 6, 256, 256]`
- deep supervision returned the expected multi-output pyramid with first output `[2, 6, 256, 256]`
- CPU forward/backward passed
- CUDA forward/backward passed locally
- legacy simple 3-slide baseline trainer `nnUNetTrainer_25D` still initialized correctly

## Input / Output Shape Trace

For BHSD single-channel CT with `K=3`:

- dataloader training batch: `[B, 3, H, W]`
- `_prepare_input`: `[B, 3, 1, H, W]`
- encoder input: `[B * 3, 1, H, W]`
- highest-resolution logits: `[B, 6, H, W]`

For `K=5`:

- dataloader-style input: `[B, 5, H, W]`
- highest-resolution logits: `[B, 6, H, W]`

If deep supervision is enabled, the first output remains the full-resolution logits and additional lower-resolution outputs follow the standard nnU-Net convention.

## Encoder Input Channel Confirmation

For `nnUNetTrainer25DCSAM` on BHSD:

- `base_num_input_channels = 1`
- `self.num_input_channels = 3`
- `self.num_input_channels_per_slice = 1`
- `FeatureFusion25DUNet` is instantiated with `input_channels=1`

This is correct:

- dataloader output is stacked 3-channel input
- the CSAM model internally reshapes that into 3 slices with 1 channel each
- the shared encoder therefore still operates on 1-channel single-slice features

## Multi-Scale Fusion Confirmation

`csam_3slide` maps to:

- trainer: `nnUNetTrainer25DCSAM`
- network class: `MultiScaleFeatureFusion25DUNet`
- fusion mode: `multiscale`

This means all encoder stages listed in `self.encoder.output_channels` are fused, not just the bottleneck.

If you want the cleaner ablation against the old baseline, use:

- `csam_bottleneck_3slide`

## Deep Supervision Status

Deep supervision is supported and follows the normal nnU-Net behavior.

Verification confirmed:

- `deep_supervision=False`: single logits tensor or equivalent single primary output
- `deep_supervision=True`: standard nnU-Net tuple/list with the first output at full resolution

## Attention Weights

After a forward pass:

- `csam_3slide` stores `last_attention_weights` as a `dict`
- keys correspond to encoder stage indices
- each tensor shape is `[B, K]`

For the bottleneck-only model:

- `last_attention_weights` is a single tensor of shape `[B, K]`

Attention tensors are now detached before caching, so:

- `attention.requires_grad == False`

## Validation / Prediction Compatibility

This was checked carefully because it is a potential blocking point.

### Training-time online validation

The 2.5D dataloader provides:

- adjacent-slice stacked input
- center-slice target

So the training path is correct for `csam_3slide`.

### Formal nnU-Net validation

The custom 2.5D base trainer overrides `perform_actual_validation()`.

That method:

1. loads a full case as `(C, Z, Y, X)`
2. converts it with `_stack_case_for_inference()` into `(C * K, Z, Y, X)`
3. uses boundary clamping at the first and last slices
4. runs sliding-window prediction over the stacked volume
5. exports a full 3D prediction volume

So formal trainer-driven validation is using adjacent-slice 2.5D input, not a broken single-slice path.

### Boundary handling

First and last slices are handled by edge clamping / replication behavior through index clipping:

- `z = 0` uses `[0, 0, 1]`
- last slice uses `[Z-2, Z-1, Z-1]`

### Important limitation

The repository still does not provide a dedicated standalone `run_experiment.py infer` pipeline specialized for the custom 2.5D trainers.

So:

- trainer-driven validation is verified
- ad hoc external inference pipelines should still be treated carefully

This is not a blocker for trainer-based training plus `--val`, but it is still a limitation worth keeping visible.

## Installation / Import Instructions

The current verified installation method is:

```bash
python scripts/install_extension.py
```

This is the preferred path right now because the repository does not yet define a packaging setup such as:

- `pyproject.toml`
- `setup.py`
- `setup.cfg`

So `pip install -e .` is not currently configured as a supported path.

The install script now copies:

- all required `nnunet25d` package files recursively
- `__init__.py` files
- CSAM package files
- the nnU-Net trainer shim

Verified import targets to check:

```bash
python -c "from nnunet25d.csam.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM; print(nnUNetTrainer25DCSAM)"
python -c "from nnunet25d.csam.feature_fusion_25d import MultiScaleFeatureFusion25DUNet, CenterGuidedSliceFusion; print(MultiScaleFeatureFusion25DUNet, CenterGuidedSliceFusion)"
```

If the shim has been installed into the active nnU-Net environment, this should also work:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.trainer_25d_feature_fusion import nnUNetTrainer25DCSAM; print(nnUNetTrainer25DCSAM)"
```

## Exact Verification Command

Run:

```bash
python scripts/verify_csam_3slide.py
```

Optional existing smoke test:

```bash
python smoke_test_25d_feature_fusion.py
```

## Known Limitations

1. No editable-install package configuration is present yet.
2. Custom 2.5D standalone inference outside the trainer validation path still needs careful handling.
3. Attention weight persistent logging is not added yet.
4. Binary Dice evaluation and report integration are still pending later stages.

## Safety Status Before Server Training

`csam_3slide` should only be considered safe to train after `scripts/verify_csam_3slide.py` passes in your active environment.

Current local status:

- importable
- trainable
- forward/backward verified
- detached attention verified
- trainer channel logic verified
- trainer-driven formal validation path explicitly verified

Based on the current local checks, `csam_3slide` is safe to move to server training, provided you first:

1. `git pull origin main`
2. `python scripts/install_extension.py`
3. `python scripts/verify_csam_3slide.py`

If step 3 fails on the server, do not start the full run yet.
