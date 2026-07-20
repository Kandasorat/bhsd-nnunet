# Project Structure

## Active project

The active codebase now centers on these locations:

- `nnunet25d/` for current custom 2.5D and upstream-architecture attention
  adaptations
- `configs/` for active experiment configs only
- `scripts/` for active run and verification entrypoints
- `final_project/` for a curated map of the current recommended project entrypoints
- `hpc/gadi/` for the only supported production PBS submission scripts

## Archive

Historical and transitional assets are moved into `archive/` with simple version-like labels:

- `1.1_*` early minimal 2.5D extension state
- `1.2_*` legacy naive/feature-fusion 2.5D config state
- `2.1_*` old feature-fusion smoke-test state

## Runtime data

These are runtime/output areas, not source-code modules:

- `nnUNet_data/`
- `results/`
- `outputs/`

## Main rule going forward

- update the active code in place
- keep superseded implementations in the labelled archive or the
  `archive/pre-consolidation` Git branch instead of leaving parallel active
  folders beside the main implementation
- keep new configs/scripts only if they represent the active recommended workflow
