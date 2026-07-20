# Script entrypoints

## Production Python entrypoints

- `check_gadi_ready.py`: validate active configs, PBS references, Early
  Stopping policy, server paths, plans, and trainer imports
- `run_experiment.py`: central config-driven training, inference, evaluation,
  metadata, and resource-monitoring CLI
- `install_extension.py`: install the active `nnunet25d` package and trainer
  discovery shims into the current Python environment
- `create_bhsd_binary_dataset.py`: create Dataset002 without modifying
  Dataset001
- `evaluate_binary_segmentation.py`: evaluate binary predictions

## Gadi usage

PBS jobs under `hpc/gadi/` are the production launchers. They install the
extension under a shared lock and then call `run_experiment.py` with an active
config.

After pulling new code on Gadi:

```bash
python scripts/check_gadi_ready.py --server
```

Use `--require-binary` after Dataset002 preparation when checking binary jobs.

## Legacy helpers

The remaining `.ps1`, `.cmd`, and generic `.sh` files support older local or
AWS workflows. They are not the recommended Gadi submission path.
