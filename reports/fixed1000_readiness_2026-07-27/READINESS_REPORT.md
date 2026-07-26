# Fixed1000 readiness report

Decision: **READINESS_FAIL**

Formal training jobs submitted: **0 / 27**

## Passed locally

- The isolated worktree was fast-forwarded to authoritative parent `624235ea347bdc709091890376a4d3cdc628de06` before implementation.
- Exactly 27 configs parse: core 15 and fold-0 diagnostic 12; array indices are complete and unique.
- All configs lock split SHA-256 `A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA`.
- Eighteen standard-library protocol tests pass.
- No fixed trainer MRO contains `BHSDEarlyStoppingMixin`; the standard fixed-length nnU-Net loop is used.
- PolyLR values at epochs 0/500/900/999 follow horizon 1000 and exponent 0.9.
- A0/C1/C2/D0/D1 state-dict signatures, parameter counts and seed-3407 initialization hashes exactly match their historical implementations.
- Parameter counts: A0 46,333,226; C1 46,333,323; C2 46,335,559; D0 and D1 46,333,338.
- Same-seed initialization reproduces; model seed 1234 differs from seed 3407.
- 2D/A0 is batch 12 at 256×256; 3D keeps batch 2 and patch 28×256×256.
- A0 order and boundary checks give `[0,0,1]`, `[1,2,3]`, `[2,3,3]` for representative edge/interior indices.
- New result namespaces contain `Fixed1000`, `NoEarlyStopping`, `FinalCheckpointPrimary` and the model seed. No historical, CSA-Net/CSAM or publication-v3 result namespace is referenced.
- The runner fails if the target result directory already exists, never adds `--c`, validates final first, then validates best into a separate sensitivity directory, with NPZ required for both.

Evidence: `local_architecture_audit.json`, `tests/fixed1000/test_fixed1000_protocol.py`, and `preregistered_fixed1000_run_matrix.csv`.

## Gate 1 incident

Gadi job `174755560.gadi-pbs` finished with `Exit_status=1`. It passed the split, 2D and A0 real-batch forward/backward/validation checks before a preflight-only checkpoint round-trip call passed a `Path` object to an nnU-Net method that requires `str`. The incident is recorded in `GATE_INCIDENTS.md`; no formal job was submitted.

The first-step timing also included one-time CUDA compilation and is not a valid 1000-epoch projection. The corrected preflight retains the same resource limit and measures the median of three post-warm-up real batches.

## Gate 2 result

Corrected Gadi job `174756749.gadi-pbs` also returned `FAIL`. Split and checkpoint-final/checkpoint-best round-trip passed. The completed 2D and A0 profiles had finite losses and low peak allocated memory, but projected 1000-epoch walltimes were approximately 92.92 h and 62.90 h respectively, above the locked 48-hour request.

The gate then stopped during 3D TorchInductor/Triton compilation with `TypeError('unexpected type fp32')`. It consequently did not profile C1/C2/D0/D1 or complete total GPU-hour/quota evaluation.

## Unresolved evidence

The following remain unresolved and prevent `PASS`:

- successful real-data batch training, backward and validation evidence for 3D and C1/C2/D0/D1;
- peak V100 memory and per-iteration timing;
- projected per-task walltime and 27-task GPU-hours;
- current scratch quota versus a provisional 243 GB projection / 270 GB reserve;
- confirmation that requested 48 h, 32 GB, 12 CPU and one V100 are sufficient;
- final Gadi clean-commit and formal-namespace absence checks.

The previous 2D/A0/3D audit estimated the core 15 at approximately 122–125 V100 GPU-hours including full validation and recommended at least 110 GB for one validation set per run. This new protocol stores both final and best probabilities, so 243 GB with a 270 GB reserve is a deliberately conservative provisional estimate pending V100/quota evidence.

## Decision boundary

Current status is `READINESS_FAIL`; consequently neither formal PBS array may be submitted. No threshold is relaxed and no third gate is authorized by this decision. Any proposal to disable `torch.compile`, introduce explicitly seeded multi-worker augmentation, chain jobs across checkpoints, change the software environment, or alter requested walltime must be separately chosen and preregistered before implementation.
