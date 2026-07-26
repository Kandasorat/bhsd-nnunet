# Fixed1000 readiness gate incidents

## Gate 1 — 174755560.gadi-pbs

- Result: `FAIL`, PBS `Exit_status=1`.
- Runtime: 00:03:41 on Tesla V100-SXM2-32GB.
- Formal training submitted: 0/27.
- Successful evidence before failure: split hash matched; 2D and A0 instantiated; real-batch train, backward and validation losses were finite; peak allocated memory was approximately 1.32 GB and 1.40 GB; historical structure/parameter checks passed.
- Failure: `UnboundLocalError: local variable 'checkpoint' referenced before assignment` during the synthetic final/best checkpoint round-trip.
- Root cause: the preflight passed a `pathlib.Path` to the installed nnU-Net `load_checkpoint` method. That implementation assigns its local `checkpoint` variable only for a `str` argument. Formal nnU-Net CLI checkpoint loading uses strings and is not affected.
- Correction: pass `str(final_probe)` and `str(best_probe)` in the preflight.
- Timing correction: Gate 1 timed the first step. Standard 2D included approximately 65 seconds of one-time CUDA compilation, producing an invalid 4672-hour extrapolation. The corrected gate records compilation as warm-up and uses the median of three subsequent real train and validation batches. The locked 48-hour threshold is unchanged.
- Decision: Gate 1 remains a recorded failure. No formal array may run. A corrected immutable revision requires a new V100 gate; it is not treated as a continuation or silent rerun of Gate 1.

## Gate 2 — 174756749.gadi-pbs

- Revision: `a2820d1d0948b37f190c28c4fee76bd14823ad5b`.
- Result: `FAIL`; formal training submitted: 0/27.
- Passed checks: split SHA-256; final/best checkpoint round-trip; finite 2D and A0 training/validation losses; historical structure and parameter equality for completed profiles.
- 2D steady-state median: train step 1.2195 s, validation step 0.5925 s, projected 1000 epochs 92.92 h, peak allocated memory 1.50 GB.
- A0 steady-state median: train step 0.8249 s, validation step 0.4043 s, projected 1000 epochs 62.90 h, peak allocated memory 1.58 GB.
- Failure: 3D `torch.compile` failed inside TorchInductor/Triton with `CompilationError` / `TypeError('unexpected type fp32')` on the Gadi Tesla V100 environment.
- Consequences: both completed walltime projections exceed the locked 48-hour PBS request; the 3D runtime path is not viable in the current environment; C1/C2/D0/D1 and quota/resource aggregation were not reached.
- Decision: `READINESS_FAIL`. The two formal arrays are not authorized. Disabling compilation, changing data-worker policy, splitting each run across chained PBS jobs, requesting a different environment, or increasing walltime would be a protocol/runtime change and requires an explicit new decision and preregistration rather than another silent gate rerun.
