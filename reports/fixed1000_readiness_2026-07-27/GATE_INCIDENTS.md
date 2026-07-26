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

