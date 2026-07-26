# Stage 3 required tests

Status: test specification only. The Stage3 B/R0/R1 wrapper and loader do not yet exist, so none of the tests in this file may be marked passed by the current audit.

All tests must live in a new isolated Stage3 test directory. They may use synthetic tensors and a temporary copy of a small checkpoint, but must not train a model, run full validation, overwrite historical outputs, or submit PBS.

| ID | Required test | Exact pass condition |
|---|---|---|
| S3-01 | Zero residual identity | With final residual weight/bias zero, B, R0, and R1 logits are bit-for-bit equal at every deep-supervision scale for the same center input. |
| S3-02 | Delta-off identity | Setting `delta_enabled=0` restores the original loaded 2D logits bit-for-bit for at least 10 random tensor shapes accepted by the plan. |
| S3-03 | Frozen center gradients | After forward/backward, every center parameter has `grad is None`; every expected residual parameter has a finite gradient once the zero output layer has received an update in a synthetic two-step test. |
| S3-04 | Frozen center state | SHA-256 of a deterministic serialization of every center parameter and buffer is identical before and after an optimizer step and after `wrapper.train()`. Center remains in evaluation mode. |
| S3-05 | Optimizer membership | The optimizer parameter-ID set equals the residual-branch trainable parameter-ID set and has empty intersection with the center parameter-ID set. |
| S3-06 | R0/R1 capacity | Total and trainable parameter counts for R0 and R1 are exactly equal; expected residual count is independently checked against 18,342. |
| S3-07 | Complete backbone passes | A forward hook on the complete center model counts exactly one call for B, R0, and R1. R0/R1 each call the shared shallow stem three times. |
| S3-08 | R0 input | At the residual-branch boundary, all three R0 slice tensors are exactly equal to the augmented center tensor. |
| S3-09 | R1 input order | With unique-valued synthetic slices, the residual boundary receives `[z-1,z,z+1]`; GT contains only z. |
| S3-10 | Boundary replication | At z=0 R1 receives `[0,0,1]`; at the last slice it receives `[last-1,last,last]`. R0 remains `[z,z,z]`. |
| S3-11 | Neighbour-swap rule | Swapping R1 neighbours leaves `n`, `delta_full`, and all final logits bit-for-bit equal. If only tolerance equality is technically possible, the predeclared maximum absolute error is `1e-7` on CPU float32. |
| S3-12 | Shared geometry | At least 32 randomized affine/mirror synthetic trials apply the same sampled geometry to all three slices and GT; channel landmarks and GT landmarks agree within one output pixel. |
| S3-13 | Shared intensity | Starting with three identical images, every enabled Stage3 intensity transform produces identical output channels in 100 fixed-seed trials. Inspect transform objects to assert every applicable `synchronize_channels=True`. |
| S3-14 | Six-class and deep supervision shapes | Training output has six channels at every shape specified by the nnU-Net deep-supervision scales; inference output is exactly `[N,6,H,W]`. Resized delta tensors exactly match each B logit shape. |
| S3-15 | Checkpoint identity | R0 and R1 for a fold load the same declared `checkpoint_best.pth` SHA-256 and reject final/latest checkpoints. A deliberately wrong hash must make the test fail. |
| S3-16 | Seed propagation | A child-process probe records model seed, data seed, Python hash seed, NumPy/PyTorch/CUDA seeds, deterministic flags, and `nnUNet_n_proc_DA=0`; values must match the config. |
| S3-17 | Paired data stream | R0/R1 with the same seed emit the same first 100 `(case_id,center_z,crop,geometry_params,intensity_params)` audit records. Only the materialized neighbour tensor may differ. |
| S3-18 | Split isolation | All 192 cases match the locked split hash; each fold train/val intersection is empty; R0/R1/B validation IDs are identical. |
| S3-19 | Metric empty cases | Synthetic cases prove: both empty Dice is NaN/excluded; GT present/pred empty Dice=0; GT absent/pred present is absent FP; class-balanced and pooled present macros differ on a constructed imbalanced example. |
| S3-20 | HD95 physical spacing | Perfect masks give 0mm; a one-voxel shift along spacing 5mm gives 5mm; GT-present/pred-empty returns the physical image diagonal; generic voxel Hausdorff cannot be imported by the Stage3 evaluator. |
| S3-21 | NSD@3mm | Perfect masks give 1; surfaces 2.9mm apart count within tolerance and 3.1mm do not; pred-empty with GT-present gives 0. |
| S3-22 | Lesion matching | Synthetic 26-connected components verify one-to-one Hungarian matching, a merged prediction cannot match two GT lesions, and `<1mL` is applied per GT component in physical volume. |
| S3-23 | Geometry and case join | A 20-case read-only sample plus all Stage3 summaries must match prediction/GT case ID, shape, spacing, origin, direction, and labels 0-5 before metrics. |
| S3-24 | Compute measurement | Parameter count, FLOPs, peak allocated memory, median latency after fixed warm-up/repetitions, and backbone-pass hook are emitted for B/R0/R1 under one device/batch/shape. R0/R1 counts must match exactly. |
| S3-25 | No fold0 performance path | The Stage3 fold0 test command refuses any request to save a trained checkpoint or compute/report Dice/HD95/NSD. Only synthetic/smoke outputs are permitted. |

The implementation test runner must exit nonzero on any failure, save JSON plus plain-text logs, list software/device versions, and write SHA-256 for itself and all generated test artifacts. Passing these tests is necessary but not sufficient for training approval.
