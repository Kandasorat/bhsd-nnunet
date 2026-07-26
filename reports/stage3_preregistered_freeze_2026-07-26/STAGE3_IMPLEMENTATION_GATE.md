# Stage3 implementation and pre-training gate

Date: 2026-07-26

## Outcome

The locked B/R0/R1 implementation is complete and all 25 code-only synthetic/static preflight tests pass. No historical checkpoint was used for a forward pass, no patient inference or performance evaluation was run, no training was started, and no PBS job was submitted.

Training is not approved by this package.

## Implemented scope

- one frozen center-only nnU-Net backbone, permanently held in `eval()` and executed under `torch.no_grad()`;
- exactly equal R0/R1 residual branches with 18,342 trainable parameters;
- exact post-augmentation center duplication for R0 and true `[z-1,z,z+1]` input for R1;
- one complete center-backbone pass and three shared shallow-stem evaluations;
- zero-initialized logit residual at every deep-supervision scale;
- Stage3-only shared intensity augmentation, including a shared all-channel application decision;
- hard guards against fold0 training/performance reporting and against training without a later approval token;
- physical-space HD95, NSD@3mm, 26-connected lesion matching, small-lesion recall, absent-class FP, and patient-cluster bootstrap code;
- locked fold/seed-specific trainer namespaces and checkpoint-hash validation.
- six immutable fold1 YAML configs whose generated commands explicitly include
  `--val_best --npz`;
- a Gadi V100-only preflight PBS job and a separately submitted six-element
  fold1 training array that refuses a missing/stale/failed gate and implicit
  resume;
- a checksum-verifying PowerShell helper for uploading the four frozen B
  checkpoints to their stable Gadi input directory.

## Preflight evidence

- Required tests: 25/25 passed.
- Split: 192 cases, five disjoint folds, each case appears in validation exactly once; locked SHA-256 matched.
- B checkpoint files: folds1-4 exist and their SHA-256 values matched the preregistration.
- Real fold1 R0/R1 trainer initialization passed strict checkpoint loading with no forward calls; both loaded the same B hash, retained identical residual initialization hashes, froze all 46,332,650 center parameters, and exposed only 18,342 residual parameters to the optimizer.
- Residual parameters: 18,342 for R0 and R1.
- The six config audit proves every generated command contains `--npz` and
  `--val_best`, uses fold1, retains checkpoints, and has the locked arm/seed
  mapping.
- Full randomly initialized architecture dry-run on local NVIDIA GeForce RTX 3060 Laptop GPU, batch1, 256x256, autocast, deterministic cuDNN, 20 warm-up and 100 measured repetitions:
  - B: 46,332,650 parameters; 28,580,024,320 Conv2d FLOPs; median 9.835 ms.
  - R0/R1: 46,350,992 total and 18,342 trainable parameters; 31,603,462,144 Conv2d FLOPs; one full backbone pass.
  - R1: median 13.517 ms.
  - R1/B latency ratio: 1.3744; R1/B peak-memory ratio: 0.7059.
- The local server-gate dry-run exited with the predeclared refusal code because
  the device is not V100, the working tree is not yet committed/clean, and the
  latency ratio exceeds 1.25. This confirms fail-closed behavior.

The local memory constraint passes, but the local latency ratio exceeds the preregistered 1.25 limit. This is an architecture-only random-weight measurement, not historical-model inference. Because planned experiments run on V100 hardware, the exact frozen measurement script must be executed there before training approval. The threshold must not be changed.

## Remaining blockers before any training

1. Commit and push the complete implementation/config/PBS set together; no
   implementation commit is currently recorded.
2. Upload and remotely verify the locked fold1-4 B checkpoints.
3. Run `stage3_v100_preflight.pbs` on the exact clean commit. The latency ratio
   must be at most 1.25 and memory ratio at most 1.30.
4. Only after that gate passes may the user separately submit the locked
   six-element fold1 array.

## Decision boundary

Do not train while any blocker above remains. Do not redesign the model based on this compute result. If the locked architecture fails the same latency gate on the target V100, Stage3 stops unless the preregistration is formally withdrawn before observing segmentation performance; any redesigned method would be a new protocol.
