# BHSD nnU-Net / CSAM Project Handoff

> **Authoritative current snapshot: 2026-07-21**
>
> Read this section first. The older Google Cloud material later in this file is
> retained only as historical context. If it conflicts with this section, this
> Gadi snapshot takes precedence.

## New-chat bootstrap

Start a new Codex conversation with:

```text
Please read C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md completely.
Treat the "Authoritative current snapshot: 2026-07-21" section as the source of truth,
then continue the Gadi BHSD nnU-Net project from the recorded current status.
Do not assume that the historical Google Cloud instructions are still current.
```

Primary handoff file:

- `C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md`

Git-repository copy:

- `C:\Users\92127\OneDrive - UNSW\project_linpeng\code\docs\PROJECT_HANDOFF.md`

## Current objective and status

Run reproducible BHSD intracranial-haemorrhage segmentation experiments on NCI
Gadi. Two task definitions are intentionally kept separate:

1. Multiclass: background plus EDH, IPH, IVH, SAH, and SDH
   (`Dataset001_BHSD`).
2. Binary: background versus any haemorrhage region
   (`Dataset002_BHSD_Binary`).

Completed:

- multiclass 2D nnU-Net baseline, folds 0-4;
- multiclass 3D full-resolution nnU-Net baseline, folds 0-4;
- local verification of all five best 2D and 3D checkpoints;
- implementation of paper-based volume-wise CSAM and official three-slice
  CSA-Net fold-0 pilots;
- implementation of reproducible binary 2D/3D five-fold Gadi workflows;
- preparation and preprocessing of `Dataset002_BHSD_Binary`, with the same
  five-fold split as Dataset001 and verified 2D/3D plans.

Not yet run at this snapshot:

- completed binary 2D and 3D five-fold training (the first arrays were
  deliberately cancelled and their incomplete result directories removed);
- volume-wise CSAM fold 0;
- CSA-Net fold 0.

All models use the same model-selection policy unless a separately declared
ablation changes it.

## Current platform and access

- Platform: NCI Australia Gadi.
- Username: `ly6399`.
- NCI project: `ke17`.
- Login command from Windows PowerShell:

  ```powershell
  ssh ly6399@gadi.nci.org.au
  ```

- Data-transfer host: `gadi-dm.nci.org.au`.
- GPU queue: `gpuvolta`.
- GPU verified: NVIDIA Tesla V100-SXM2-32GB.
- PBS allocates GPU compute nodes; do not train on a `gadi-login-XX` node.
- The login-node number, such as `gadi-login-03`, is not important.

## NCI allocation and storage

Last verified:

- Compute grant: 200.00 KSU for `2026.q3`.
- Used: 1.16 KSU; available: 198.84 KSU; reserved: 0.00 SU.
- Scratch allocation: 1.00 TiB and 202,000 files.
- Scratch used: 81.37 GiB and approximately 47.72K files.
- Shared root: `/scratch/ke17/bhsd-nnunet`.
- `/g/data/ke17` was not available and must not be assumed.
- Scratch is working storage and may be purged; important results require an
  additional approved UNSW backup.
- Project directories use group `ke17`, setgid, and `umask 007`.

## Current Gadi layout

```text
/scratch/ke17/bhsd-nnunet/
├── software/bhsd-nnunet/             # Git checkout
├── envs/bhsd-nnunet-py310/           # Python environment
├── data/
│   ├── archives/
│   ├── nnUNet_raw/
│   │   ├── Dataset001_BHSD/
│   │   └── Dataset002_BHSD_Binary/
│   └── nnUNet_preprocessed/
│       ├── Dataset001_BHSD/
│       └── Dataset002_BHSD_Binary/
├── runs/
│   ├── nnUNet_results/
│   └── experiment_metadata/
├── logs/
├── pbs/
├── manifests/
├── staging/
└── cache/
```

Environment variables:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"
```

## Verified environment

- Python module: `python3/3.10.4`.
- Virtual environment: `/scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310`.
- PyTorch: `2.7.1+cu118`.
- PyTorch CUDA runtime: `11.8`.
- nnU-Net v2: `2.6.4`.
- CUDA availability and V100 matrix smoke test: passed.

Activation:

```bash
module purge
module load python3/3.10.4
source /scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310/bin/activate
```

## Dataset and plans

- Dataset: `Dataset001_BHSD`.
- 192 training images and 192 labels.
- Five folds in `splits_final.json`.
- Fold sizes: approximately 153/39 or 154/38 train/validation.
- 2D patch: `256 x 256`; batch size 12.
- 3D full-resolution patch: `28 x 256 x 256`; batch size 2.
- Full-case validation can show `512 x 512` image dimensions; this does not
  change the training patch size.

## Locked training/checkpoint policy

Use consistently for 2D, 3D, 2.5D, and CSAM:

- Maximum 1000 epochs.
- Early-stopping minimum training duration 300 epochs.
- Early-stopping patience 100 epochs, counted only after epoch 300.
- Minimum improvement 0.0001.
- Monitoring metric `ema_fg_dice`.
- Primary weights: `checkpoint_best.pth`.
- Full-case validation: `--val_best`.
- Save probabilities: `--npz`.
- Retain `checkpoint_final.pth` for audit, not as the primary reported model.

This is the policy for new experiments from 2026-07-21 onward. The already
completed multiclass 2D and 3D five-fold baselines used the earlier patience-50
implementation and must be reported as such; changing the repository config
does not retroactively change those checkpoints. Do not resume an old result
directory under the new policy. Use a clean result directory, or archive and
rename the old one first, whenever a method is deliberately rerun.

`checkpoint_best.pth` is selected using the fold's training-time validation
metric. It is not the validation dataset. The same fold is then used for
full-case validation; this pragmatic design can be mildly optimistic compared
with nested cross-validation and must be described transparently.

## Git and formal Gadi scripts

- GitHub: `https://github.com/Kandasorat/bhsd-nnunet.git`.
- Active branch: `main`.
- Gadi checkout: `/scratch/ke17/bhsd-nnunet/software/bhsd-nnunet`.

Key commits:

- `2f5548f`: patch-size override fix.
- `bfc881c`: reproducible Gadi baseline jobs.
- `f7adc7c`: rerunnable, checkpoint-safe Gadi arrays.
- `4567280`: paper-based volume CSAM and official CSA-Net fold-0 pilots.
- `4082e63`: binary 2D/3D Dataset002 preparation and five-fold Gadi jobs.
- `6faeb61`: standard three-slice 2.5D fold-0 Gadi pilot.
- Latest `main`: consolidated active workspace and min-300/patience-100 policy.

Formal scripts:

```text
hpc/gadi/train_2d_folds.pbs
hpc/gadi/train_3d_folds.pbs
hpc/gadi/README.md
```

They submit folds 0-4, request one V100 per fold, request 12 CPUs/32 GB RAM/
20 GB jobfs/48 hours, use `#PBS -r y`, detect checkpoints for safe resume,
serialise extension installation, and write outputs outside Git.

The Windows workspace was consolidated on 2026-07-21. Runtime data, model
weights, results, and local outputs are ignored by Git; active source and Gadi
scripts remain tracked; superseded material is retained under `archive/`.

## Completed 2D five-fold baseline

PBS array: `174157789[].gadi-pbs`.

All folds exited successfully with `Exit_status = 0`, ran on V100 nodes, and
produced best/final checkpoints plus full-case validation summaries. Stderr
contained only non-fatal PyTorch Inductor online-softmax warnings.

| Fold | Stop epoch | Mean validation Dice |
|---|---:|---:|
| 0 | 161 | 0.2420124567 |
| 1 | 245 | 0.2229470776 |
| 2 | 198 | 0.2485912328 |
| 3 | 346 | 0.2669322398 |
| 4 | 229 | 0.2235124472 |

```text
Five-fold Mean Dice = 0.240799 +/- 0.018457
```

Server results:

```text
/scratch/ke17/bhsd-nnunet/runs/nnUNet_results/Dataset001_BHSD/nnUNetTrainer_BHSDEarlyStop__nnUNetPlans__2d
```

## Verified local 2D backup

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups\2d_baseline_2026-07-20
```

Verified:

- 5 best checkpoints and 5 final checkpoints.
- 5 valid validation summaries.
- 10 PBS stdout/stderr files and experiment metadata.
- 644 files; approximately 35.03 GB / 32.62 GiB.
- `checkpoint_best_sha256.csv` exists.
- All five local best-checkpoint hashes match the manifest.
- Local Dice values match Gadi logs.

The local 2D backup is complete. Keep the Gadi copy while model comparison,
ensembling, and qualitative error analysis remain possible follow-up work.

## Completed multiclass 3D five-fold baseline

PBS array `174185152[].gadi-pbs` completed for all five folds and released its
V100 resources.

| Fold | Mean validation Dice |
|---|---:|
| 0 | 0.267546 |
| 1 | 0.301143 |
| 2 | 0.295452 |
| 3 | 0.303674 |
| 4 | 0.255343 |

Five-fold multiclass Dice (mean +/- sample SD):

| Metric | 2D | 3D |
|---|---:|---:|
| Macro foreground | 0.240799 +/- 0.018457 | 0.284632 +/- 0.021806 |
| EDH | 0.064668 +/- 0.037201 | 0.071781 +/- 0.023380 |
| IPH | 0.485073 +/- 0.062815 | 0.531283 +/- 0.092922 |
| IVH | 0.393000 +/- 0.023434 | 0.482229 +/- 0.084118 |
| SAH | 0.148408 +/- 0.036866 | 0.192744 +/- 0.030440 |
| SDH | 0.112848 +/- 0.024415 | 0.145121 +/- 0.023369 |

3D beats 2D on every fold and especially improves IVH, but multiclass
performance remains weak for rare EDH, SAH, and SDH. These are
cross-validation validation results, not independent test-set results.

Server results:

```text
/scratch/ke17/bhsd-nnunet/runs/nnUNet_results/Dataset001_BHSD/nnUNetTrainer_BHSDEarlyStop__nnUNetPlans__3d_fullres
```

Server size breakdown: approximately 32 GiB total, 28.1 GiB in five validation
folders, 1.7 GiB in five best checkpoints, and 1.7 GiB in five final
checkpoints. Do not delete the whole server directory yet because the local
backup does not contain all probability maps and prediction volumes.

Verified local 3D core backup:

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups\3d_baseline_2026-07-20
```

For every fold, `checkpoint_best.pth`, `validation/summary.json`, and
`progress.png` are present. All checkpoint ZIP containers are readable with
262 entries, all JSON files parse, all plots are valid 3000 x 5400 PNG files,
and the final local integrity check returned `ALL_COMPLETE=True`.

## Paper-based attention pilots ready in Git

Commit `4567280` contains two distinct multiclass fold-0 pilots:

1. `nnUNetTrainerCSAMVolumeOfficial`: ordered overlapping 32-slice windows
   from one volume, with full-volume validation coverage.
2. `nnUNetTrainer25DCSANetOfficial`: official previous/centre/next-slice
   CSA-Net, predicting the centre slice.

Shared rules are 256 x 256 patches, maximum 1000 epochs, minimum 300 epochs,
patience 100 after the minimum-duration warm-up, minimum delta 0.0001,
`ema_fg_dice`, and `checkpoint_best.pth` formal validation.

```text
hpc/gadi/train_csam_volume_fold0.pbs
hpc/gadi/train_csa_net_fold0.pbs
hpc/gadi/ATTENTION_FOLD0.md
```

CSA-Net additionally requires the official `R50+ViT-B_16.npz` at:

```text
/scratch/ke17/bhsd-nnunet/software/pretrained/R50+ViT-B_16.npz
```

Neither attention pilot had been submitted at this snapshot.

## Binary 2D/3D workflow: immediate continuation point

The user decided to obtain binary baselines because multiclass subtype
segmentation is weak and the final project may focus on any-haemorrhage
segmentation.

Binary training is separate and does not modify Dataset001. Dataset002 keeps
copied raw labels `0..5`, while its own `dataset.json` defines one region:

```text
hemorrhage = {1, 2, 3, 4, 5}
```

Separate raw, preprocessed, and result directories prevent task conflicts:

```text
Dataset001_BHSD          # multiclass
Dataset002_BHSD_Binary   # binary region task
```

Commit `4082e63` adds:

```text
hpc/gadi/prepare_binary_dataset.pbs
hpc/gadi/train_2d_binary_folds.pbs
hpc/gadi/train_3d_binary_folds.pbs
hpc/gadi/BINARY_BASELINES.md
```

The preparation job validates the binary region definition, runs planning and
preprocessing, copies the exact Dataset001 five-fold split, and aborts if the
3D in-plane patch is not 256 x 256.

Submission of the preparation job was not confirmed before this handoff. The
next chat must first pull `main` and submit only the preparation job:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
cd "$BHSD_ROOT/software/bhsd-nnunet"
git pull --ff-only origin main
git rev-parse --short HEAD   # expected: 4082e63

cd "$BHSD_ROOT/logs"
PREP_JOB=$(qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/prepare_binary_dataset.pbs")
echo "$PREP_JOB"
qstat -u "$USER"
```

Do not submit binary GPU arrays until the preparation job finishes with
`Exit_status = 0` and prints `Binary dataset preparation completed successfully`.

## Routine commands

```powershell
ssh ly6399@gadi.nci.org.au
```

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet

cd "$BHSD_ROOT/software/bhsd-nnunet"
git pull --ff-only origin main
git rev-parse --short HEAD
git status --short

qstat -u "$USER"
qstat -t "JOB_ID[].gadi-pbs"
qstat -x -t "JOB_ID[].gadi-pbs"
```

Submit formal arrays from `$BHSD_ROOT/logs`:

```bash
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_folds.pbs"
```

Do not resubmit a completed fold without a documented reason.

## Immediate next actions

1. Pull the latest GitHub `main` on Gadi and run
   `python scripts/check_gadi_ready.py --server --require-binary`.
2. Do not resubmit the completed multiclass 2D/3D arrays merely because the
   Early Stopping defaults changed; those results are historical baselines.
3. Before any new run, confirm that its result directory is absent or belongs
   to the same protocol. Never resume a patience-50 checkpoint as a
   min-300/patience-100 experiment.
4. Run the standard three-slice 2.5D fold-0 experiment under the new locked
   policy, using a clean output path.
5. Review that pilot before running the volume-wise CSAM fold-0 and official
   CSA-Net fold-0 pilots; run the two attention pilots separately.
6. Restart the binary 2D/3D arrays only when binary baselines are again the
   active priority. Dataset002 preparation does not need to be repeated unless
   its source data or plans change.
7. Download `checkpoint_best.pth`, `validation/summary.json`, `progress.png`,
   and experiment metadata after each completed run.

## Historical material below

The remaining sections describe the earlier Google Cloud/L4 workflow and prior
CSAM work. They are retained for provenance, but their paths, branch names,
commands, and "current status" statements are no longer authoritative.

---



## Purpose
This file is the single source of truth for:
- project background
- code/repo locations
- VM environment requirements
- current training setup
- official CSAM vs custom nnU-Net integration boundaries
- result locations
- recovery and rebuild steps

Use this file in future chats as the starting context.

---

## Quick Start For New Chats
Paste this to a new chat:

`Please read [PROJECT_HANDOFF.md](C:/Users/92127/Documents/Codex/2026-07-04/google-cloud-vm-csam-gcloud-cli/outputs/PROJECT_HANDOFF.md) first and continue from there.`

---

## Project Identity
- Project: BHSD segmentation experiments with nnU-Net, 2D, 3D, 25D, and official-CSAM integration
- Main local code folder: `C:\Users\92127\OneDrive - UNSW\project_linpeng\code`
- GitHub repo: `https://github.com/Kandasorat/bhsd-nnunet.git`
- Main working branch for CSAM integration: `codex/publish-csam`
- Cloud bucket: `gs://bhsd-nnunet-data-92127/`

---

## Current Status
- A new Google Cloud VM was rebuilt after the old VM was deleted.
- The environment was restored from Git + bucket + manual setup.
- Official CSAM was integrated into nnU-Net with custom bridge code.
- Official CSAM `fold_0` training completed successfully.
- Final validation log showed:
  - `Validation complete`
  - `Mean Validation Dice: 0.22685461106033036`

---

## Local / Remote Paths

### Local Windows
- Project root: `C:\Users\92127\OneDrive - UNSW\project_linpeng\code`

### Remote VM
- Repo: `~/project/repo`
- Data root: `~/project/data`
- Outputs root: `~/project/outputs`
- Experiment summaries: `~/project/repo/results`

### nnU-Net env vars on VM
- `nnUNet_raw=$HOME/project/data`
- `nnUNet_preprocessed=$HOME/project/data`
- `nnUNet_results=$HOME/project/outputs/nnUNet_results`

---

## Current Google Cloud VM Reference
These are the current known-good characteristics. Future VMs do not need to be identical, but should be functionally similar.

- Platform: Google Cloud Compute Engine
- Zone used in this run: `us-central1-c`
- OS: Debian GNU/Linux 13
- GPU: NVIDIA L4
- GPU driver: installed successfully, `nvidia-smi` works
- Access method: Google Cloud Console SSH
- Python on VM: Python 3.13.5 available system-wide
- Conda: Miniconda installed
- Main env: `bhsd-nnunet`
- Bucket access: VM service account must be able to read/write `gs://bhsd-nnunet-data-92127/`

### Minimum functional requirements for any future VM or platform
- Linux VM with SSH access
- NVIDIA GPU with working driver
- enough VRAM for official CSAM at `256x256`
- Python + conda
- Git access to the GitHub repo
- bucket/object storage access
- enough disk for:
  - repo
  - dataset zips
  - extracted dataset
  - nnU-Net results

---

## Bucket Contents Previously Used
Known bucket objects included:
- `Dataset001_BHSD_raw.zip`
- `Dataset001_BHSD_preprocessed.zip`
- other nnUNet-related zips and prior result archives

The bucket is also used to upload downloaded result bundles from the VM.

---

## Code Ownership Boundary

### Official CSAM code
These are vendored from the official CSAM repo and should be treated as the official model core:
- `nnunet25d/csam/CSAM_modules.py`
- `nnunet25d/csam/CSAM_networks.py`

Official upstream repo:
- [aL3x-O-o-Hung/CSAM](https://github.com/aL3x-O-o-Hung/CSAM)

Important note:
- The official repo does **not** provide an nnU-Net integration.
- It provides its own model code and training entry, not nnU-Net-specific trainers.

### Custom nnU-Net integration written for this project
These are project-specific bridge files:
- `nnunet25d/csam/official_wrapper.py`
- `nnunet25d/csam/trainer_official.py`
- `nnunet25d/trainer_csam_official.py`
- `nnunet25d/trainer_csam_official_shim.py`
- early-stop integration
- patch-size override logic
- AMP compatibility fixes

So the correct description is:
- official CSAM model core
- custom nnU-Net integration layer

---

## Important Integration Decisions

### Deep supervision compatibility
Official CSAM does not follow nnU-Net default decoder expectations.
We added a compatibility layer so nnU-Net does not break when toggling deep supervision.

### AMP / half precision compatibility
Official CSAM uncertainty sampling used `LowRankMultivariateNormal`, which caused half-precision / cholesky failures under nnU-Net AMP.
We fixed this by locally disabling autocast only inside the uncertainty sampling block, while keeping the model semantics aligned with official CSAM.

### Patch size unification
Current project policy:
- 2D spatial patch size: `256 x 256`
- 25D spatial patch size: `256 x 256`
- official CSAM spatial patch size: `256 x 256`
- 3D fullres stays spatial `256 x 256` with depth unchanged by its own 3D config

Implementation detail:
- We do **not** directly assign to `configuration_manager.patch_size` because it is read-only.
- We override:
  - `self.configuration_manager.configuration["patch_size"] = [256, 256]`

---

## Key Git Commits For This Recovery
- `c6e57cb` Add official CSAM trainer and integration
- `9929170` Fix official CSAM deep supervision compatibility
- `ec7767c` Align official CSAM sampling semantics
- `c48b7ea` Disable autocast in official CSAM uncertainty sampling
- `aaf7d8e` Reduce official CSAM trainer batch size to one
- `ef99151` Unify 2D and 25D patch size to 256
- `2f5548f` Fix patch-size override for read-only config manager

Note:
- `aaf7d8e` was later superseded in direction by the patch-size unification work.
- Always inspect current branch state before assuming every intermediate change remains active exactly as originally introduced.

---

## Current Config Summary

### 2D baseline config file
- `configs/baseline_2d.yaml`

### 3D baseline config file
- `configs/baseline_3d.yaml`

### Official CSAM config file
- `configs/csam_official_3slice.yaml`

### Important note about plans
Original `nnUNetPlans.json` had:
- 2D patch size `512 x 512`
- 3D fullres patch size `[28, 256, 256]`

The project now overrides 2D-like training patch size in trainer code to enforce `256 x 256`.

---

## Training Commands

### Official CSAM training
Run on VM:

```bash
cd ~/project/repo
conda activate bhsd-nnunet
python scripts/run_experiment.py train --config csam_official_3slice
```

Common tmux flow:

```bash
tmux new -s csam_official
cd ~/project/repo
conda activate bhsd-nnunet
python scripts/run_experiment.py train --config csam_official_3slice
```

### Reinstall extension after pulling code changes
```bash
cd ~/project/repo
conda activate bhsd-nnunet
python nnunet25d/install_extension.py
```

---

## Why Official CSAM Is Slow
This was already observed and is expected relative to simpler 2D/3D baselines.

Main reasons:
- semantic + positional + slice attention
- uncertainty sampling
- custom wrapper/bridge into nnU-Net
- per-epoch validation
- local AMP protection around uncertainty block

Observed rough speed:
- about `370 seconds / epoch`

---

## Result Locations

### Small experiment-summary directory
- `~/project/repo/results/csam_official_3slice`

Contains things like:
- `stage_metrics.csv`
- `train_fold_0_metadata.json`
- `train_fold_0_resource_samples.csv`

### Full nnU-Net output directory
- `~/project/outputs/nnUNet_results/Dataset001_BHSD/nnUNetTrainer25DCSAMOfficial__nnUNetPlans__2d`

Contains things like:
- `fold_0/checkpoint_final.pth`
- `fold_0/checkpoint_best.pth`
- `fold_0/progress.png`
- `fold_0/validation/summary.json`
- `fold_0/validation/*.nii.gz`
- multiple `training_log_*.txt`

---

## Result Packaging / Download Workflow

### Small summary archive
```bash
cd ~/project/repo/results
tar -czf csam_official_3slice_results.tar.gz csam_official_3slice
gsutil cp csam_official_3slice_results.tar.gz gs://bhsd-nnunet-data-92127/
```

This archive is small because it only contains summary files.

### Full nnU-Net output archive
```bash
cd ~/project/outputs/nnUNet_results/Dataset001_BHSD
tar -czf csam_official_nnunet_fold0.tar.gz nnUNetTrainer25DCSAMOfficial__nnUNetPlans__2d
gsutil cp csam_official_nnunet_fold0.tar.gz gs://bhsd-nnunet-data-92127/
```

If this appears to hang, open a second SSH window and inspect:

```bash
ps -ef | grep -E "tar|gsutil" | grep -v grep
ls -lh ~/project/outputs/nnUNet_results/Dataset001_BHSD/csam_official_nnunet_fold0.tar.gz
gsutil ls -lh gs://bhsd-nnunet-data-92127/ | tail
```

---

## VM Rebuild Checklist
Use this when rebuilding on Google Cloud or another provider.

### Provisioning requirements
- Linux VM
- SSH access
- NVIDIA GPU
- enough disk
- bucket access
- Git access

### Setup checklist
1. Create VM
2. Confirm GPU attached
3. SSH in
4. Verify OS and Python
5. Install/verify GPU driver
6. Clone repo to `~/project/repo`
7. Download dataset zips from bucket
8. Extract into `~/project/data`
9. Install/verify Miniconda
10. Create/activate `bhsd-nnunet`
11. Set:
   - `nnUNet_raw`
   - `nnUNet_preprocessed`
   - `nnUNet_results`
12. Run:
   - `python nnunet25d/install_extension.py`
13. Verify torch CUDA works
14. Run training

### GPU verification commands
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## Update Rules For This File
Whenever the project changes, update this file instead of relying on memory.

Always update when:
- VM provider changes
- GPU type changes
- branch changes
- env vars change
- repo paths change
- official/custom integration boundary changes
- training commands change
- result locations change
- patch-size policy changes
- key new results are produced

Recommended update sections:
- Current Status
- Current Google Cloud VM Reference
- Key Git Commits
- Current Config Summary
- Result Locations
- Result Packaging / Download Workflow

---

## Next Likely Follow-Up Work
- download uploaded result archives from bucket
- compare official CSAM against 2D and 3D baselines
- decide whether to run more folds
- possibly test `NoUncertainty` variant
- possibly improve runtime if more experiments are needed
