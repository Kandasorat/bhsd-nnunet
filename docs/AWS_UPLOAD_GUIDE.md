# AWS Upload and First-Run Guide

This guide is for preparing the local project on Windows and then uploading it to a Linux GPU server.

## 1. Recommended layout on AWS

Place code and data in separate directories:

```text
~/projects/bhsd-nnunet/
~/data/nnUNet_data/
```

The final server-side data tree should look like:

```text
~/data/nnUNet_data/nnUNet_raw
~/data/nnUNet_data/nnUNet_preprocessed
~/data/nnUNet_data/nnUNet_results
```

## 2. Create an upload bundle locally

If the dataset is already on AWS, create a code-only bundle:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_aws_bundle.ps1
```

This creates:

```text
aws_upload_bundle/
  bhsd-nnunet/
  UPLOAD_TO_AWS.md
```

If you ever also need to restage the dataset locally, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_aws_bundle.ps1 -IncludeData
```

That version additionally creates:

```text
aws_upload_bundle/nnUNet_data/
```

## 3. Upload to AWS

For your current workflow, upload only:

- `aws_upload_bundle/bhsd-nnunet` -> `~/projects/bhsd-nnunet`

Do not re-upload the dataset if it is already present under `~/data/nnUNet_data`.

## 4. Server startup steps

After login:

```bash
cd ~/projects/bhsd-nnunet
conda env create -f environment.yml || conda env update -f environment.yml
conda activate bhsd-nnunet
export PROJECT_ROOT=~/projects/bhsd-nnunet
export nnUNet_raw=~/data/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=~/data/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=~/data/nnUNet_data/nnUNet_results
python scripts/install_extension.py
```

## 5. First recommended run order

Run in this order:

```bash
bash scripts/preprocess.sh baseline_2d
bash scripts/train.sh baseline_2d
bash scripts/train.sh baseline_3d
bash scripts/train.sh baseline_25d_3slide
```

After training, continue with:

```bash
bash scripts/infer.sh baseline_2d
bash scripts/evaluate.sh baseline_2d
bash scripts/collect_results.sh
bash scripts/analyze.sh /path/to/all_case_metrics.csv baseline_2d
```

## 6. Notes

- Keep `nnUNet_data` outside the code repository on AWS.
- Do not edit generated files in `results/` manually.
- `aws_upload_bundle/` is a local staging directory and should not be committed.
