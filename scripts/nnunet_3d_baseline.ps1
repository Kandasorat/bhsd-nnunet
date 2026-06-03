$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_data\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_data\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_data\nnUNet_results"

# Conservative default for Windows. Remove or override on servers if desired.
$env:nnUNet_n_proc_DA = "0"

Write-Host "nnUNet_raw=$env:nnUNet_raw"
Write-Host "nnUNet_preprocessed=$env:nnUNet_preprocessed"
Write-Host "nnUNet_results=$env:nnUNet_results"
Write-Host "nnUNet_n_proc_DA=$env:nnUNet_n_proc_DA"

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train Dataset001_BHSD 3d_fullres 0 --npz
