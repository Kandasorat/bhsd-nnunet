param(
    [string]$DatasetName = "Dataset001_BHSD",
    [string]$DatasetId = "1",
    [string]$Fold = "0"
)

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
Write-Host "DatasetName=$DatasetName"
Write-Host "DatasetId=$DatasetId"
Write-Host "Fold=$Fold"
Write-Host "Note: this is the legacy 2.5D 3-slide baseline using nnUNetTrainer_25D on top of the existing 2D pipeline."

nnUNetv2_plan_and_preprocess -d $DatasetId --verify_dataset_integrity
python (Join-Path $projectRoot "scripts\install_extension.py")
nnUNetv2_train $DatasetName 2d $Fold -tr nnUNetTrainer_25D --npz
