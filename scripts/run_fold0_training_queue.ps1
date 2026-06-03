$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "results\run_logs"
$stdoutLog = Join-Path $logDir "fold0_train_queue.stdout.log"
$stderrLog = Join-Path $logDir "fold0_train_queue.stderr.log"
$pythonExe = "C:\Users\92127\anaconda3\envs\bhsd\python.exe"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_data\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_data\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_data\nnUNet_results"
$env:nnUNet_n_proc_DA = "0"

Set-Location $projectRoot

function Write-QueueLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $stdoutLog -Value "[$timestamp] $Message"
}

function Invoke-TrainingConfig {
    param([string]$ConfigName)
    Write-QueueLog "Starting training for $ConfigName"
    & $pythonExe "scripts\run_experiment.py" train --config $ConfigName 2>> $stderrLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $ConfigName with exit code $LASTEXITCODE"
    }
    Write-QueueLog "Finished training for $ConfigName"
}

Write-QueueLog "Fold-0 training queue started"
Invoke-TrainingConfig "baseline_2d"
Invoke-TrainingConfig "baseline_3d"
Invoke-TrainingConfig "naive_25d_3slice"
Invoke-TrainingConfig "naive_25d_5slice"
Write-QueueLog "Fold-0 training queue finished"
