param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigName,

    [ValidateSet("preprocess", "train", "infer", "evaluate", "run_all")]
    [string]$Stage = "train",

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "C:\Users\92127\anaconda3\envs\bhsd\python.exe"
$logDir = Join-Path $projectRoot "results\run_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPrefix = "${ConfigName}_${Stage}"
if ($Resume) {
    $logPrefix = "${logPrefix}_resume"
}
$stdoutLog = Join-Path $logDir "${logPrefix}_${timestamp}.stdout.log"
$stderrLog = Join-Path $logDir "${logPrefix}_${timestamp}.stderr.log"

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_data\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_data\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_data\nnUNet_results"
$env:nnUNet_n_proc_DA = "0"

$command = @(
    $pythonExe,
    "scripts\run_experiment.py",
    $Stage,
    "--config",
    $ConfigName
)
if ($Resume) {
    $command += "--resume"
}

Write-Host "Project root: $projectRoot"
Write-Host "stdout log:  $stdoutLog"
Write-Host "stderr log:  $stderrLog"
Write-Host "Command:     $($command -join ' ')"

Set-Location $projectRoot

$pythonArgs = @("scripts\run_experiment.py", $Stage, "--config", $ConfigName)
if ($Resume) {
    $pythonArgs += "--resume"
}

& $pythonExe @pythonArgs 1>> $stdoutLog 2>> $stderrLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "Experiment finished with exit code $exitCode"
Write-Host "stdout log: $stdoutLog"
Write-Host "stderr log: $stderrLog"
exit $exitCode
