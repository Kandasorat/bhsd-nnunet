param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigName,

    [ValidateSet("preprocess", "train", "infer", "evaluate", "run_all")]
    [string]$Stage = "train",

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_experiment_foreground.ps1"

$argumentList = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-ConfigName", $ConfigName,
    "-Stage", $Stage
)
if ($Resume) {
    $argumentList += "-Resume"
}

$proc = Start-Process -FilePath "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList $argumentList `
    -WorkingDirectory $projectRoot `
    -PassThru

Write-Host "Started experiment window."
Write-Host "PID: $($proc.Id)"
Write-Host "Config: $ConfigName"
Write-Host "Stage: $Stage"
Write-Host "Resume: $Resume"
