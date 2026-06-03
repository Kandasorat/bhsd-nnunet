param(
    [switch]$IncludeData
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$projectParent = Split-Path -Parent $projectRoot
$bundleRoot = Join-Path $projectParent 'aws_upload_bundle'
$codeBundle = Join-Path $bundleRoot 'bhsd-nnunet'
$dataBundle = Join-Path $bundleRoot 'nnUNet_data'
$sourceData = Join-Path $projectRoot 'nnUNet_data'

if (Test-Path $bundleRoot) {
    Remove-Item -LiteralPath $bundleRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $bundleRoot | Out-Null
New-Item -ItemType Directory -Path $codeBundle | Out-Null

$null = robocopy $projectRoot $codeBundle /E /XD .git nnUNet_data aws_upload_bundle __pycache__ results /XF *.pyc *.pyo
$robocopyExit = $LASTEXITCODE
if ($robocopyExit -ge 8) {
    throw "robocopy failed while creating the code bundle (exit code $robocopyExit)."
}

$resultsBundleDir = Join-Path $codeBundle 'results'
New-Item -ItemType Directory -Path $resultsBundleDir -Force | Out-Null
$resultsReadme = Join-Path $projectRoot 'results\README.md'
if (Test-Path $resultsReadme) {
    Copy-Item -LiteralPath $resultsReadme -Destination (Join-Path $resultsBundleDir 'README.md') -Force
}

Get-ChildItem -LiteralPath $codeBundle -Recurse -Directory -Force |
    Where-Object { $_.Name -eq '__pycache__' } |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $codeBundle -Recurse -Include *.pyc,*.pyo -File -Force |
    Remove-Item -Force

$uploadSummary = @()
$uploadSummary += 'AWS upload bundle created successfully.'
$uploadSummary += ''
$uploadSummary += 'Always upload the code bundle:'
$uploadSummary += '- bhsd-nnunet  -> ~/projects/bhsd-nnunet'

if ($IncludeData -and (Test-Path $sourceData)) {
    New-Item -ItemType Directory -Path $dataBundle | Out-Null
    $null = robocopy $sourceData $dataBundle /E /XD __pycache__ /XF *.pyc *.pyo
    $dataRobocopyExit = $LASTEXITCODE
    if ($dataRobocopyExit -ge 8) {
        throw "robocopy failed while creating the data bundle (exit code $dataRobocopyExit)."
    }
    $uploadSummary += '- nnUNet_data  -> ~/data/nnUNet_data'
    $uploadSummary += ''
    $uploadSummary += 'This run included a data bundle because -IncludeData was used.'
} else {
    $uploadSummary += ''
    $uploadSummary += 'This run created a code-only bundle.'
    $uploadSummary += 'Use -IncludeData only if you also need to restage the dataset locally.'
}

$uploadSummary += ''
$uploadSummary += 'Then on the server run:'
$uploadSummary += ''
$uploadSummary += 'cd ~/projects/bhsd-nnunet'
$uploadSummary += 'conda env create -f environment.yml || conda env update -f environment.yml'
$uploadSummary += 'conda activate bhsd-nnunet'
$uploadSummary += 'export PROJECT_ROOT=~/projects/bhsd-nnunet'
$uploadSummary += 'export nnUNet_raw=~/data/nnUNet_data/nnUNet_raw'
$uploadSummary += 'export nnUNet_preprocessed=~/data/nnUNet_data/nnUNet_preprocessed'
$uploadSummary += 'export nnUNet_results=~/data/nnUNet_data/nnUNet_results'
$uploadSummary += 'python scripts/install_extension.py'

Set-Content -LiteralPath (Join-Path $bundleRoot 'UPLOAD_TO_AWS.md') -Value $uploadSummary -Encoding UTF8

Write-Host "Created AWS upload bundle: $bundleRoot"
Write-Host "Code bundle: $codeBundle"
if ($IncludeData -and (Test-Path $dataBundle)) {
    Write-Host "Data bundle: $dataBundle"
} else {
    Write-Host "Data bundle skipped. Re-run with -IncludeData if needed."
}
