param(
    [string]$Remote = "ly6399@gadi-dm.nci.org.au",
    [string]$LocalRoot = "D:\BHSD_server_backups\multiclass_2d_min300_patience100",
    [string]$RemoteRoot = "/scratch/ke17/bhsd-nnunet/frozen_baselines/multiclass_2d_min300_patience100"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$expected = @{
    1 = "AA88CEF38BF82C2B9880C02696C83A89099CCF653E4E763284D90953BD787AD1"
    2 = "2BB596645A3DCAD78AA4156E66F1E0AAC133E563B19A9D91718C5D6A65044458"
    3 = "78B3775392B633F09E9E892C2658A42819E040285D1055034DC113C59B326CF8"
    4 = "E59B3FD75BA8A1D397564A8CBE61988A36646E29D848635774A943B135620628"
}

foreach ($fold in 1..4) {
    $path = Join-Path $LocalRoot "fold_$fold\checkpoint_best.pth"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing checkpoint: $path"
    }
    $observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToUpperInvariant()
    if ($observed -ne $expected[$fold]) {
        throw "Fold $fold SHA-256 mismatch: $observed"
    }
}

& ssh $Remote "mkdir -p '$RemoteRoot'/fold_{1,2,3,4}"
if ($LASTEXITCODE -ne 0) { throw "Remote mkdir failed" }

foreach ($fold in 1..4) {
    $path = Join-Path $LocalRoot "fold_$fold\checkpoint_best.pth"
    & scp $path "${Remote}:$RemoteRoot/fold_$fold/checkpoint_best.pth"
    if ($LASTEXITCODE -ne 0) { throw "scp failed for fold $fold" }
}

$remoteOutput = & ssh $Remote "cd '$RemoteRoot' && sha256sum fold_{1,2,3,4}/checkpoint_best.pth"
if ($LASTEXITCODE -ne 0) { throw "Remote sha256sum failed" }
$remoteOutput | ForEach-Object { Write-Host $_ }

foreach ($fold in 1..4) {
    $matchingLine = @($remoteOutput | Where-Object { $_ -match "fold_$fold/checkpoint_best\.pth$" })
    if ($matchingLine.Count -ne 1) { throw "Missing or duplicate remote hash line for fold $fold" }
    $remoteHash = ($matchingLine -split '\s+')[0].ToUpperInvariant()
    if ($remoteHash -ne $expected[$fold]) {
        throw "Remote fold $fold SHA-256 mismatch: $remoteHash"
    }
}

Write-Host "All four Stage3 baseline checkpoints uploaded and SHA-256 verified."
