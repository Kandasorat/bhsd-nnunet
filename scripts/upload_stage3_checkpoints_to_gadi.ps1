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

$relativeFiles = 1..4 | ForEach-Object { "fold_$_/checkpoint_best.pth" }
$remoteFiles = $relativeFiles -join " "
$remoteCommand = "set -e; mkdir -p '$RemoteRoot'; tar -xf - -C '$RemoteRoot'; cd '$RemoteRoot'; sha256sum $remoteFiles"

$tarInfo = [System.Diagnostics.ProcessStartInfo]::new()
$tarInfo.FileName = (Get-Command tar.exe -ErrorAction Stop).Source
$tarInfo.Arguments = "-cf - -C `"$LocalRoot`" $remoteFiles"
$tarInfo.UseShellExecute = $false
$tarInfo.CreateNoWindow = $true
$tarInfo.RedirectStandardOutput = $true
$tarInfo.RedirectStandardError = $true

$sshInfo = [System.Diagnostics.ProcessStartInfo]::new()
$sshInfo.FileName = (Get-Command ssh.exe -ErrorAction Stop).Source
$sshInfo.Arguments = "-T $Remote `"$remoteCommand`""
$sshInfo.UseShellExecute = $false
$sshInfo.CreateNoWindow = $false
$sshInfo.RedirectStandardInput = $true
$sshInfo.RedirectStandardOutput = $true
$sshInfo.RedirectStandardError = $false

Write-Host "Opening one SSH connection for all four checkpoints. Enter the NCI password once."
Write-Host "Streaming approximately 1.4 GB; this mode has no per-file progress display."
$sshProcess = [System.Diagnostics.Process]::Start($sshInfo)
$tarProcess = [System.Diagnostics.Process]::Start($tarInfo)
try {
    $tarProcess.StandardOutput.BaseStream.CopyTo($sshProcess.StandardInput.BaseStream)
    $sshProcess.StandardInput.Close()
    $tarProcess.WaitForExit()
    $tarError = $tarProcess.StandardError.ReadToEnd()
    if ($tarProcess.ExitCode -ne 0) {
        throw "Local tar stream failed with exit code $($tarProcess.ExitCode): $tarError"
    }
    $remoteText = $sshProcess.StandardOutput.ReadToEnd()
    $sshProcess.WaitForExit()
    if ($sshProcess.ExitCode -ne 0) {
        throw "Single-session upload or remote SHA-256 verification failed with exit code $($sshProcess.ExitCode)"
    }
}
finally {
    if (-not $tarProcess.HasExited) { $tarProcess.Kill() }
    if (-not $sshProcess.HasExited) { $sshProcess.Kill() }
    $tarProcess.Dispose()
    $sshProcess.Dispose()
}

$remoteOutput = @($remoteText -split "`r?`n" | Where-Object { $_ })
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
