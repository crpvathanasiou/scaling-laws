param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Mode = "small",
    [switch]$SkipTokenizer,
    [switch]$SkipDataPrep
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "========================================"
Write-Host "Scaling Laws Pipeline"
Write-Host "Mode: $Mode"
Write-Host "SkipTokenizer: $SkipTokenizer"
Write-Host "SkipDataPrep: $SkipDataPrep"
Write-Host "Working directory: $RepoRoot"
Write-Host "========================================"

if (-not $SkipTokenizer) {
    Write-Host "`n[1/3] START Tokenizer"
    .\scripts\01_tokenizer.ps1
    Write-Host "[1/3] DONE Tokenizer"
}
else {
    Write-Host "`n[1/3] SKIP Tokenizer"
}

if (-not $SkipDataPrep) {
    Write-Host "`n[2/3] START Data preparation"
    .\scripts\02_prepare_data.ps1
    Write-Host "[2/3] DONE Data preparation"
}
else {
    Write-Host "`n[2/3] SKIP Data preparation"
}

Write-Host "`n[3/3] START Sweep ($Mode)"
.\scripts\04_run_sweep.ps1 -Mode $Mode
Write-Host "[3/3] DONE Sweep ($Mode)"

Write-Host "`n========================================"
Write-Host "Pipeline completed successfully."
Write-Host "========================================"
