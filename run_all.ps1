param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Mode = "small",
    [switch]$SkipTokenizer,
    [switch]$SkipDataPrep,
    [switch]$SkipSweep,
    [switch]$SkipScalingFits,
    [switch]$SkipFrontier
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "========================================"
Write-Host "Scaling Laws Pipeline"
Write-Host "Mode: $Mode"
Write-Host "SkipTokenizer: $SkipTokenizer"
Write-Host "SkipDataPrep: $SkipDataPrep"
Write-Host "SkipSweep: $SkipSweep"
Write-Host "SkipScalingFits: $SkipScalingFits"
Write-Host "SkipFrontier: $SkipFrontier"
Write-Host "Working directory: $RepoRoot"
Write-Host "========================================"

if (-not $SkipTokenizer) {
    Write-Host "`n[1/5] START Tokenizer"
    .\scripts\01_tokenizer.ps1
    Write-Host "[1/5] DONE Tokenizer"
}
else {
    Write-Host "`n[1/5] SKIP Tokenizer"
}

if (-not $SkipDataPrep) {
    Write-Host "`n[2/5] START Data preparation"
    .\scripts\02_prepare_data.ps1
    Write-Host "[2/5] DONE Data preparation"
}
else {
    Write-Host "`n[2/5] SKIP Data preparation"
}

if (-not $SkipSweep) {
    Write-Host "`n[3/5] START Sweep ($Mode)"
    .\scripts\05_run_sweep.ps1 -Mode $Mode
    Write-Host "[3/5] DONE Sweep ($Mode)"
}
else {
    Write-Host "`n[3/5] SKIP Sweep"
}

if (-not $SkipScalingFits) {
    Write-Host "`n[4/5] START Scaling-law fitting"
    .\scripts\06_fit_scaling_laws.ps1
    Write-Host "[4/5] DONE Scaling-law fitting"
}
else {
    Write-Host "`n[4/5] SKIP Scaling-law fitting"
}

if (-not $SkipFrontier) {
    Write-Host "`n[5/5] START Compute frontier"
    .\scripts\07_compute_frontier.ps1
    Write-Host "[5/5] DONE Compute frontier"
}
else {
    Write-Host "`n[5/5] SKIP Compute frontier"
}

Write-Host "`n========================================"
Write-Host "Pipeline completed successfully."
Write-Host "========================================"
