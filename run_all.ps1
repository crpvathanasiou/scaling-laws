param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Mode = "small",
    [switch]$SkipTokenizer,
    [switch]$SkipDataPrep,
    [switch]$SkipSweep,
    [switch]$SkipScalingFits,
    [switch]$SkipFrontier,
    [switch]$SkipTrainingCurves
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
    Write-Host "`n[1/6] START Tokenizer"
    .\scripts\01_tokenizer.ps1
    Write-Host "[1/6] DONE Tokenizer"
}
else {
    Write-Host "`n[1/6] SKIP Tokenizer"
}

if (-not $SkipDataPrep) {
    Write-Host "`n[2/6] START Data preparation"
    .\scripts\02_prepare_data.ps1
    Write-Host "[2/6] DONE Data preparation"
}
else {
    Write-Host "`n[2/6] SKIP Data preparation"
}

if (-not $SkipSweep) {
    Write-Host "`n[3/6] START Sweep ($Mode)"
    .\scripts\05_run_sweep.ps1 -Mode $Mode
    Write-Host "[3/6] DONE Sweep ($Mode)"
}
else {
    Write-Host "`n[3/6] SKIP Sweep"
}

if (-not $SkipScalingFits) {
    Write-Host "`n[4/6] START Scaling-law fitting"
    .\scripts\06_fit_scaling_laws.ps1
    Write-Host "[4/6] DONE Scaling-law fitting"
}
else {
    Write-Host "`n[4/6] SKIP Scaling-law fitting"
}

if (-not $SkipFrontier) {
    Write-Host "`n[5/6] START Compute frontier"
    .\scripts\07_compute_frontier.ps1
    Write-Host "[5/6] DONE Compute frontier"
}
else {
    Write-Host "`n[5/6] SKIP Compute frontier"
}

if (-not $SkipTrainingCurves) {
    Write-Host "`n[6/6] START Training-curve plotting"
    .\scripts\08_plot_training_curves.ps1
    Write-Host "[6/6] DONE Training-curve plotting"
}
else {
    Write-Host "`n[6/6] SKIP Training-curve plotting"
}

Write-Host "`n========================================"
Write-Host "Pipeline completed successfully."
Write-Host "========================================"
