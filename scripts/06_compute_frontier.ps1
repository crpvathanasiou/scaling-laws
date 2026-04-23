$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running compute frontier analysis..."

python .\src\scaling_laws\analysis\compute_frontier.py `
  --config .\configs\analysis.yaml

Write-Host "Compute frontier analysis completed."
