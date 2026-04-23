$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running scaling law analysis..."

python .\src\scaling_laws\analysis\fit_scaling_laws.py `
  --config .\configs\analysis.yaml

Write-Host "Scaling law analysis completed."

