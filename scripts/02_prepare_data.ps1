$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running data preparation..."

python .\src\scaling_laws\data\prepare_data.py `
  --config .\configs\data.yaml

Write-Host "Data preparation completed."
