$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running model training..."

python .\src\scaling_laws\train\train_model.py `
  --config .\configs\train_base.yaml

Write-Host "Model training completed."
