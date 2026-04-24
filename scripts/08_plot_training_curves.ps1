$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Generating training curve figures..."

python .\src\scaling_laws\analysis\plot_training_curves.py `
  --config .\configs\training_curves.yaml

Write-Host "Training curve figures completed."
