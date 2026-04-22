$ErrorActionPreference = "Stop"

# Move to repository root, no matter where the script is called from.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running tokenizer profiling + final tokenizer training..."

python .\src\scaling_laws\tokenizer\train_tokenizer.py `
  --config .\configs\tokenizer.yaml

if ($LASTEXITCODE -ne 0) {
  throw "Tokenizer run failed."
}

Write-Host "Tokenizer run completed."
