param(
    [ValidateSet("small", "medium", "large", "all", "dry-run")]
    [string]$Mode = "small"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

switch ($Mode) {
    "small"   { $ConfigPath = ".\configs\sweep_small.yaml";  $DryRun = $false }
    "medium"  { $ConfigPath = ".\configs\sweep_medium.yaml"; $DryRun = $false }
    "large"   { $ConfigPath = ".\configs\sweep_large.yaml";  $DryRun = $false }
    "all"     { $ConfigPath = ".\configs\sweep.yaml";        $DryRun = $false }
    "dry-run" { $ConfigPath = ".\configs\sweep.yaml";        $DryRun = $true  }
}

Write-Host "Running sweep mode: $Mode"

if ($DryRun) {
    python .\src\scaling_laws\experiments\run_sweep.py `
      --config $ConfigPath `
      --dry_run
}
else {
    python .\src\scaling_laws\experiments\run_sweep.py `
      --config $ConfigPath
}

