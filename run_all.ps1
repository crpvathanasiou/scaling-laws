$ErrorActionPreference = "Stop"

Write-Host "== 1. Train tokenizer =="
powershell -ExecutionPolicy Bypass -File .\scripts\01_tokenizer.ps1

Write-Host "== 2. Prepare data =="
powershell -ExecutionPolicy Bypass -File .\scripts\03_prepare_data.ps1

Write-Host "== 3. Run sweep =="
powershell -ExecutionPolicy Bypass -File .\scripts\05_run_sweep.ps1

Write-Host "== 4. Fit scaling laws =="
powershell -ExecutionPolicy Bypass -File .\scripts\06_fit_scaling_laws.ps1
