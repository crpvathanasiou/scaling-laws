$ErrorActionPreference = "Stop"

# Move to repository root, no matter where the script is called from.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Running tokenizer profiling + final tokenizer training..."

python .\src\scaling_laws\tokenizer\train_tokenizer.py `
  --mode both `
  --dataset_name "alexliap/tinystories-gr" `
  --text_column "greek_translation" `
  --val_size 0.01 `
  --seed 42 `
  --profile_vocab_sizes 4000 8000 12000 16000 `
  --final_vocab_size 8000 `
  --min_frequency 2 `
  --profile_train_subset 50000 `
  --profile_eval_subset 5000 `
  --smallest_model_d_model 128 `
  --smallest_model_total_params 1000000 `
  --tied_embeddings `
  --token_histogram_sample_size 5000 `
  --artifacts_dir "artifacts/tokenizer"

if ($LASTEXITCODE -ne 0) {
  throw "Tokenizer run failed."
}

Write-Host "Tokenizer run completed."
