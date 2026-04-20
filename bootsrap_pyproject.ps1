# bootstrap_project.ps1
# Creates missing folders/files for the scaling-laws project without overwriting existing work.

$ErrorActionPreference = "Stop"

$Root = Get-Location

function Ensure-Directory {
    param([string]$RelativePath)

    $Path = Join-Path $Root $RelativePath
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "[created dir] $RelativePath"
    }
    else {
        Write-Host "[skip dir]    $RelativePath"
    }
}

function Ensure-File {
    param(
        [string]$RelativePath,
        [string]$Content = ""
    )

    $Path = Join-Path $Root $RelativePath
    $Parent = Split-Path $Path -Parent

    if (-not (Test-Path $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    if (-not (Test-Path $Path)) {
        Set-Content -Path $Path -Value $Content -Encoding UTF8
        Write-Host "[created file] $RelativePath"
    }
    else {
        Write-Host "[skip file]    $RelativePath"
    }
}

function Ensure-GitKeep {
    param([string]$RelativeDir)

    Ensure-Directory $RelativeDir
    $GitKeep = Join-Path $RelativeDir ".gitkeep"
    Ensure-File $GitKeep ""
}

# -------------------------------------------------------------------
# Directories
# -------------------------------------------------------------------

$Directories = @(
    "artifacts",
    "artifacts\tokenizer",
    "artifacts\data",
    "artifacts\runs",
    "artifacts\analysis",
    "results",
    "reports",
    "reports\figures",
    "reports\tables",
    "configs",
    "scripts",
    "src",
    "src\scaling_laws",
    "src\scaling_laws\tokenizer",
    "src\scaling_laws\data",
    "src\scaling_laws\models",
    "src\scaling_laws\train",
    "src\scaling_laws\experiments",
    "src\scaling_laws\analysis",
    "src\scaling_laws\utils",
    "tests"
)

foreach ($Dir in $Directories) {
    Ensure-Directory $Dir
}

# Keep empty dirs in git if needed
$GitKeepDirs = @(
    "artifacts",
    "artifacts\tokenizer",
    "artifacts\data",
    "artifacts\runs",
    "artifacts\analysis",
    "reports\figures",
    "reports\tables"
)

foreach ($Dir in $GitKeepDirs) {
    Ensure-GitKeep $Dir
}

# -------------------------------------------------------------------
# File templates
# -------------------------------------------------------------------

$Files = @{
    "configs\tokenizer.yaml" = @"
seed: 42
dataset_name: alexliap/tinystories-gr
text_column: greek_translation
val_size: 0.01
profile_vocab_sizes: [4000, 8000, 12000, 16000]
final_vocab_size: 8000
min_frequency: 2
profile_train_subset: 50000
profile_eval_subset: 5000
smallest_model_d_model: 128
smallest_model_total_params: 1000000
tied_embeddings: true
"@

    "configs\data.yaml" = @"
context_length: 256
stride: 256
add_bos_eos: true
"@

    "configs\model_1m.yaml" = @"
name: model_1m
vocab_size: 8000
context_length: 256
n_layer: 4
n_head: 4
n_embd: 128
dropout: 0.0
tie_word_embeddings: true
"@

    "configs\model_5m.yaml" = @"
name: model_5m
vocab_size: 8000
context_length: 256
n_layer: 8
n_head: 8
n_embd: 256
dropout: 0.0
tie_word_embeddings: true
"@

    "configs\model_20m.yaml" = @"
name: model_20m
vocab_size: 8000
context_length: 256
n_layer: 12
n_head: 8
n_embd: 512
dropout: 0.0
tie_word_embeddings: true
"@

    "configs\sweep.yaml" = @"
models:
  - configs/model_1m.yaml
  - configs/model_5m.yaml
  - configs/model_20m.yaml

token_budgets:
  - 2000000
  - 20000000
  - 200000000

output_csv: results/results.csv
"@

    "configs\analysis.yaml" = @"
results_csv: results/results.csv
plots_dir: reports/figures
scaling_fit_json: results/scaling_fits.json
frontier_csv: results/compute_frontier.csv
"@

    "run_all.ps1" = @'
$ErrorActionPreference = "Stop"

Write-Host "== 1. Profile tokenizer =="
powershell -ExecutionPolicy Bypass -File .\scripts\01_profile_tokenizer.ps1

Write-Host "== 2. Train final tokenizer =="
powershell -ExecutionPolicy Bypass -File .\scripts\02_train_tokenizer.ps1

Write-Host "== 3. Prepare data =="
powershell -ExecutionPolicy Bypass -File .\scripts\03_prepare_data.ps1

Write-Host "== 4. Run sweep =="
powershell -ExecutionPolicy Bypass -File .\scripts\05_run_sweep.ps1

Write-Host "== 5. Fit scaling laws =="
powershell -ExecutionPolicy Bypass -File .\scripts\06_fit_scaling_laws.ps1
'@

    "scripts\01_profile_tokenizer.ps1" = @'
python -m scaling_laws.tokenizer.profile_tokenizer --config configs/tokenizer.yaml
'@

    "scripts\02_train_tokenizer.ps1" = @'
python -m scaling_laws.tokenizer.train_tokenizer --config configs/tokenizer.yaml
'@

    "scripts\03_prepare_data.ps1" = @'
python -m scaling_laws.data.prepare_data --tokenizer-dir artifacts/tokenizer/final_vocab_8000 --config configs/data.yaml
'@

    "scripts\04_run_single_experiment.ps1" = @'
python -m scaling_laws.experiments.run_experiment --model-config configs/model_1m.yaml --token-budget 2000000
'@

    "scripts\05_run_sweep.ps1" = @'
python -m scaling_laws.experiments.run_sweep --config configs/sweep.yaml
'@

    "scripts\06_fit_scaling_laws.ps1" = @'
python -m scaling_laws.analysis.fit_scaling_laws --config configs/analysis.yaml
python -m scaling_laws.analysis.compute_frontier --config configs/analysis.yaml
python -m scaling_laws.analysis.plot_results --config configs/analysis.yaml
'@

    "src\scaling_laws\__init__.py" = @'
__all__ = []
__version__ = "0.1.0"
'@

    "src\scaling_laws\tokenizer\__init__.py" = ""
    "src\scaling_laws\data\__init__.py" = ""
    "src\scaling_laws\models\__init__.py" = ""
    "src\scaling_laws\train\__init__.py" = ""
    "src\scaling_laws\experiments\__init__.py" = ""
    "src\scaling_laws\analysis\__init__.py" = ""
    "src\scaling_laws\utils\__init__.py" = ""

    "src\scaling_laws\tokenizer\profile_tokenizer.py" = @'
def main():
    raise NotImplementedError("TODO: implement tokenizer profiling")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\tokenizer\train_tokenizer.py" = @'
def main():
    raise NotImplementedError("TODO: implement final tokenizer training")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\data\prepare_data.py" = @'
def main():
    raise NotImplementedError("TODO: implement dataset tokenization and chunking")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\data\dataset.py" = @'
class LMDataset:
    pass
'@

    "src\scaling_laws\data\collator.py" = @'
class CausalLMCollator:
    pass
'@

    "src\scaling_laws\models\config.py" = @'
class ModelConfig:
    pass
'@

    "src\scaling_laws\models\gpt.py" = @'
class GPT:
    pass
'@

    "src\scaling_laws\models\param_count.py" = @'
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'@

    "src\scaling_laws\train\train_model.py" = @'
def main():
    raise NotImplementedError("TODO: implement training entrypoint")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\train\trainer.py" = @'
class Trainer:
    pass
'@

    "src\scaling_laws\train\eval.py" = @'
def evaluate(*args, **kwargs):
    raise NotImplementedError("TODO: implement validation loop")
'@

    "src\scaling_laws\experiments\run_experiment.py" = @'
def main():
    raise NotImplementedError("TODO: implement single experiment run")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\experiments\run_sweep.py" = @'
def main():
    raise NotImplementedError("TODO: implement experiment sweep")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\experiments\aggregate_results.py" = @'
def main():
    raise NotImplementedError("TODO: implement result aggregation")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\analysis\fit_scaling_laws.py" = @'
def main():
    raise NotImplementedError("TODO: implement power-law fitting")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\analysis\compute_frontier.py" = @'
def main():
    raise NotImplementedError("TODO: implement compute-optimal frontier")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\analysis\plot_results.py" = @'
def main():
    raise NotImplementedError("TODO: implement plotting")

if __name__ == "__main__":
    main()
'@

    "src\scaling_laws\utils\io.py" = @'
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
'@

    "src\scaling_laws\utils\logging.py" = @'
import logging

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
'@

    "src\scaling_laws\utils\seeds.py" = @'
import random

def set_seed(seed: int):
    random.seed(seed)
'@

    "src\scaling_laws\utils\paths.py" = @'
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
'@

    "reports\report.md" = @'
# Scaling Laws for a Small Greek Language Model

## TODO
- Dataset and preprocessing
- Tokenizer design
- Model architecture
- Experiment grid
- Training curves
- Scaling law fits
- Compute-optimal frontier
- Limitations
'@

    "results\results.csv" = @'
run_id,n_params,n_tokens,flops,val_loss
'@

    "results\scaling_fits.json" = @'
{}
'@

    "results\compute_frontier.csv" = @'
flops_budget,n_params,n_tokens,predicted_val_loss
'@

    "tests\test_tokenizer.py" = @'
def test_placeholder():
    assert True
'@

    "tests\test_param_count.py" = @'
def test_placeholder():
    assert True
'@

    "tests\test_results_schema.py" = @'
def test_placeholder():
    assert True
'@
}

foreach ($RelativePath in $Files.Keys) {
    Ensure-File -RelativePath $RelativePath -Content $Files[$RelativePath]
}

Write-Host ""
Write-Host "Project scaffold check complete."
Write-Host "Nothing existing was overwritten."