````md
# Scaling Laws for a Small Greek Language Model

This repository contains a complete experimental pipeline for deriving empirical scaling laws for a small Greek-language causal language model trained from scratch.

The pipeline covers:

- training a Greek-only BPE tokenizer from scratch
- preparing tokenized training/validation data
- training multiple decoder-only language models from random initialization
- running a structured sweep over model size and token budget
- collecting results into a unified results table
- fitting simple power-law trends
- computing an empirical compute-optimal frontier
- generating per-run and combined training-curve figures
- producing figures and report assets for the final assignment

---

## Project goal

The project is designed around the take-home assignment requirement to:

- use the `alexliap/tinystories-gr` dataset
- train a tokenizer only on the `greek_translation` column
- train several small Greek language models from scratch
- measure validation loss as a function of:
  - model size
  - number of training tokens
  - estimated FLOPs
- fit empirical scaling laws
- identify the compute-optimal frontier


---

## Repository structure

```text
configs/         # YAML configs for tokenizer, data prep, models, training, sweep, analysis
scripts/         # PowerShell and bash entrypoints for each pipeline stage
src/             # Python source code
results/         # aggregated experiment results
reports/         # report source, report assets, figures, tables, PDF
artifacts/       # generated runtime/local outputs (tokenizer, data, runs, analysis)
RUNBOOK.md       # operational step-by-step execution guide
README.md        # project overview
````

---

## Dataset

The dataset used is:

* `alexliap/tinystories-gr`

Only the following field is used:

* `greek_translation`

No English text is used anywhere in tokenizer training or model training.

---

## Tokenizer

The project uses a single Greek-only BPE tokenizer trained from scratch on the training split of the Greek corpus.

### Tokenizer design

* tokenizer type: BPE
* normalization: `NFKC`
* pre-tokenization: whitespace
* special tokens:

  * `[PAD]`
  * `[UNK]`
  * `[BOS]`
  * `[EOS]`

The tokenizer is profiled first across candidate vocabulary sizes, then trained once with the selected final vocabulary size and reused unchanged across all model runs.

### Selected vocabulary size

* final vocabulary size: `8000`

### Run tokenizer

#### Windows PowerShell

```powershell
.\scripts\01_tokenizer.ps1
```

#### Bash

```bash
bash scripts/01_tokenizer.sh
```

### Tokenizer outputs

Generated under:

* `artifacts/tokenizer/`

Main outputs include:

* `tokenizer_profile.csv`
* `tokenizer_profile.json`
* `plots/fertility_vs_vocab_size.png`
* `plots/embedding_share_vs_vocab_size.png`
* `plots/token_frequency_histogram.png`
* `final_vocab_8000/tokenizer.json`
* `final_vocab_8000/tokenizer_config.json`
* `final_vocab_8000/special_tokens_map.json`
* `final_vocab_8000/tokenizer_preview.json`
* `final_vocab_8000/vocab.txt`

---

## Data preparation

After tokenizer training, the dataset is encoded with the frozen tokenizer and packed into fixed-length token chunks.

### Data prep behavior

* load `greek_translation`
* clean empty / invalid rows
* create train / validation split
* encode with the frozen tokenizer
* pack into fixed-length chunks
* save train and validation arrays

### Run data preparation

#### Windows PowerShell

```powershell
.\scripts\02_prepare_data.ps1
```

#### Bash

```bash
bash scripts/02_prepare_data.sh
```

### Data outputs

Generated under:

* `artifacts/data/`

Main outputs include:

* `train_chunks.npy`
* `val_chunks.npy`
* `data_prep_metadata.json`
* `sample_batch_preview.json`

---

## Model training

The models are decoder-only causal language models trained from random initialization.

### Model family

The same model family is used across all experiments, while scaling:

* number of layers
* number of heads
* embedding dimension

### Available model configs

* `configs/model_small.yaml`
* `configs/model_medium.yaml`
* `configs/model_large.yaml`

### Base training config

* `configs/train_base.yaml`

### Run a single debug training run

#### Windows PowerShell

```powershell
.\scripts\03_train_model.ps1
```

#### Bash

```bash
bash scripts/03_train_model.sh
```

This is mainly used as a sanity check before the full experiment sweep.

---

## Single experiment runner

A single standardized experiment can also be launched through:

#### Windows PowerShell

```powershell
.\scripts\04_run_single_experiment.ps1
```

#### Bash

```bash
bash scripts/04_run_single_experiment.sh
```

This stage is useful to validate:

* token-budget-to-max-steps conversion
* output directory layout
* result logging
* per-run artifacts

---

## Experiment sweep

The structured sweep is run over:

* multiple model sizes
* multiple token budgets

Current sweep batching is split by model family.

### Sweep modes

* `small`
* `medium`
* `large`

### Run sweep

#### Dry run

```powershell
.\scripts\05_run_sweep.ps1 -Mode dry-run
```

#### Small sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode small
```

#### Medium sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode medium
```

#### Large sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode large
```

#### Bash

```bash
bash scripts/05_run_sweep.sh small
bash scripts/05_run_sweep.sh medium
bash scripts/05_run_sweep.sh large
```

### Sweep outputs

Per run, generated under:

* `artifacts/runs/<run_name>/`

Main per-run outputs:

* `run_spec.json`
* `train_loss_curve.csv`
* `eval_metrics.json`
* `run_summary.json`
* `final_model.pt`

Aggregate output:

* `results/results.csv`

---

## Scaling-law analysis

After all runs are complete, the pipeline fits simple empirical power laws and generates plots.

### Run scaling-law fitting

#### Windows PowerShell

```powershell
.\scripts\06_fit_scaling_laws.ps1
```

#### Bash

```bash
bash scripts/06_fit_scaling_laws.sh
```

### Analysis outputs

Generated under:

* `artifacts/analysis/`

Main outputs:

* `cleaned_results.csv`
* `scaling_fits.json`
* `analysis_summary.json`
* `loss_vs_params.png`
* `loss_vs_tokens.png`
* `loss_vs_flops.png`

---

## Compute frontier

The project also computes an empirical compute-optimal frontier from the experimental runs.

### Run compute frontier

#### Windows PowerShell

```powershell
.\scripts\07_compute_frontier.ps1
```

#### Bash

```bash
bash scripts/07_compute_frontier.sh
```

### Frontier outputs

Generated under:

* `artifacts/analysis/`

Main outputs:

* `compute_frontier.csv`
* `compute_frontier.json`
* `compute_frontier.png`

---

## Training curves

Training-curve figures are generated from the per-run `train_loss_curve.csv` files after the sweep is complete.

### Run training-curve plotting

#### Windows PowerShell

```powershell
.\scripts\08_plot_training_curves.ps1
```

#### Bash

```bash
bash scripts/08_plot_training_curves.sh
```

### Training-curve outputs

Generated under:

* `reports/figures/training_curves/`
* `reports/figures/all_training_curves.png`
* `reports/tables/training_curves_summary.csv`

---

## Recommended execution order

```powershell
.\scripts\01_tokenizer.ps1
.\scripts\02_prepare_data.ps1
.\scripts\03_train_model.ps1
.\scripts\04_run_single_experiment.ps1

.\scripts\05_run_sweep.ps1 -Mode small
.\scripts\05_run_sweep.ps1 -Mode medium
.\scripts\05_run_sweep.ps1 -Mode large

.\scripts\06_fit_scaling_laws.ps1
.\scripts\07_compute_frontier.ps1
.\scripts\08_plot_training_curves.ps1
```

Notes:

* `03_train_model.ps1` and `04_run_single_experiment.ps1` are optional sanity-check stages before the full sweep.
* The full operational guide is documented in `RUNBOOK.md`.

---

## One-command pipeline

A root-level pipeline script is also available.

### Run full pipeline

#### Windows PowerShell

```powershell
.\run_all.ps1 -Mode all
```

#### Bash

```bash
bash ./run_all.sh all
```

### Run only one sweep stage while skipping tokenizer and data prep

#### Windows PowerShell

```powershell
.\run_all.ps1 -Mode small -SkipTokenizer -SkipDataPrep
```

The root-level pipeline script runs the main pipeline and the post-sweep analysis stages currently wired into `run_all.ps1` / `run_all.sh`.

---

## Main configs

* `configs/tokenizer.yaml`
* `configs/data.yaml`
* `configs/model_small.yaml`
* `configs/model_medium.yaml`
* `configs/model_large.yaml`
* `configs/train_base.yaml`
* `configs/sweep.yaml`
* `configs/sweep_small.yaml`
* `configs/sweep_medium.yaml`
* `configs/sweep_large.yaml`
* `configs/analysis.yaml`
* `configs/training_curves.yaml`

---

## Main deliverables produced by the pipeline

Core deliverables:

* `results.csv` (root-level final copy for submission)
* `reports/report.md` (editable report source)
* `reports/report.pdf`
* report-ready figures under `reports/figures/`
* report-ready tables under `reports/tables/`

Runtime / local reproducibility artifacts are generated under:

* `artifacts/tokenizer/`
* `artifacts/data/`
* `artifacts/runs/`
* `artifacts/analysis/`

---

## Notes

* The tokenizer is trained only on the training split.
* The same frozen tokenizer is reused across all experiments.
* All models are trained from random initialization.
* Validation loss is the main metric used for scaling-law analysis.
* The pipeline is scriptable end-to-end and does not depend on notebooks.
* `artifacts/runs/` contains per-run local outputs such as checkpoints and run logs; these are useful for reproducibility but are not intended as core submission deliverables.

---

```
```
