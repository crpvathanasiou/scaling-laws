# RUNBOOK

Βάλε αυτό ως:

`RUNBOOK.md`

````markdown
# RUNBOOK

Operational guide for running the Greek scaling-laws pipeline end-to-end.

---

## 1. Environment setup

Create and activate the project environment, then install dependencies.

### Windows PowerShell
```powershell
poetry install
poetry shell
````

### Verify GPU

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Tokenizer

Run tokenizer profiling and final tokenizer training.

### PowerShell

```powershell
.\scripts\01_tokenizer.ps1
```

### Expected outputs

* `artifacts/tokenizer/tokenizer_profile.csv`
* `artifacts/tokenizer/tokenizer_profile.json`
* `artifacts/tokenizer/plots/`
* `artifacts/tokenizer/final_vocab_8000/tokenizer.json`
* `artifacts/tokenizer/final_vocab_8000/tokenizer_config.json`
* `artifacts/tokenizer/final_vocab_8000/special_tokens_map.json`

---

## 3. Data preparation

Build train/validation token chunks from the frozen tokenizer.

### PowerShell

```powershell
.\scripts\02_prepare_data.ps1
```

### Expected outputs

* `artifacts/data/train_chunks.npy`
* `artifacts/data/val_chunks.npy`
* `artifacts/data/data_prep_metadata.json`
* `artifacts/data/sample_batch_preview.json`

---

## 4. Single debug training run

Optional sanity check before the full sweep.

### PowerShell

```powershell
.\scripts\03_train_model.ps1
```

### Expected outputs

* `artifacts/runs/run_small_debug/`
* `results/results.csv`

---

## 5. Sweep runs

Run the experiment grid in batches.

### Dry run

```powershell
.\scripts\05_run_sweep.ps1 -Mode dry-run
```

### Small model sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode small
```

### Medium model sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode medium
```

### Large model sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode large
```

### Full sweep

```powershell
.\scripts\05_run_sweep.ps1 -Mode all
```

### Expected outputs

Per run:

* `artifacts/runs/<run_name>/run_spec.json`
* `artifacts/runs/<run_name>/train_loss_curve.csv`
* `artifacts/runs/<run_name>/eval_metrics.json`
* `artifacts/runs/<run_name>/run_summary.json`
* `artifacts/runs/<run_name>/final_model.pt`

Aggregate:

* `results/results.csv`

---

## 6. Scaling-law analysis

Fit simple empirical power laws and generate plots.

### PowerShell

```powershell
.\scripts\06_fit_scaling_laws.ps1
```

### Expected outputs

* `artifacts/analysis/cleaned_results.csv`
* `artifacts/analysis/scaling_fits.json`
* `artifacts/analysis/analysis_summary.json`
* `artifacts/analysis/loss_vs_params.png`
* `artifacts/analysis/loss_vs_tokens.png`
* `artifacts/analysis/loss_vs_flops.png`

---

## 7. Compute frontier

Compute the empirical compute-optimal frontier.

### PowerShell

```powershell
.\scripts\07_compute_frontier.ps1
```

### Expected outputs

* `artifacts/analysis/compute_frontier.csv`
* `artifacts/analysis/compute_frontier.json`
* `artifacts/analysis/compute_frontier.png`

---

## 8. One-command pipeline

Run the main pipeline through the root script.

### Small sweep only, skip tokenizer and data prep

```powershell
.\run_all.ps1 -Mode small -SkipTokenizer -SkipDataPrep
```

### Full pipeline

```powershell
.\run_all.ps1 -Mode all
```

---

## 9. Recommended execution order

## 9. Recommended execution order

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
```

* `03_train_model.ps1` and `04_run_single_experiment.ps1` are optional sanity-check steps before running the full sweep.

---

## 10. Notes

* The tokenizer is frozen after selection and reused across all runs.
* The selected vocabulary size is `8000`.
* The dataset used is `alexliap/tinystories-gr`, column `greek_translation`.
* The experiment grid consists of 3 model sizes × 3 token budgets.
* Validation loss is the main metric used in scaling-law analysis.

```


