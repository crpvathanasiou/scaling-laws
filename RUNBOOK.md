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

### Notes

* A CUDA-capable GPU is recommended for model training and sweeps.
* Tokenizer training, data preparation, and analysis stages can run on CPU.
* If Hugging Face downloads are slow, configure `HF_TOKEN` locally.

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

## 5. Single standardized experiment

Optional standardized experiment run before the full sweep.

### PowerShell

```powershell
.\scripts\04_run_single_experiment.ps1
```

### Expected outputs

* `artifacts/runs/<run_name>/run_spec.json`
* `artifacts/runs/<run_name>/train_loss_curve.csv`
* `artifacts/runs/<run_name>/eval_metrics.json`
* `artifacts/runs/<run_name>/run_summary.json`
* `artifacts/runs/<run_name>/final_model.pt`
* `results/results.csv`

---

## 6. Sweep runs

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

### Practical note

The sweep can take a substantial amount of time. A common workflow is:

* run `small`
* then `medium`
* then `large`

instead of rerunning the entire grid from scratch.

---

## 7. Scaling-law analysis

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

## 8. Compute frontier

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

## 9. Training curves

Generate per-run and combined training-curve figures from the stored run logs.

### PowerShell

```powershell
.\scripts\08_plot_training_curves.ps1
```

### Expected outputs

* `reports/figures/training_curves/`
* `reports/figures/all_training_curves.png`
* `reports/tables/training_curves_summary.csv`

---

## 10. Report asset preparation

Prepare the final report assets by copying selected analysis outputs into the report folders.

### Recommended report-ready copies

Copy these figures into `reports/figures/` if they are not already there:

* `artifacts/analysis/loss_vs_params.png`
* `artifacts/analysis/loss_vs_tokens.png`
* `artifacts/analysis/loss_vs_flops.png`
* `artifacts/analysis/compute_frontier.png`

Copy these tables into `reports/tables/` if they are not already there:

* `results/results.csv`
* `artifacts/analysis/cleaned_results.csv`
* `artifacts/analysis/compute_frontier.csv`

### Important distinction

* `artifacts/` contains runtime and local reproducibility outputs.
* `reports/figures/` and `reports/tables/` contain report-ready assets.

---

## 11. One-command pipeline

A root-level pipeline script is also available.

### Small sweep only, skip tokenizer and data prep

```powershell
.\run_all.ps1 -Mode small -SkipTokenizer -SkipDataPrep
```

### Full pipeline

```powershell
.\run_all.ps1 -Mode all
```

### Notes

- `run_all.ps1` also runs the training-curve plotting stage and generates the corresponding report figures and summary table.

---

## 12. Recommended execution order

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

### Notes

* `03_train_model.ps1` and `04_run_single_experiment.ps1` are optional sanity-check steps before the full sweep.
* If the expensive sweep runs are already complete, you can rerun only the post-sweep stages (`06`, `07`, `08`) as needed.

---

## 13. Final submission checklist

Before packaging the submission, verify that the following exist:

### Core deliverables

* `results.csv` at the repository root
* `reports/report.md`
* `reports/report.pdf`

### Report assets

* figures under `reports/figures/`
* tables under `reports/tables/`

### Runtime artifacts kept locally

* `artifacts/tokenizer/`
* `artifacts/data/`
* `artifacts/runs/`
* `artifacts/analysis/`

### Final sanity checks

* the tokenizer is frozen and reused across all runs
* the final selected vocabulary size is `8000`
* the dataset used is `alexliap/tinystories-gr`, column `greek_translation`
* the experiment grid is 3 model sizes × 3 token budgets
* validation loss is the main metric used in the scaling-law analysis

```


```
