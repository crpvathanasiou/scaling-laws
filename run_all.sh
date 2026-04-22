#!/usr/bin/env bash
set -e

echo "1. Train tokenizer"
bash scripts/01_tokenizer.sh

echo "2. Prepare tokenized data"
bash scripts/03_prepare_data.sh

echo "3. Run full sweep"
bash scripts/05_run_sweep.sh

echo "4. Fit scaling laws"
bash scripts/06_fit_scaling_laws.sh

echo "Done."