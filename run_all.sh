#!/usr/bin/env bash
set -e

echo "1. Profile tokenizer"
bash scripts/01_profile_tokenizer.sh

echo "2. Train final tokenizer"
bash scripts/02_train_tokenizer.sh

echo "3. Prepare tokenized data"
bash scripts/03_prepare_data.sh

echo "4. Run full sweep"
bash scripts/05_run_sweep.sh

echo "5. Fit scaling laws"
bash scripts/06_fit_scaling_laws.sh

echo "Done."