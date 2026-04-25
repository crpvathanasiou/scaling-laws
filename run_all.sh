#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-small}"

SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
SKIP_SCALING_FITS="${SKIP_SCALING_FITS:-0}"
SKIP_FRONTIER="${SKIP_FRONTIER:-0}"
SKIP_TRAINING_CURVES="${SKIP_TRAINING_CURVES:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Installing dependencies from poetry.lock..."
poetry install --no-interaction

if [ -f ./results/results.csv ]; then
  rm -f ./results/results.csv
  echo "Removed existing runtime file: ./results/results.csv"
fi

echo "========================================"
echo "Scaling Laws Pipeline"
echo "Mode: $MODE"
echo "SKIP_TOKENIZER: $SKIP_TOKENIZER"
echo "SKIP_DATA_PREP: $SKIP_DATA_PREP"
echo "SKIP_SWEEP: $SKIP_SWEEP"
echo "SKIP_SCALING_FITS: $SKIP_SCALING_FITS"
echo "SKIP_FRONTIER: $SKIP_FRONTIER"
echo "Working directory: $REPO_ROOT"
echo "SKIP_TRAINING_CURVES: $SKIP_TRAINING_CURVES"
echo "========================================"

if [ "$SKIP_TOKENIZER" = "0" ]; then
  echo
  echo "[1/6] START Tokenizer"
  bash ./scripts/01_tokenizer.sh
  echo "[1/6] DONE Tokenizer"
else
  echo
  echo "[1/6] SKIP Tokenizer"
fi

if [ "$SKIP_DATA_PREP" = "0" ]; then
  echo
  echo "[2/6] START Data preparation"
  bash ./scripts/02_prepare_data.sh
  echo "[2/6] DONE Data preparation"
else
  echo
  echo "[2/6] SKIP Data preparation"
fi

if [ "$SKIP_SWEEP" = "0" ]; then
  echo
  echo "[3/6] START Sweep ($MODE)"
  bash ./scripts/05_run_sweep.sh "$MODE"
  echo "[3/6] DONE Sweep ($MODE)"
else
  echo
  echo "[3/6] SKIP Sweep"
fi

if [ "$SKIP_SCALING_FITS" = "0" ]; then
  echo
  echo "[4/6] START Scaling-law fitting"
  bash ./scripts/06_fit_scaling_laws.sh
  echo "[4/6] DONE Scaling-law fitting"
else
  echo
  echo "[4/6] SKIP Scaling-law fitting"
fi

if [ "$SKIP_FRONTIER" = "0" ]; then
  echo
  echo "[5/6] START Compute frontier"
  bash ./scripts/07_compute_frontier.sh
  echo "[5/6] DONE Compute frontier"
else
  echo
  echo "[5/6] SKIP Compute frontier"
fi
if [ "$SKIP_TRAINING_CURVES" = "0" ]; then
  echo
  echo "[6/6] START Training-curve plotting"
  bash ./scripts/08_plot_training_curves.sh
  echo "[6/6] DONE Training-curve plotting"
else
  echo
  echo "[6/6] SKIP Training-curve plotting"
fi

echo
echo "========================================"
echo "Pipeline completed successfully."
echo "========================================"
