#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-small}"

SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
SKIP_SCALING_FITS="${SKIP_SCALING_FITS:-0}"
SKIP_FRONTIER="${SKIP_FRONTIER:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "Scaling Laws Pipeline"
echo "Mode: $MODE"
echo "SKIP_TOKENIZER: $SKIP_TOKENIZER"
echo "SKIP_DATA_PREP: $SKIP_DATA_PREP"
echo "SKIP_SWEEP: $SKIP_SWEEP"
echo "SKIP_SCALING_FITS: $SKIP_SCALING_FITS"
echo "SKIP_FRONTIER: $SKIP_FRONTIER"
echo "Working directory: $REPO_ROOT"
echo "========================================"

if [ "$SKIP_TOKENIZER" = "0" ]; then
  echo
  echo "[1/5] START Tokenizer"
  bash ./scripts/01_tokenizer.sh
  echo "[1/5] DONE Tokenizer"
else
  echo
  echo "[1/5] SKIP Tokenizer"
fi

if [ "$SKIP_DATA_PREP" = "0" ]; then
  echo
  echo "[2/5] START Data preparation"
  bash ./scripts/02_prepare_data.sh
  echo "[2/5] DONE Data preparation"
else
  echo
  echo "[2/5] SKIP Data preparation"
fi

if [ "$SKIP_SWEEP" = "0" ]; then
  echo
  echo "[3/5] START Sweep ($MODE)"
  bash ./scripts/05_run_sweep.sh "$MODE"
  echo "[3/5] DONE Sweep ($MODE)"
else
  echo
  echo "[3/5] SKIP Sweep"
fi

if [ "$SKIP_SCALING_FITS" = "0" ]; then
  echo
  echo "[4/5] START Scaling-law fitting"
  bash ./scripts/06_fit_scaling_laws.sh
  echo "[4/5] DONE Scaling-law fitting"
else
  echo
  echo "[4/5] SKIP Scaling-law fitting"
fi

if [ "$SKIP_FRONTIER" = "0" ]; then
  echo
  echo "[5/5] START Compute frontier"
  bash ./scripts/07_compute_frontier.sh
  echo "[5/5] DONE Compute frontier"
else
  echo
  echo "[5/5] SKIP Compute frontier"
fi

echo
echo "========================================"
echo "Pipeline completed successfully."
echo "========================================"
