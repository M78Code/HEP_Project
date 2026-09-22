#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/dataset/tar_zip/A}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/results/ohba_traditional_reproduction}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$RESULT_DIR"

echo "[START] Ohba traditional-method protocol reproduction"
echo "project : $PROJECT_DIR"
echo "raw     : $RAW_DIR"
echo "results : $RESULT_DIR"
echo "metric  : all-event Gaussian-fit sigma; not train/test RMSE"
echo "CFD scan: 0.10, 0.20, 0.30, 0.40, 0.50"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.reproduce_ohba_traditional \
    --raw-dir "$RAW_DIR" \
    --output-dir "$RESULT_DIR" \
    --baseline-range "${BASELINE_RANGE:-0:400}" \
    --integration-range "${INTEGRATION_RANGE:-450:800}" \
    --cfd-fractions "${CFD_FRACTIONS:-0.10,0.20,0.30,0.40,0.50}" \
    --mode-bins "${MODE_BINS:-120}" \
    --residual-range "${RESIDUAL_MIN:--40}" "${RESIDUAL_MAX:-40}" \
    --residual-bin-width "${RESIDUAL_BIN_WIDTH:-0.5}" \
    --weight-step "${WEIGHT_STEP:-0.01}" \
    2>&1 | tee "$RESULT_DIR/reproduction.log"

echo "OHBA TRADITIONAL REPRODUCTION: COMPLETE"
