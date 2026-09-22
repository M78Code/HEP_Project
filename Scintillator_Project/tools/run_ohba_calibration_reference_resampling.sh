#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results/ohba_all_usable_leave_one_position_out}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

echo "[START] calibration-reference resampling"
echo "results root      : $RESULTS_ROOT"
echo "reference pool    : ${REFERENCE_POOL:-100} events per position"
echo "calibration counts: ${CALIBRATION_COUNTS:-0,10,30,100}"
echo "repeats           : ${REPEATS:-500}"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.resample_ohba_calibration_transfer \
    --results-root "$RESULTS_ROOT" \
    --reference-pool "${REFERENCE_POOL:-100}" \
    --calibration-counts "${CALIBRATION_COUNTS:-0,10,30,100}" \
    --repeats "${REPEATS:-500}" \
    --seed "${SEED:-20260922}"

echo "OHBA CALIBRATION-REFERENCE RESAMPLING: COMPLETE"
