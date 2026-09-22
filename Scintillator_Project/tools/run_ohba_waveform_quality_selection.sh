#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_DIR/dataset/split}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/results/ohba_quality_selection}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$RESULT_DIR"

echo "[START] Ohba waveform-quality selection scan"
echo "project : $PROJECT_DIR"
echo "splits  : $SPLIT_DIR"
echo "results : $RESULT_DIR"
echo "rule    : train calibration, validation selection, held-out test"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.scan_ohba_waveform_quality \
    --split-dir "$SPLIT_DIR" \
    --output-dir "$RESULT_DIR" \
    --retentions "${RETENTIONS:-1.00,0.99,0.98,0.95,0.90,0.85,0.80}" \
    --minimum-validation-retention "${MINIMUM_VALIDATION_RETENTION:-0.80}" \
    --minimum-train-events-per-position "${MINIMUM_TRAIN_EVENTS_PER_POSITION:-80}" \
    2>&1 | tee "$RESULT_DIR/quality_selection.log"

echo "OHBA WAVEFORM-QUALITY SELECTION: COMPLETE"
