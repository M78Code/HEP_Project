#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/results/ohba_quality_selected_hybrid/seed_20260825}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/audit}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR"

echo "[START] quality-selected hybrid audit"
echo "run    : $RUN_DIR"
echo "output : $OUTPUT_DIR"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.audit_ohba_quality_selected_hybrid \
    --predictions "$RUN_DIR/predictions.npz" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-repeats "${BOOTSTRAP_REPEATS:-2000}" \
    --seed "${SEED:-20260825}"

echo "OHBA QUALITY-SELECTED HYBRID AUDIT: COMPLETE"
