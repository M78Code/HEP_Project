#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results/ohba_all_usable_leave_one_position_out}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[START] position-holdout offset audit"
echo "results: $RESULTS_ROOT"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.audit_ohba_position_holdout_offsets \
    --results-root "$RESULTS_ROOT"

echo "OHBA POSITION-HOLDOUT OFFSET AUDIT: COMPLETE"
