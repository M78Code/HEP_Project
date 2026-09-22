#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-$HOME/HEP_Project/Scintillator_Project}"
RESULTS="${RESULTS:-$PROJECT/results}"
OUTPUT="${OUTPUT:-$RESULTS/ohba_hybrid_report_20260922}"

cd "$PROJECT"
echo "[START] build Ohba hybrid reporting package"
echo "project : $PROJECT"
echo "results : $RESULTS"
echo "output  : $OUTPUT"

python3 src/scripts/build_ohba_hybrid_report.py \
  --results "$RESULTS" \
  --output "$OUTPUT"
