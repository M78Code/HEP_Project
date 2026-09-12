#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
MAX_ENTRIES=${MAX_ENTRIES:-1000000}
OUT=${OUT:-"$PROJECT/results/aohba_stopping_definition_overlap_${MAX_ENTRIES}"}
LOGDIR=${LOGDIR:-"$HOME/aohba_stopping_definition_overlap_logs_${MAX_ENTRIES}"}

cd "$PROJECT"
source "$HOME/setup_ynakagami_root.sh"
mkdir -p build/tools "$OUT" "$LOGDIR"

/usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
    -I/home/ynakagami/simpledet/SimpleDet/common/include \
    -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
    tools/audit/audit_treemc_selection_counts.cc \
    -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
    $(root-config --libs) \
    -o build/tools/audit_treemc_selection_counts

EXE="$PROJECT/build/tools/audit_treemc_selection_counts"
PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so
PRELOAD+=:/home/ynakagami/GEANT/install/lib/libG4event.so
PRELOAD+=:/home/ynakagami/simpledet/build/common/libGAPSCommon.so

run_one()
{
    local particle=$1
    local label=$2
    local input=$3
    local output="$OUT/${particle}.csv"

    echo "[START] $particle max_entries=$MAX_ENTRIES"
    /usr/bin/time -f 'wall_seconds=%e' \
        -o "$LOGDIR/${particle}.time" \
        env LD_PRELOAD="$PRELOAD" "$EXE" \
        --input "$input" \
        --max-entries "$MAX_ENTRIES" \
        --target-label "$label" \
        >"$output" 2>"$LOGDIR/${particle}.log"
    echo "[DONE] $particle"
}

run_one antiP 0 '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' &
PID_ANTIP=$!
run_one antiD 1 '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' &
PID_ANTID=$!

STATUS=0
if ! wait "$PID_ANTIP"; then STATUS=1; fi
if ! wait "$PID_ANTID"; then STATUS=1; fi
if [[ "$STATUS" -ne 0 ]]; then
    echo "ERROR: audit failed; inspect $LOGDIR" >&2
    exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

python - "$OUT" "$MAX_ENTRIES" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
max_entries = int(sys.argv[2])
summary = {"max_entries_per_particle": max_entries, "particles": {}}

for particle in ("antiP", "antiD"):
    with (output / f"{particle}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"{particle}: expected one CSV row, found {len(rows)}")
    row = rows[0]
    values = {
        key: int(value)
        for key, value in row.items()
        if key != "input"
    }
    summary_count = values["summary_tracker_stopped"]
    legacy_count = values["stopped"]
    overlap = values["summary_and_legacy"]
    union = summary_count + legacy_count - overlap
    metrics = {
        "counts": values,
        "summary_recall_by_legacy": overlap / legacy_count if legacy_count else None,
        "legacy_recall_by_summary": overlap / summary_count if summary_count else None,
        "jaccard_overlap": overlap / union if union else None,
    }
    summary["particles"][particle] = metrics

    print(f"\n===== {particle} =====")
    print(f"beta-range events            : {values['beta_in_range']:,}")
    print(f"stopping-volume tracker      : {summary_count:,}")
    print(f"kinetic-zero in tracker      : {values['kinetic_zero_in_tracker']:,}")
    print(f"any zero-length step         : {values['has_zero_step']:,}")
    print(f"same-step zero condition     : {values['same_step_zero']:,}")
    print(f"legacy strict track stopped  : {legacy_count:,}")
    print(f"both definitions             : {overlap:,}")
    print(f"stopping-volume only         : {values['summary_only']:,}")
    print(f"legacy-track only            : {values['legacy_only']:,}")
    print(f"legacy covered by summary    : {metrics['summary_recall_by_legacy']:.8f}")
    print(f"summary covered by legacy    : {metrics['legacy_recall_by_summary']:.8f}")
    print(f"Jaccard overlap              : {metrics['jaccard_overlap']:.8f}")

(output / "overlap_summary.json").write_text(json.dumps(summary, indent=2))
(output / "_SUCCESS").write_text("ok\n")
print(f"\nsaved: {output / 'overlap_summary.json'}")
print("AOHBA STOPPING-DEFINITION OVERLAP AUDIT: VALID")
PY

echo "===== timing ====="
for file in "$LOGDIR"/*.time; do
    echo "$(basename "$file"): $(cat "$file")"
done
echo "AOHBA STOPPING-DEFINITION OVERLAP PILOT: COMPLETE"
