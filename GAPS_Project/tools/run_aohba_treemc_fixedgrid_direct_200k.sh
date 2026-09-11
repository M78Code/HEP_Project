#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
RAW=${RAW:-/mnt/aohba/aohba_treemc_fixedgrid_direct_200k_raw}
DATASET=${DATASET:-/mnt/aohba/aohba_treemc_fixedgrid_direct_200k}
LOGDIR=${LOGDIR:-"$HOME/aohba_treemc_fixedgrid_direct_200k_logs"}
EVENTS_PER_CLASS=100000

cd "$PROJECT"
source "$HOME/setup_ynakagami_root.sh"

mkdir -p build/tools "$RAW" "$LOGDIR"
/usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
    -I/home/ynakagami/simpledet/SimpleDet/common/include \
    -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
    tools/export/export_treemc_topiso_like_csv.cc \
    -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
    $(root-config --libs) \
    -o build/tools/export_treemc_topiso_like_csv

EXE="$PROJECT/build/tools/export_treemc_topiso_like_csv"
PRELOAD="/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so"

run_export() {
    local particle=$1
    local label=$2
    local input_glob=$3
    local geometry=$4
    local output="$RAW/$particle"
    local log="$LOGDIR/$particle.log"

    if test -f "$output/_SUCCESS"; then
        echo "[SKIP] $particle direct export already complete"
        return 0
    fi
    if test -e "$output"; then
        echo "ERROR: incomplete output exists: $output" >&2
        return 1
    fi

    echo "[START] $particle: $EVENTS_PER_CLASS selected events"
    /usr/bin/time -f 'wall_seconds=%e' -o "$LOGDIR/$particle.time" \
        env LD_PRELOAD="$PRELOAD" "$EXE" \
        --input "$input_glob" \
        --geometry-file "$geometry" \
        --output-npy-dir "$output" \
        --max-events "$EVENTS_PER_CLASS" \
        --target-label "$label" \
        >"$log" 2>&1
    echo "[DONE] $particle"
}

run_export \
    antiP 0 \
    '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' \
    /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
pid_antip=$!

run_export \
    antiD 1 \
    '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' \
    /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
pid_antid=$!

status=0
if ! wait "$pid_antip"; then status=1; fi
if ! wait "$pid_antid"; then status=1; fi
if test "$status" -ne 0; then
    echo "ERROR: at least one direct export failed; inspect $LOGDIR" >&2
    exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

if test -f "$DATASET/_SUCCESS"; then
    echo "[SKIP] balanced dataset already complete: $DATASET"
elif test -e "$DATASET"; then
    echo "ERROR: incomplete dataset exists: $DATASET" >&2
    exit 1
else
    python -u src/scripts/build_treemc_fixedgrid_binary_dataset.py \
        --antip-dir "$RAW/antiP" \
        --antid-dir "$RAW/antiD" \
        --output-dir "$DATASET" \
        --train-per-class 80000 \
        --val-per-class 10000 \
        --test-per-class 10000
fi

python - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
manifest = json.loads((root / "dataset_manifest.json").read_text())
expected = {"train": 160000, "val": 20000, "test": 20000}
for split, count in expected.items():
    directory = root / f"{split}_nakagami_style_4M"
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    voxels = np.load(directory / "voxels.npy", mmap_mode="r")
    primary = np.load(directory / "tof_primary.npy", mmap_mode="r")
    paddles = np.load(directory / "tof_paddles.npy", mmap_mode="r")
    values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(values.tolist(), counts.tolist()))
    assert len(labels) == count
    assert label_counts == {0: count // 2, 1: count // 2}
    assert voxels.shape == (count, 10, 12, 12)
    assert primary.shape == (count, 11)
    assert paddles.shape == (count, 172)
    assert not np.any(paddles)
    print(f"{split}: events={count:,} labels={label_counts}")
assert manifest["events"] == 200000
print("AOHBA TREEMC FIXED-GRID DIRECT 200K DATASET: VALID")
PY

du -sh "$RAW" "$DATASET"
echo "raw output: $RAW"
echo "training dataset: $DATASET"
echo "AOHBA TREEMC FIXED-GRID DIRECT 200K: COMPLETE"
