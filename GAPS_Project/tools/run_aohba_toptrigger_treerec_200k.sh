#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-all}
case "$PHASE" in
    export|cache|train|cache-train|all) ;;
    *)
        echo "usage: $0 [export|cache|train|cache-train|all]" >&2
        exit 2
        ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
PROVENANCE=${PROVENANCE:-/mnt/aohba/aohba_treemc_toptrigger_only_200k_provenance}
CACHE=${CACHE:-/mnt/aohba/aohba_toptrigger_only_treerec_200k_global_log}
LOGDIR=${LOGDIR:-"$HOME/aohba_toptrigger_only_200k_logs"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_toptrigger_only_treerec_200k"}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-100000}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
TAG="aohba_toptrigger_only_treerec_200k_global_log_seed${SEED}"

cd "$PROJECT"

run_export()
{
    source "$HOME/setup_ynakagami_root.sh"
    mkdir -p build/tools "$PROVENANCE" "$LOGDIR"

    /usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
        -I/home/ynakagami/simpledet/SimpleDet/common/include \
        -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
        tools/export/export_treemc_topiso_like_csv.cc \
        -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
        $(root-config --libs) \
        -o build/tools/export_treemc_topiso_like_csv

    local exe="$PROJECT/build/tools/export_treemc_topiso_like_csv"
    local preload
    preload="/home/ynakagami/GEANT/install/lib/libG4geometry.so"
    preload+=":/home/ynakagami/GEANT/install/lib/libG4event.so"
    preload+=":/home/ynakagami/simpledet/build/common/libGAPSCommon.so"

    export_one()
    {
        local particle=$1
        local label=$2
        local input_glob=$3
        local geometry=$4
        local output="$PROVENANCE/$particle"
        local log="$LOGDIR/export_${particle}.log"

        if [[ -f "$output/_SUCCESS" ]]; then
            echo "[SKIP] $particle provenance already complete"
            return
        fi
        if [[ -e "$output" ]]; then
            echo "ERROR: incomplete output exists: $output" >&2
            return 1
        fi

        echo "[START] $particle top-trigger provenance"
        /usr/bin/time -f 'wall_seconds=%e' \
            -o "$LOGDIR/export_${particle}.time" \
            env LD_PRELOAD="$preload" "$exe" \
            --input "$input_glob" \
            --geometry-file "$geometry" \
            --output-npy-dir "$output" \
            --max-events "$EVENTS_PER_CLASS" \
            --target-label "$label" \
            --selection toptrigger \
            --provenance-only \
            >"$log" 2>&1
        echo "[DONE] $particle top-trigger provenance"
    }

    export_one \
        antiP 0 \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!

    export_one \
        antiD 1 \
        /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root \
        /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
    local pid_antid=$!

    local status=0
    if ! wait "$pid_antip"; then status=1; fi
    if ! wait "$pid_antid"; then status=1; fi
    if [[ "$status" -ne 0 ]]; then
        echo "ERROR: provenance export failed; inspect $LOGDIR" >&2
        exit 1
    fi

    activate_naka
    python - "$PROVENANCE" "$EVENTS_PER_CLASS" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = int(sys.argv[2])
for particle, label in (("antiP", 0), ("antiD", 1)):
    directory = root / particle
    manifest = json.loads((directory / "export_manifest.json").read_text())
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    entries = np.load(directory / "source_entries.npy", mmap_mode="r")
    files = np.load(directory / "source_file_indices.npy", mmap_mode="r")
    assert manifest["selection"] == "toptrigger"
    assert manifest["provenance_only"] is True
    assert len(labels) == len(entries) == len(files) == expected
    assert np.all(labels == label)
    assert np.all(entries >= 0)
    assert np.all(files >= 0)
    print(f"{particle}: {expected:,} top-trigger events")
print("TOP-TRIGGER PROVENANCE: VALID")
PY
}

activate_naka()
{
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

run_cache()
{
    activate_naka
    if [[ -f "$CACHE/_SUCCESS" && -f "$CACHE/cache_manifest.json" ]]; then
        echo "[SKIP] cache already complete: $CACHE"
        return
    fi
    if [[ -e "$CACHE" ]]; then
        echo "ERROR: incomplete cache exists: $CACHE" >&2
        exit 1
    fi
    python -u src/data_parse/build_aohba_fixedgrid_selected_treerec_cache.py \
        --fixedgrid-raw-dir "$PROVENANCE" \
        --output-dir "$CACHE" \
        --chunk-size 10000 \
        --k 8
}

latest_run_dir()
{
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${TAG}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

run_train()
{
    activate_naka
    [[ -f "$CACHE/_SUCCESS" ]] || {
        echo "ERROR: cache is incomplete: $CACHE" >&2
        exit 1
    }
    mkdir -p "$RESULT_ROOT"

    local run_dir model checkpoint
    run_dir=$(latest_run_dir || true)
    if [[ -n "$run_dir" && -f "$run_dir/evaluation_test/metrics.json" ]]; then
        echo "[SKIP] training and evaluation already complete: $run_dir"
        return
    fi

    checkpoint=""
    if [[ -n "$run_dir" ]]; then
        checkpoint=$(find "$run_dir" -maxdepth 1 -type f \
            -name '*_last_checkpoint.pth' -print | sort | tail -1)
    fi

    local train_args=(
        --split-cache-dir "$CACHE"
        --epochs 80
        --model gravnet
        --batch-size 512
        --num-workers 2
        --non-blocking-transfer
        --seed "$SEED"
        --dataset-tag "$TAG"
        --result-dir "$RESULT_ROOT"
    )
    if [[ -n "$checkpoint" ]]; then
        echo "[RESUME] $checkpoint"
        train_args+=(--resume-checkpoint "$checkpoint")
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_aohba.py "${train_args[@]}"

    run_dir=$(latest_run_dir)
    model=$(find "$run_dir" -maxdepth 1 -type f \
        -name '*_best.pth' -print | sort | tail -1)
    [[ -n "$model" ]] || {
        echo "ERROR: best checkpoint not found" >&2
        exit 1
    }

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$model" \
        --output-dir "$run_dir/evaluation_test" \
        --model gravnet \
        --batch-size 512 \
        --seed "$SEED"

    echo "model: $model"
    echo "evaluation: $run_dir/evaluation_test"
}

echo "phase           : $PHASE"
echo "selection       : toptrigger only"
echo "provenance      : $PROVENANCE"
echo "cache           : $CACHE"
echo "physical GPU    : $GPU"
echo "seed            : $SEED"

case "$PHASE" in
    export) run_export ;;
    cache) run_cache ;;
    train) run_train ;;
    cache-train) run_cache; run_train ;;
    all) run_export; run_cache; run_train ;;
esac

echo "AOHBA TOP-TRIGGER-ONLY TREEREC 200K $PHASE: COMPLETE"
