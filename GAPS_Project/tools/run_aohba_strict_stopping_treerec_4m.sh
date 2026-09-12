#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    export|cache|train) ;;
    *)
        echo "usage: $0 [export|cache|train]" >&2
        exit 2
        ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
PROVENANCE=${PROVENANCE:-/mnt/aohba/aohba_strict_stopping_treerec_4m_provenance}
CACHE=${CACHE:-/mnt/aohba/aohba_strict_stopping_treerec_4m_global_log}
LOGDIR=${LOGDIR:-"$HOME/aohba_strict_stopping_treerec_4m_logs"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_strict_stopping_treerec_4m"}
ANTIP_INPUT=${ANTIP_INPUT:-'/mnt/aohba/GAPS_Sim_2tof/antiP/*.root'}
ANTID_INPUT=${ANTID_INPUT:-'/mnt/aohba/GAPS_Sim_2tof/antiD/*.root'}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-2000000}
TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-1600000}
VAL_PER_CLASS=${VAL_PER_CLASS:-200000}
TEST_PER_CLASS=${TEST_PER_CLASS:-200000}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
TAG="aohba_strict_stopping_treerec_4m_global_log_seed${SEED}"

cd "$PROJECT"

activate_naka()
{
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

latest_run_dir()
{
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${TAG}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

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

    export_particle()
    {
        local particle=$1
        local label=$2
        local input_glob=$3
        local geometry=$4
        local output="$PROVENANCE/$particle"

        if [[ -f "$output/_SUCCESS" ]]; then
            echo "[SKIP] $particle provenance already complete"
            return
        fi
        if [[ -e "$output" ]]; then
            echo "ERROR: incomplete provenance exists: $output" >&2
            return 1
        fi

        echo "[START] $particle: $EVENTS_PER_CLASS strict-stop events"
        /usr/bin/time -f 'wall_seconds=%e' \
            -o "$LOGDIR/export_${particle}.time" \
            env LD_PRELOAD="$preload" "$exe" \
            --input "$input_glob" \
            --geometry-file "$geometry" \
            --output-npy-dir "$output" \
            --max-events "$EVENTS_PER_CLASS" \
            --target-label "$label" \
            --selection stopped-toptrigger \
            --provenance-only \
            2>&1 \
            | sed "s/^/[$particle] /" \
            | tee "$LOGDIR/export_${particle}.log"
        echo "[DONE] $particle"
    }

    export_particle antiP 0 \
        "$ANTIP_INPUT" \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!
    export_particle antiD 1 \
        "$ANTID_INPUT" \
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
    python -u - \
        "$PROVENANCE" "$EVENTS_PER_CLASS" \
        "$TRAIN_PER_CLASS" "$VAL_PER_CLASS" "$TEST_PER_CLASS" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = int(sys.argv[2])
train = int(sys.argv[3])
val = int(sys.argv[4])
test = int(sys.argv[5])
if train + val + test != expected:
    raise RuntimeError("train/val/test counts do not sum to events per class")

summary = {"events_per_class": expected, "splits": {}, "particles": {}}
ranges = {
    "train": (0, train),
    "val": (train, train + val),
    "test": (train + val, expected),
}
for particle, label in (("antiP", 0), ("antiD", 1)):
    directory = root / particle
    manifest = json.loads((directory / "export_manifest.json").read_text())
    if manifest["selection"] != "stopped-toptrigger":
        raise RuntimeError(f"{particle}: incorrect selection")
    if manifest["events"] != expected or not manifest["provenance_only"]:
        raise RuntimeError(f"{particle}: incorrect manifest counts")
    arrays = {
        name: np.load(directory / name, mmap_mode="r")
        for name in (
            "labels.npy", "betas.npy", "source_file_indices.npy",
            "source_entries.npy", "random_seeds.npy",
        )
    }
    if {len(array) for array in arrays.values()} != {expected}:
        raise RuntimeError(f"{particle}: provenance length mismatch")
    if not np.all(arrays["labels.npy"] == label):
        raise RuntimeError(f"{particle}: label mismatch")
    if not np.isfinite(arrays["betas.npy"]).all():
        raise RuntimeError(f"{particle}: non-finite beta")

    split_sources = {}
    for split, (start, stop) in ranges.items():
        split_sources[split] = set(
            np.unique(arrays["source_file_indices.npy"][start:stop]).tolist()
        )
    overlaps = {
        "train_val": len(split_sources["train"] & split_sources["val"]),
        "train_test": len(split_sources["train"] & split_sources["test"]),
        "val_test": len(split_sources["val"] & split_sources["test"]),
    }
    summary["particles"][particle] = {
        "events": expected,
        "beta_min": float(np.min(arrays["betas.npy"])),
        "beta_max": float(np.max(arrays["betas.npy"])),
        "source_files_used": int(len(np.unique(arrays["source_file_indices.npy"]))),
        "source_file_overlap_counts": overlaps,
    }

for split, (start, stop) in ranges.items():
    summary["splits"][split] = {
        "events_per_class": stop - start,
        "events": 2 * (stop - start),
    }

(root / "provenance_audit.json").write_text(json.dumps(summary, indent=2))
(root / "_SUCCESS").write_text("ok\n")
print(json.dumps(summary, indent=2))
print("AOHBA STRICT-STOPPING TREEREC 4M PROVENANCE: VALID")
PY
}

run_cache()
{
    activate_naka
    [[ -f "$PROVENANCE/_SUCCESS" ]] || {
        echo "ERROR: incomplete provenance: $PROVENANCE" >&2
        exit 1
    }
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
        --train-events-per-class "$TRAIN_PER_CLASS" \
        --val-events-per-class "$VAL_PER_CLASS" \
        --test-events-per-class "$TEST_PER_CLASS" \
        --chunk-size 10000 \
        --k 8

    python - "$CACHE/cache_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
expected = {
    "train": (3_200_000, {"0": 1_600_000, "1": 1_600_000}),
    "val": (400_000, {"0": 200_000, "1": 200_000}),
    "test": (400_000, {"0": 200_000, "1": 200_000}),
}
for split, (events, labels) in expected.items():
    row = manifest["splits"][split]
    if row["events"] != events:
        raise RuntimeError(f"{split}: event count mismatch")
    observed = {str(key): value for key, value in row["label_counts"].items()}
    if observed != labels:
        raise RuntimeError(f"{split}: label count mismatch")
print("AOHBA STRICT-STOPPING TREEREC 4M CACHE: VALID")
PY
}

run_train()
{
    activate_naka
    [[ -f "$CACHE/_SUCCESS" && -f "$CACHE/cache_manifest.json" ]] || {
        echo "ERROR: incomplete cache: $CACHE" >&2
        exit 1
    }
    mkdir -p "$RESULT_ROOT"

    local run_dir checkpoint model eval_dir
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
    eval_dir="$run_dir/evaluation_test"

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$model" \
        --output-dir "$eval_dir" \
        --model gravnet \
        --batch-size 512 \
        --seed "$SEED"

    python -u src/scripts/evaluate_beta_matched.py \
        --result "strict-stop-4M=$eval_dir" \
        --bins 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
        --repeats 20 \
        --seed "$SEED" \
        --output "$eval_dir/beta_matched_metrics.json"

    echo "model      : $model"
    echo "evaluation : $eval_dir"
}

echo "phase                  : $PHASE"
echo "selection              : strict track stop + truth top-trigger"
echo "events per class       : $EVENTS_PER_CLASS"
echo "train/val/test per class: $TRAIN_PER_CLASS/$VAL_PER_CLASS/$TEST_PER_CLASS"
echo "physical GPU           : $GPU"
echo "seed                   : $SEED"
df -h /mnt/aohba

case "$PHASE" in
    export) run_export ;;
    cache) run_cache ;;
    train) run_train ;;
esac

echo "AOHBA STRICT-STOPPING TREEREC 4M $PHASE: COMPLETE"
