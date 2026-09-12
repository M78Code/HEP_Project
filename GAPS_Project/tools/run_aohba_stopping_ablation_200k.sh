#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    export-match|cache-train|compare) ;;
    *)
        echo "usage: $0 [export-match|cache-train|compare]" >&2
        exit 2
        ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CANDIDATES=${CANDIDATES:-/mnt/aohba/aohba_stopping_ablation_candidates_300k}
MATCHED=${MATCHED:-/mnt/aohba/aohba_stopping_ablation_beta_matched_200k}
LOGDIR=${LOGDIR:-"$HOME/aohba_stopping_ablation_200k_logs"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_stopping_ablation_200k"}
CANDIDATES_PER_CLASS=${CANDIDATES_PER_CLASS:-300000}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-100000}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
GROUP_A=${GROUP_A:-stopped}
GROUP_B=${GROUP_B:-nonstopped}
SELECTION_A=${SELECTION_A:-stopped-toptrigger}
SELECTION_B=${SELECTION_B:-toptrigger-nonstopped}
LABEL_A=${LABEL_A:-Stopped in tracker + top-trigger}
LABEL_B=${LABEL_B:-Not stopped + top-trigger}

cd "$PROJECT"

activate_naka()
{
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

run_export_match()
{
    source "$HOME/setup_ynakagami_root.sh"
    mkdir -p build/tools "$CANDIDATES" "$LOGDIR"

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

    export_candidate()
    {
        local group=$1
        local selection=$2
        local particle=$3
        local label=$4
        local input_glob=$5
        local geometry=$6
        local output="$CANDIDATES/$group/$particle"
        local log="$LOGDIR/export_${group}_${particle}.log"

        if [[ -f "$output/_SUCCESS" ]]; then
            echo "[SKIP] $group/$particle candidate export"
            return
        fi
        if [[ -e "$output" ]]; then
            echo "ERROR: incomplete candidate directory exists: $output" >&2
            return 1
        fi
        mkdir -p "$CANDIDATES/$group"

        echo "[START] $group/$particle selection=$selection"
        /usr/bin/time -f 'wall_seconds=%e' \
            -o "$LOGDIR/export_${group}_${particle}.time" \
            env LD_PRELOAD="$preload" "$exe" \
            --input "$input_glob" \
            --geometry-file "$geometry" \
            --output-npy-dir "$output" \
            --max-events "$CANDIDATES_PER_CLASS" \
            --target-label "$label" \
            --selection "$selection" \
            --provenance-only \
            >"$log" 2>&1
        echo "[DONE] $group/$particle"
    }

    export_group()
    {
        local group=$1
        local selection=$2

        export_candidate \
            "$group" "$selection" antiP 0 \
            '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' \
            /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
        local pid_antip=$!

        export_candidate \
            "$group" "$selection" antiD 1 \
            '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' \
            /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
        local pid_antid=$!

        local status=0
        if ! wait "$pid_antip"; then status=1; fi
        if ! wait "$pid_antid"; then status=1; fi
        if [[ "$status" -ne 0 ]]; then
            echo "ERROR: $group candidate export failed; inspect $LOGDIR" >&2
            exit 1
        fi
    }

    export_group "$GROUP_A" "$SELECTION_A"
    export_group "$GROUP_B" "$SELECTION_B"

    activate_naka
    if [[ -f "$MATCHED/_SUCCESS" ]]; then
        echo "[SKIP] beta-bin-matched provenance already complete: $MATCHED"
        return
    fi
    if [[ -e "$MATCHED" ]]; then
        echo "ERROR: incomplete matched directory exists: $MATCHED" >&2
        exit 1
    fi

    python -u - \
        "$CANDIDATES" "$MATCHED" "$EVENTS_PER_CLASS" "$SEED" \
        "$GROUP_A" "$GROUP_B" "$SELECTION_A" "$SELECTION_B" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import numpy as np

candidates = Path(sys.argv[1])
output = Path(sys.argv[2])
target = int(sys.argv[3])
seed = int(sys.argv[4])
groups = (sys.argv[5], sys.argv[6])
particles = ("antiP", "antiD")
expected_selection = {
    groups[0]: sys.argv[7],
    groups[1]: sys.argv[8],
}
array_names = (
    "labels.npy",
    "betas.npy",
    "random_seeds.npy",
    "chain_entries.npy",
    "source_file_indices.npy",
    "source_entries.npy",
)

pools = {}
for group in groups:
    for particle in particles:
        directory = candidates / group / particle
        manifest = json.loads((directory / "export_manifest.json").read_text())
        if manifest["selection"] != expected_selection[group]:
            raise RuntimeError(f"{directory}: unexpected selection")
        arrays = {
            name: np.load(directory / name, mmap_mode="r")
            for name in array_names
        }
        lengths = {len(array) for array in arrays.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"{directory}: array length mismatch")
        pools[(group, particle)] = {
            "directory": directory,
            "manifest": manifest,
            "arrays": arrays,
        }

# A 0.005-wide bin is narrow compared with the generated beta interval while
# retaining enough events to construct identical four-way bin populations.
edges = np.linspace(0.20, 0.50, 61, dtype=np.float64)
per_bin = {}
for key, pool in pools.items():
    beta = np.asarray(pool["arrays"]["betas.npy"], dtype=np.float64)
    rows = []
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == len(edges) - 2:
            mask = (beta >= low) & (beta <= high)
        else:
            mask = (beta >= low) & (beta < high)
        rows.append(np.flatnonzero(mask))
    per_bin[key] = rows

capacity = np.array(
    [min(len(per_bin[key][i]) for key in pools) for i in range(len(edges) - 1)],
    dtype=np.int64,
)
if int(capacity.sum()) < target:
    raise RuntimeError(
        f"only {int(capacity.sum()):,} beta-bin-matched events per population; "
        "increase CANDIDATES_PER_CLASS and use a new CANDIDATES directory"
    )

raw = capacity.astype(np.float64) * (target / float(capacity.sum()))
allocation = np.minimum(np.floor(raw).astype(np.int64), capacity)
remaining = target - int(allocation.sum())
order = np.argsort(-(raw - allocation))
for bin_index in order:
    if remaining == 0:
        break
    if allocation[bin_index] < capacity[bin_index]:
        allocation[bin_index] += 1
        remaining -= 1
if remaining != 0 or int(allocation.sum()) != target:
    raise RuntimeError("failed to allocate the requested matched sample")

slot_bins = np.repeat(np.arange(len(allocation), dtype=np.int64), allocation)
np.random.default_rng(seed).shuffle(slot_bins)

output.mkdir(parents=True)
for population_index, (key, pool) in enumerate(pools.items()):
    group, particle = key
    rng = np.random.default_rng(seed + 1000 + population_index)
    selected = np.empty(target, dtype=np.int64)
    for bin_index, count in enumerate(allocation):
        if count == 0:
            continue
        choices = rng.choice(per_bin[key][bin_index], int(count), replace=False)
        rng.shuffle(choices)
        selected[np.flatnonzero(slot_bins == bin_index)] = choices

    destination = output / group / particle
    destination.mkdir(parents=True)
    for name, array in pool["arrays"].items():
        np.save(destination / name, np.asarray(array[selected]))
    shutil.copy2(pool["directory"] / "source_files.txt", destination)
    manifest = dict(pool["manifest"])
    manifest.update({
        "source": "beta-bin-matched TreeMc provenance",
        "events": target,
        "candidate_directory": str(pool["directory"]),
        "matching_seed": seed,
        "beta_bin_edges": edges.tolist(),
        "beta_bin_counts": allocation.tolist(),
        "split_events_per_class": {
            "train": 80_000,
            "val": 10_000,
            "test": 10_000,
        },
    })
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    (destination / "_SUCCESS").write_text("ok\n")

for start, stop, split in ((0, 80_000, "train"), (80_000, 90_000, "val"), (90_000, 100_000, "test")):
    reference = None
    for key, pool in pools.items():
        directory = output / key[0] / key[1]
        beta = np.load(directory / "betas.npy")
        counts, _ = np.histogram(beta[start:stop], bins=edges)
        if reference is None:
            reference = counts
        elif not np.array_equal(counts, reference):
            raise RuntimeError(f"{split}: four-way beta-bin counts differ")
    print(f"[{split}] events/class={stop-start:,}; four-way beta-bin counts match")

summary = {
    "events_per_class_per_group": target,
    "events_per_group": 2 * target,
    "groups": list(groups),
    "beta_bin_edges": edges.tolist(),
    "beta_bin_counts": allocation.tolist(),
    "seed": seed,
}
(output / "matching_manifest.json").write_text(json.dumps(summary, indent=2))
(output / "_SUCCESS").write_text("ok\n")
print(json.dumps(summary, indent=2))
print("AOHBA STOPPING ABLATION BETA-BIN MATCHING: VALID")
PY
}

latest_run_dir()
{
    local result_root=$1
    local tag=$2
    find "$result_root" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${tag}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

run_cache_train_group()
{
    local group=$1
    local provenance="$MATCHED/$group"
    local cache="/mnt/aohba/aohba_stopping_ablation_${group}_200k_global_log"
    local result_root="$RESULT_ROOT/$group"
    local tag="aohba_stopping_ablation_${group}_200k_global_log_seed${SEED}"

    if [[ -f "$cache/_SUCCESS" && -f "$cache/cache_manifest.json" ]]; then
        echo "[SKIP] $group cache already complete: $cache"
    elif [[ -e "$cache" ]]; then
        echo "ERROR: incomplete cache exists: $cache" >&2
        exit 1
    else
        echo "[CACHE START] $group"
        python -u src/data_parse/build_aohba_fixedgrid_selected_treerec_cache.py \
            --fixedgrid-raw-dir "$provenance" \
            --output-dir "$cache" \
            --chunk-size 10000 \
            --k 8
        echo "[CACHE DONE] $group"
    fi

    mkdir -p "$result_root"
    local run_dir model checkpoint
    run_dir=$(latest_run_dir "$result_root" "$tag" || true)
    if [[ -n "$run_dir" && -f "$run_dir/evaluation_test/metrics.json" ]]; then
        echo "[SKIP] $group training and evaluation already complete: $run_dir"
        return
    fi

    checkpoint=""
    if [[ -n "$run_dir" ]]; then
        checkpoint=$(find "$run_dir" -maxdepth 1 -type f \
            -name '*_last_checkpoint.pth' -print | sort | tail -1)
    fi

    local train_args=(
        --split-cache-dir "$cache"
        --epochs 80
        --model gravnet
        --batch-size 512
        --num-workers 2
        --non-blocking-transfer
        --seed "$SEED"
        --dataset-tag "$tag"
        --result-dir "$result_root"
    )
    if [[ -n "$checkpoint" ]]; then
        echo "[RESUME] $checkpoint"
        train_args+=(--resume-checkpoint "$checkpoint")
    fi

    echo "[TRAIN START] $group on physical GPU $GPU"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_aohba.py "${train_args[@]}"

    run_dir=$(latest_run_dir "$result_root" "$tag")
    model=$(find "$run_dir" -maxdepth 1 -type f \
        -name '*_best.pth' -print | sort | tail -1)
    [[ -n "$model" ]] || {
        echo "ERROR: $group best checkpoint not found" >&2
        exit 1
    }

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$cache" \
        --model-path "$model" \
        --output-dir "$run_dir/evaluation_test" \
        --model gravnet \
        --batch-size 512 \
        --seed "$SEED"
    echo "[TRAIN DONE] $group"
    echo "evaluation: $run_dir/evaluation_test"
}

run_cache_train()
{
    activate_naka
    [[ -f "$MATCHED/_SUCCESS" ]] || {
        echo "ERROR: matched provenance is incomplete: $MATCHED" >&2
        exit 1
    }

    # Deliberately serial: a completed first group is retained if MC1 stops.
    run_cache_train_group "$GROUP_A"
    run_cache_train_group "$GROUP_B"
}

run_compare()
{
    activate_naka
    local tag_a tag_b run_a run_b out
    tag_a="aohba_stopping_ablation_${GROUP_A}_200k_global_log_seed${SEED}"
    tag_b="aohba_stopping_ablation_${GROUP_B}_200k_global_log_seed${SEED}"
    run_a=$(latest_run_dir "$RESULT_ROOT/$GROUP_A" "$tag_a")
    run_b=$(latest_run_dir "$RESULT_ROOT/$GROUP_B" "$tag_b")
    out="$RESULT_ROOT/comparison"

    for directory in "$run_a/evaluation_test" "$run_b/evaluation_test"; do
        [[ -f "$directory/labels.npy" && -f "$directory/scores.npy" ]] || {
            echo "ERROR: evaluation arrays missing: $directory" >&2
            exit 1
        }
    done

    python src/scripts/visual/compare_binary_eval.py \
        --item "$LABEL_A" "$run_a/evaluation_test" \
        --item "$LABEL_B" "$run_b/evaluation_test" \
        --out-dir "$out" \
        --x-min 0.50 \
        --y-max 20000 \
        --mark-efficiencies 0.90 0.95 0.98 0.99

    echo "figure : $out/rejection_compare.png"
    echo "metrics: $out/metrics_compare.json"
}

echo "phase                 : $PHASE"
echo "candidate events/class: $CANDIDATES_PER_CLASS"
echo "matched events/class  : $EVENTS_PER_CLASS"
echo "physical GPU          : $GPU"
echo "seed                  : $SEED"
echo "group A               : $GROUP_A ($SELECTION_A)"
echo "group B               : $GROUP_B ($SELECTION_B)"

case "$PHASE" in
    export-match) run_export_match ;;
    cache-train) run_cache_train ;;
    compare) run_compare ;;
esac

echo "AOHBA STOPPING ABLATION 200K $PHASE: COMPLETE"
