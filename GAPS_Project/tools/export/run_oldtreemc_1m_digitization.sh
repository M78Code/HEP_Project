#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-all}
if [[ "$PHASE" != "select" && "$PHASE" != "export" && \
      "$PHASE" != "digitize" && "$PHASE" != "all" ]]; then
    echo "usage: $0 [select|export|digitize|all]" >&2
    exit 2
fi

PROJECT=${PROJECT:-$HOME/HEP_Project/GAPS_Project}
CSVROOT=${CSVROOT:-/mnt/ynakagami2/SimulationData/210713_renew_topIso_flat/csvFiles/atrest_shuffled}
ROOTDIR=${ROOTDIR:-/mnt/ynakagami2/SimulationData/210713_renew_topIso_flat/rootFiles}
SELECTION=${SELECTION:-/mnt/aohba/oldtreemc_source_disjoint_1m_selection}
SKIMS=${SKIMS:-/mnt/aohba/oldtreemc_source_disjoint_1m_skims}
DIGITIZED=${DIGITIZED:-/mnt/aohba/oldtreemc_source_disjoint_1m_digitized}
EXPORT_LOGDIR=${EXPORT_LOGDIR:-$HOME/oldtreemc_1m_export_logs}
DIGITIZE_LOGDIR=${DIGITIZE_LOGDIR:-$HOME/oldtreemc_1m_digitize_logs}
EXPORT_JOBS=${EXPORT_JOBS:-2}
DIGITIZE_JOBS=${DIGITIZE_JOBS:-4}
EVENTS_PER_SHARD=${EVENTS_PER_SHARD:-50000}

MYSIM=${MYSIM:-$HOME/simpledet_treerec_fix}
MYBUILD=${MYBUILD:-$MYSIM/build}
CRANE=${CRANE:-$MYBUILD/analysis/CraneBaseProcessing}
EXPORTER=${EXPORTER:-$PROJECT/build/tools/skim_treemc_entries}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
source "$HOME/setup_ynakagami_root.sh"
export GAPS=$MYSIM/runtime
export LD_LIBRARY_PATH="$MYBUILD/analysis:$MYBUILD/reconstruction:$MYBUILD/common:$ROOTSYS/lib:/home/ynakagami/GEANT/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so

cd "$PROJECT"

run_selection()
{
    if [[ -f "$SELECTION/selection_manifest.json" ]]; then
        echo "[SKIP] existing selection: $SELECTION/selection_manifest.json"
        return
    fi
    if [[ -e "$SELECTION" ]]; then
        echo "ERROR: selection directory exists without a manifest: $SELECTION" >&2
        exit 1
    fi

    python src/scripts/select_oldtreemc_source_disjoint_dataset.py \
        --csv-dir "train=$CSVROOT/train_4M" \
        --csv-dir "val=$CSVROOT/valid_4M" \
        --csv-dir "test=$CSVROOT/valid_4M" \
        --root-dir "$ROOTDIR" \
        --output-dir "$SELECTION" \
        --purpose "old-domain source-disjoint 1M digitized TreeRec" \
        --selection train:antip:1627528606:0:50000 \
        --selection train:antip:1627528610:0:50000 \
        --selection train:antip:1627528613:0:50000 \
        --selection train:antip:1627528617:0:50000 \
        --selection train:antip:1627528621:0:50000 \
        --selection train:antip:1627528704:0:50000 \
        --selection train:antip:1627528707:0:50000 \
        --selection train:antip:1627528710:0:50000 \
        --selection train:antid:1627550259:1:50000 \
        --selection train:antid:1627550262:1:50000 \
        --selection train:antid:1627550265:1:50000 \
        --selection train:antid:1627550269:1:50000 \
        --selection train:antid:1627550272:1:50000 \
        --selection train:antid:1627550275:1:50000 \
        --selection train:antid:1627550279:1:50000 \
        --selection train:antid:1627550282:1:50000 \
        --selection val:antip:1627528718:0:50000 \
        --selection val:antid:1627550289:1:50000 \
        --selection test:antip:1627528714:0:50000 \
        --selection test:antid:1627550286:1:50000
}

run_export()
{
    [[ -f "$SELECTION/selection_manifest.json" ]] || {
        echo "ERROR: selection manifest is missing" >&2
        exit 1
    }
    [[ -x "$EXPORTER" ]] || {
        echo "ERROR: skim exporter is missing: $EXPORTER" >&2
        exit 1
    }
    mkdir -p "$EXPORT_LOGDIR"
    python tools/export/export_treemc_selection_manifest.py \
        --selection-manifest "$SELECTION/selection_manifest.json" \
        --exporter "$EXPORTER" \
        --output-dir "$SKIMS" \
        --log-dir "$EXPORT_LOGDIR" \
        --events-per-shard "$EVENTS_PER_SHARD" \
        --jobs "$EXPORT_JOBS"
}

run_digitization()
{
    [[ -f "$SKIMS/export_manifest.json" ]] || {
        echo "ERROR: skim export manifest is missing" >&2
        exit 1
    }
    [[ -x "$CRANE" ]] || {
        echo "ERROR: Crane executable is missing: $CRANE" >&2
        exit 1
    }
    mkdir -p "$DIGITIZE_LOGDIR"
    python tools/export/digitize_treemc_skims.py \
        --input-dir "$SKIMS" \
        --output-dir "$DIGITIZED" \
        --log-dir "$DIGITIZE_LOGDIR" \
        --crane "$CRANE" \
        --jobs "$DIGITIZE_JOBS" \
        --expected-events 1000000
}

echo "phase        : $PHASE"
echo "selection    : $SELECTION"
echo "skims        : $SKIMS"
echo "digitized    : $DIGITIZED"
echo "export jobs  : $EXPORT_JOBS"
echo "digitize jobs: $DIGITIZE_JOBS"
df -h /mnt/aohba

case "$PHASE" in
    select)
        run_selection
        ;;
    export)
        run_export
        ;;
    digitize)
        run_digitization
        ;;
    all)
        run_selection
        run_export
        run_digitization
        ;;
esac

echo "OLD TREEMC SOURCE-DISJOINT 1M $PHASE: COMPLETE"
