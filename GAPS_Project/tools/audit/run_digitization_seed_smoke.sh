#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 INPUT.root CRANE [OUTPUT_DIR]" >&2
    exit 2
fi

INPUT=$1
CRANE=$2
if [[ $# -eq 3 ]]; then
    OUT=$3
    if [[ -e "$OUT" ]]; then
        echo "output already exists: $OUT" >&2
        exit 2
    fi
    mkdir -p "$OUT"
else
    OUT=$(mktemp -d /tmp/digitization-seed-smoke.XXXXXX)
fi

if [[ ! -f "$INPUT" ]]; then
    echo "input missing: $INPUT" >&2
    exit 2
fi
if [[ ! -x "$CRANE" ]]; then
    echo "Crane executable missing: $CRANE" >&2
    exit 2
fi

run_one() {
    local name=$1
    local seed=$2
    "$CRANE" \
        -i "$INPUT" \
        -o "$OUT/$name.root" \
        --nevents 100 \
        --input-tree-name TreeMc \
        --do-digitization 1 \
        --do-reconstruction 0 \
        --clone-mc 1 \
        --keep-not-triggered 1 \
        --digitization-seed "$seed" \
        >"$OUT/$name.log" 2>&1
    echo "[DONE] $name seed=$seed"
}

echo "output: $OUT"
run_one same_seed_a 271828
run_one same_seed_b 271828
run_one different_seed 314159

python - "$OUT" <<'PY'
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import uproot


BASE = "Rec/hitseries_/hitseries_."


def load(path):
    with uproot.open(path) as root_file:
        tree = root_file["TreeRec"]
        volume = tree[BASE + "volume_id_"].array(library="ak")
        energy = tree[BASE + "energydep_"].array(library="ak")
        position = tree[BASE + "hit_position_"].array(library="ak")
        time = tree[BASE + "hit_time_"].array(library="ak")
    xyz = np.column_stack(
        [
            ak.to_numpy(ak.flatten(position[axis], axis=None))
            for axis in ("fX", "fY", "fZ")
        ]
    )
    return {
        "volume": volume,
        "energy": ak.to_numpy(ak.flatten(energy, axis=None)),
        "position": xyz,
        "time": ak.to_numpy(ak.flatten(time, axis=None)),
    }


def exact(left, right):
    return np.array_equal(left, right, equal_nan=True)


out = Path(sys.argv[1])
same_a = load(out / "same_seed_a.root")
same_b = load(out / "same_seed_b.root")
different = load(out / "different_seed.root")

assert ak.to_list(same_a["volume"]) == ak.to_list(same_b["volume"])
assert ak.to_list(same_a["volume"]) == ak.to_list(different["volume"])

for field in ("energy", "position", "time"):
    assert exact(same_a[field], same_b[field]), (
        f"same seed is not reproducible for {field}"
    )

assert np.array_equal(
    np.isfinite(same_a["time"]), np.isfinite(different["time"])
), "different seed changed the time finite mask"

changed = {
    field: not exact(same_a[field], different[field])
    for field in ("energy", "position", "time")
}
assert all(changed.values()), f"different seed did not change all responses: {changed}"

print("events:", len(same_a["volume"]))
print("hits:", len(same_a["energy"]))
print("same seed: EXACT")
print("different seed preserves volume structure: YES")
print("different seed changed responses:", changed)
print("DIGITIZATION SEED SMOKE: PASS")
PY
