#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel, build_tof_features
from GAPS_Project.src.data_parse.tof_paddles import (
    build_tof_paddle_energy,
    load_paddle_ids,
    make_paddle_index,
)

def load_events(path: Path):
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj["events"] if isinstance(obj, dict) else obj

def collect_refs(pkl_base: Path, particle: str, n: int, rng: random.Random):
    files = sorted((pkl_base / particle).glob("*.pkl"))
    rng.shuffle(files)
    refs = []
    for p in files:
        events = load_events(p)
        idxs = list(range(len(events)))
        rng.shuffle(idxs)
        for i in idxs:
            refs.append((str(p), i))
            if len(refs) >= n:
                return refs
    raise RuntimeError(f"not enough events for {particle}: got {len(refs)}, need {n}")

def split_refs(refs, train_frac, val_frac):
    n = len(refs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return refs[:n_train], refs[n_train:n_train+n_val], refs[n_train+n_val:]

def collect_paddle_ids(refs):
    ids = set()
    cache = {}
    for path, idx in tqdm(refs, desc="scan TOF paddles"):
        p = Path(path)
        if p not in cache:
            cache.clear()
            cache[p] = load_events(p)
        ev = cache[p][idx]
        vids = np.asarray(ev["volume_id"], dtype=np.int64)
        tof = vids[(vids // 100_000_000) == 1]
        ids.update(int(tof_paddle_id(v)) for v in tof)
    ids = sorted(ids)
    if len(ids) != 172:
        raise RuntimeError(f"expected 172 TOF paddles, got {len(ids)}")
    return ids

def export_split(name, refs, out_dir, paddle_index, grid_x, grid_y):
    split_dir = out_dir / f"{name}_nakagami_style_4M"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(refs)
    voxels = np.lib.format.open_memmap(split_dir / "voxels.npy", mode="w+", dtype=np.float32, shape=(n, 10, grid_x, grid_y))
    tof_paddles = np.lib.format.open_memmap(split_dir / "tof_paddles.npy", mode="w+", dtype=np.float32, shape=(n, 172))
    tof_primary = np.lib.format.open_memmap(split_dir / "tof_primary.npy", mode="w+", dtype=np.float32, shape=(n, 11))
    labels = np.lib.format.open_memmap(split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(n,))
    betas = np.lib.format.open_memmap(split_dir / "betas.npy", mode="w+", dtype=np.float32, shape=(n,))

    cache = {}
    label_counts = {0: 0, 1: 0}

    for j, (path, idx) in enumerate(tqdm(refs, desc=f"export {name}")):
        p = Path(path)
        if p not in cache:
            cache.clear()
            cache[p] = load_events(p)
        ev = cache[p][idx]

        label = int(ev["label"])
        if label not in (0, 1):
            label = 1 if label < 0 and abs(label) != 2212 else 0

        voxels[j] = build_sili_voxel(ev, grid_x=grid_x, grid_y=grid_y)
        tof_paddles[j] = build_tof_paddle_energy(ev["energy"], ev["volume_id"], paddle_index)
        tof_primary[j] = build_tof_features(ev)
        labels[j] = label
        betas[j] = float(ev.get("beta", 0.0))
        label_counts[label] += 1

    for a in [voxels, tof_paddles, tof_primary, labels, betas]:
        a.flush()

    summary = {
        "split": name,
        "n_events": n,
        "label_counts": label_counts,
        "voxel_shape": [10, grid_x, grid_y],
        "tof_paddles": 172,
        "tof_primary": 11,
    }
    with (split_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("saved", split_dir, summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-base-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--events-per-class", type=int, required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--grid-x", type=int, default=12)
    ap.add_argument("--grid-y", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--paddle-ids", type=Path, default=Path("dataset/aohba_atrest_tof172_sharded/paddle_ids.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    refs0 = collect_refs(args.pkl_base_dir, "antiP", args.events_per_class, rng)
    refs1 = collect_refs(args.pkl_base_dir, "antiD", args.events_per_class, rng)

    train0, val0, test0 = split_refs(refs0, args.train_frac, args.val_frac)
    train1, val1, test1 = split_refs(refs1, args.train_frac, args.val_frac)

    train = train0 + train1
    val = val0 + val1
    test = test0 + test1
    # Keep refs grouped by pkl path for fast sequential loading.
    # Shuffling across events is unnecessary for exported arrays; training loader shuffles later.
    train = sorted(train)
    val = sorted(val)
    test = sorted(test)

    paddle_ids = load_paddle_ids(args.paddle_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paddle_ids.json").write_text(args.paddle_ids.read_text())
    paddle_index = make_paddle_index(paddle_ids)
    print(f"loaded paddle mapping: {args.paddle_ids} ({len(paddle_ids)} paddles)", flush=True)

    export_split("train", train, args.output_dir, paddle_index, args.grid_x, args.grid_y)
    export_split("val", val, args.output_dir, paddle_index, args.grid_x, args.grid_y)
    export_split("test", test, args.output_dir, paddle_index, args.grid_x, args.grid_y)

    print("done:", args.output_dir)

if __name__ == "__main__":
    main()
