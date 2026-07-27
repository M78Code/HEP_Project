#!/usr/bin/env python3
"""Export Nakagami 1457-column CSV files to fixed-grid voxel arrays.

This script is the clean reproduction entry point for the Nakagami-data 4M
experiments used in the thesis.  It supports the 1457-column CSV layout:

  row[2]       : label, 0=anti-proton, 1=anti-deuteron
  row[4]       : generated primary beta
  row[6:1446]  : 1440 Si(Li) energy values, reshaped to (10, 12, 12)
  row[1446:1457]: 11 auxiliary / TOF-like features

The output layout is shared by the CNN+DNN and sparse-voxel GNN scripts:

  <output>/<split>_nakagami_style_4M/
    voxels.npy       (N, 10, 12, 12)
    tof_primary.npy  (N, 11)
    labels.npy       (N,)
    betas.npy        (N,)
    summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


N_COLS = 1457
N_VOXEL = 10 * 12 * 12
N_TOF_PRIMARY = 11

LABEL_COL = 2
BETA_COL = 4
SI_START = 6
SI_END = 1446
TOF_PRIMARY_START = 1446
TOF_PRIMARY_END = 1457


def infer_particle_label(path: Path) -> int | None:
    name = path.name.lower()
    if "pbar" in name or "antip" in name:
        return 0
    if "dbar" in name or "antid" in name:
        return 1
    return None


def list_csv_files(items: Iterable[str], pattern: str, exclude_inflight: bool = True) -> list[Path]:
    files: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            files.append(p)
        else:
            files.extend(Path(x) for x in sorted(glob.glob(item)))

    out: list[Path] = []
    seen: set[str] = set()
    for p in files:
        if p.suffix.lower() != ".csv":
            continue
        if exclude_inflight and "Inflight" in p.name:
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def parse_label(row: list[str], fallback: int | None) -> int | None:
    try:
        label = int(float(row[LABEL_COL]))
    except Exception:
        label = fallback
    if label not in (0, 1):
        label = fallback
    return label if label in (0, 1) else None


def collect_refs(
    files: list[Path],
    events_per_class: int | None,
    seed: int,
) -> tuple[list[tuple[str, int, int]], int, dict[int, int]]:
    """Collect row references, optionally balanced by class."""
    rng = random.Random(seed)
    refs_by_label: dict[int, list[tuple[str, int, int]]] = {0: [], 1: []}
    bad = 0

    shuffled = list(files)
    rng.shuffle(shuffled)
    for fp in tqdm(shuffled, desc="collect refs", dynamic_ncols=True):
        fallback = infer_particle_label(fp)
        if (
            events_per_class is not None
            and fallback in (0, 1)
            and len(refs_by_label[fallback]) >= events_per_class
        ):
            continue

        with fp.open() as f:
            for row_idx, row in enumerate(csv.reader(f)):
                if len(row) != N_COLS:
                    bad += 1
                    continue
                label = parse_label(row, fallback)
                if label is None:
                    bad += 1
                    continue
                if events_per_class is None or len(refs_by_label[label]) < events_per_class:
                    refs_by_label[label].append((str(fp), row_idx, label))
                elif fallback in (0, 1):
                    break

        if events_per_class is not None and all(
            len(refs_by_label[label]) >= events_per_class for label in (0, 1)
        ):
            break

    if events_per_class is not None:
        for label in (0, 1):
            if len(refs_by_label[label]) < events_per_class:
                raise RuntimeError(
                    f"not enough label={label} events: "
                    f"{len(refs_by_label[label])} < {events_per_class}"
                )

    refs = refs_by_label[0] + refs_by_label[1]
    rng.shuffle(refs)
    return refs, bad, {label: len(items) for label, items in refs_by_label.items()}


def split_refs(
    refs: list[tuple[str, int, int]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[tuple[str, int, int]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[tuple[str, int, int]]] = {0: [], 1: []}
    for ref in refs:
        by_label[ref[2]].append(ref)
    for items in by_label.values():
        rng.shuffle(items)

    splits: dict[str, list[tuple[str, int, int]]] = {"train": [], "val": [], "test": []}
    for items in by_label.values():
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for items in splits.values():
        rng.shuffle(items)
    return splits


def row_to_arrays(row: list[str]) -> tuple[np.ndarray, np.ndarray, int, float]:
    label = int(float(row[LABEL_COL]))
    beta = float(row[BETA_COL])

    si = np.asarray(row[SI_START:SI_END], dtype=np.float32)
    if si.size != N_VOXEL:
        raise ValueError(f"bad Si(Li) voxel size: {si.size}")
    voxel = si.reshape(10, 12, 12)

    tof_primary = np.asarray(row[TOF_PRIMARY_START:TOF_PRIMARY_END], dtype=np.float32)
    if tof_primary.size != N_TOF_PRIMARY:
        raise ValueError(f"bad tof_primary size: {tof_primary.size}")

    return voxel, tof_primary, label, beta


def export_split(split: str, refs: list[tuple[str, int, int]], output_dir: Path) -> None:
    split_dir = output_dir / f"{split}_nakagami_style_4M"
    split_dir.mkdir(parents=True, exist_ok=True)

    refs_sorted = sorted(refs, key=lambda x: (x[0], x[1]))
    n = len(refs_sorted)

    voxels = np.lib.format.open_memmap(
        split_dir / "voxels.npy", mode="w+", dtype=np.float32, shape=(n, 10, 12, 12)
    )
    tof_primary = np.lib.format.open_memmap(
        split_dir / "tof_primary.npy", mode="w+", dtype=np.float32, shape=(n, N_TOF_PRIMARY)
    )
    labels = np.lib.format.open_memmap(
        split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    betas = np.lib.format.open_memmap(
        split_dir / "betas.npy", mode="w+", dtype=np.float32, shape=(n,)
    )

    by_path: dict[str, dict[int, int]] = {}
    for out_idx, (path, row_idx, _) in enumerate(refs_sorted):
        by_path.setdefault(path, {})[row_idx] = out_idx

    bad = 0
    for path, row_map in tqdm(by_path.items(), desc=f"export {split}", dynamic_ncols=True):
        with Path(path).open() as f:
            for row_idx, row in enumerate(csv.reader(f)):
                out_idx = row_map.get(row_idx)
                if out_idx is None:
                    continue
                try:
                    voxel, tof_primary_arr, label, beta = row_to_arrays(row)
                except Exception:
                    bad += 1
                    continue
                voxels[out_idx] = voxel
                tof_primary[out_idx] = tof_primary_arr
                labels[out_idx] = label
                betas[out_idx] = beta

    for arr in (voxels, tof_primary, labels, betas):
        arr.flush()

    label_counts = {
        "0": int((np.asarray(labels) == 0).sum()),
        "1": int((np.asarray(labels) == 1).sum()),
    }
    summary = {
        "split": split,
        "n_events": n,
        "label_counts": label_counts,
        "voxel_shape": [10, 12, 12],
        "tof_primary": N_TOF_PRIMARY,
        "bad_rows_during_export": bad,
    }
    with (split_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("saved", split_dir, summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV files, directories, or glob patterns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="CNN*Atrest*.csv")
    parser.add_argument("--events-per-class", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    files = list_csv_files(args.inputs, pattern=args.glob)
    if not files:
        raise FileNotFoundError("no CSV files matched")

    print("layout: topiso1457")
    print("files:", len(files))
    print("first:", files[0])

    refs, bad_total, label_counts = collect_refs(files, args.events_per_class, args.seed)
    print("collected refs:", len(refs), "bad:", bad_total, "labels:", label_counts, flush=True)
    if args.count_only:
        return

    splits = split_refs(refs, args.train_frac, args.val_frac, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "Nakagami 1457-column fixed-grid CSV",
        "layout": "topiso1457",
        "inputs": [str(p) for p in files],
        "events_per_class": args.events_per_class,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "splits": {name: len(items) for name, items in splits.items()},
        "label_counts_collected": label_counts,
        "bad_total": bad_total,
        "columns": {
            "label": LABEL_COL,
            "beta": BETA_COL,
            "sili_voxel": [SI_START, SI_END],
            "tof_primary": [TOF_PRIMARY_START, TOF_PRIMARY_END],
        },
        "note": "Clean export does not create tof_paddles.npy for Nakagami CSV data.",
    }
    with (args.output_dir / "export_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for split, split_refs_ in splits.items():
        export_split(split, split_refs_, args.output_dir)

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
