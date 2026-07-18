#!/usr/bin/env python3
"""Export Nakagami Atrest CSV into sparse-voxel GNN input arrays.

This is for the fixed-length Atrest CSV used by Nakagami's CNN+DNN input,
not for VolID hit/step CSV.  Each event row is converted to the same directory
layout consumed by ``train_aohba_sparse_voxel_gnn.py``:

  <out>/<split>_nakagami_style_4M/
    voxels.npy       (N, 10, 12, 12)
    tof_paddles.npy  (N, 172)  -- zeros for old Atrest CSV without paddle map
    tof_primary.npy  (N, 11)
    labels.npy       (N,)
    betas.npy        (N,)

Supported layouts:
  renew_topiso:
    col 0       : random seed / file id
    col 1       : ROOT entry index
    col 2       : label (0=pbar, 1=dbar)
    col 4       : primary beta
    col 3:1443  : Si(Li) voxel edep, 1440 values -> (10, 12, 12)
    col 1620:1631: 11 TOF/global features

  topiso1457:
    col 0       : random seed / file id
    col 1       : ROOT entry index
    col 2       : label (0=pbar, 1=dbar)
    col 4       : primary beta
    col 6:1446  : Si(Li) voxel edep, 1440 values -> (10, 12, 12)
    col 1446:1457: 11 TOF/global features

  topiso1452_fig62:
    col 0       : random seed / file id
    col 1       : ROOT entry index
    col 2       : label (0=pbar, 1=dbar)
    col 3:1443  : Si(Li) voxel edep, 1440 values -> (10, 12, 12)
    col 1443:1446: 3 DNN/TOF-like features used by Nakagami Fig. 6.2
    beta is not explicitly available; betas.npy is filled with 0.

  onlyprimary:
    col 0       : file/random id
    col 1       : event id
    col 2       : label (0=pbar, 1=dbar)
    col 6:1446  : Si(Li) voxel edep
    col 1446:1618: TOF paddle edep, 172 values
    col 1618:1629: 11 TOF/global features
    beta is not explicitly available; betas.npy is filled with 0.
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


N_VOXEL = 10 * 12 * 12
N_TOF_PADDLE = 172
N_TOF_PRIMARY = 11


@dataclass(frozen=True)
class Layout:
    n_cols: int
    label_col: int
    beta_col: int | None
    si_start: int
    si_end: int
    tof_paddle_start: int | None
    tof_paddle_end: int | None
    tof_primary_start: int
    tof_primary_end: int

    @property
    def tof_primary_dim(self) -> int:
        return self.tof_primary_end - self.tof_primary_start


LAYOUTS = {
    "renew_topiso": Layout(
        n_cols=1631,
        label_col=2,
        beta_col=4,
        si_start=3,
        si_end=1443,
        tof_paddle_start=None,
        tof_paddle_end=None,
        tof_primary_start=1620,
        tof_primary_end=1631,
    ),
    "topiso1457": Layout(
        n_cols=1457,
        label_col=2,
        beta_col=4,
        si_start=6,
        si_end=1446,
        tof_paddle_start=None,
        tof_paddle_end=None,
        tof_primary_start=1446,
        tof_primary_end=1457,
    ),
    "topiso1452_fig62": Layout(
        n_cols=1452,
        label_col=2,
        beta_col=None,
        si_start=3,
        si_end=1443,
        tof_paddle_start=None,
        tof_paddle_end=None,
        tof_primary_start=1443,
        tof_primary_end=1446,
    ),
    "onlyprimary": Layout(
        n_cols=1631,
        label_col=2,
        beta_col=None,
        si_start=6,
        si_end=1446,
        tof_paddle_start=1446,
        tof_paddle_end=1618,
        tof_primary_start=1618,
        tof_primary_end=1629,
    ),
}


def infer_particle_label(path: Path) -> int | None:
    name = path.name.lower()
    if "pbar" in name or "antip" in name:
        return 0
    if "dbar" in name or "antid" in name:
        return 1
    return None


def list_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob(args.glob)))
        elif p.is_file():
            files.append(p)
        else:
            files.extend(Path(x) for x in sorted(globlib.glob(item)))

    out = []
    seen = set()
    for p in files:
        if not p.name.endswith(".csv"):
            continue
        if args.exclude_volid and "VolID" in p.name:
            continue
        if args.exclude_inflight and "Inflight" in p.name:
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def count_rows(files: Iterable[Path], layout: Layout) -> tuple[int, int, dict[int, int]]:
    total = 0
    bad = 0
    labels = {0: 0, 1: 0}
    for fp in tqdm(list(files), desc="count rows", dynamic_ncols=True):
        fallback_label = infer_particle_label(fp)
        with fp.open() as f:
            for row in csv.reader(f):
                if len(row) != layout.n_cols:
                    bad += 1
                    continue
                try:
                    label = int(float(row[layout.label_col]))
                except Exception:
                    if fallback_label is None:
                        bad += 1
                        continue
                    label = fallback_label
                if label not in (0, 1):
                    if fallback_label is None:
                        bad += 1
                        continue
                    label = fallback_label
                labels[label] += 1
                total += 1
    return total, bad, labels


def collect_refs(files: list[Path], layout: Layout, seed: int, events_per_class: int | None):
    """Collect (path, row_index) references, optionally balanced by class."""
    rng = random.Random(seed)
    refs_by_label = {0: [], 1: []}
    bad = 0

    shuffled = list(files)
    rng.shuffle(shuffled)
    for fp in tqdm(shuffled, desc="collect refs", dynamic_ncols=True):
        fallback_label = infer_particle_label(fp)
        if (
            events_per_class is not None
            and fallback_label in (0, 1)
            and len(refs_by_label[fallback_label]) >= events_per_class
        ):
            continue
        with fp.open() as f:
            for row_idx, row in enumerate(csv.reader(f)):
                if len(row) != layout.n_cols:
                    bad += 1
                    continue
                try:
                    label = int(float(row[layout.label_col]))
                except Exception:
                    label = fallback_label
                if label not in (0, 1):
                    label = fallback_label
                if label not in (0, 1):
                    bad += 1
                    continue
                if events_per_class is None or len(refs_by_label[label]) < events_per_class:
                    refs_by_label[label].append((str(fp), row_idx, label))
                elif events_per_class is not None and fallback_label in (0, 1):
                    break
        if events_per_class is not None and all(
            len(refs_by_label[k]) >= events_per_class for k in (0, 1)
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
    return refs, bad, {k: len(v) for k, v in refs_by_label.items()}


def split_refs(refs, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for ref in refs:
        by_label[ref[2]].append(ref)
    for v in by_label.values():
        rng.shuffle(v)

    splits = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for items in splits.values():
        rng.shuffle(items)
    return splits


def row_to_arrays(row: list[str], layout: Layout):
    label = int(float(row[layout.label_col]))
    beta = 0.0 if layout.beta_col is None else float(row[layout.beta_col])

    si = np.asarray(row[layout.si_start : layout.si_end], dtype=np.float32)
    if si.size != N_VOXEL:
        raise ValueError(f"bad Si size: {si.size}")
    voxel = si.reshape(10, 12, 12)

    tof_paddle = np.zeros((N_TOF_PADDLE,), dtype=np.float32)
    if layout.tof_paddle_start is not None and layout.tof_paddle_end is not None:
        tof_paddle = np.asarray(
            row[layout.tof_paddle_start : layout.tof_paddle_end],
            dtype=np.float32,
        )
        if tof_paddle.size != N_TOF_PADDLE:
            raise ValueError(f"bad TOF paddle size: {tof_paddle.size}")

    tof_primary = np.asarray(
        row[layout.tof_primary_start : layout.tof_primary_end],
        dtype=np.float32,
    )
    if tof_primary.size != layout.tof_primary_dim:
        raise ValueError(f"bad TOF primary size: {tof_primary.size}")

    return voxel, tof_paddle, tof_primary, label, beta


def export_split(name: str, refs, out_dir: Path, layout: Layout):
    split_dir = out_dir / f"{name}_nakagami_style_4M"
    split_dir.mkdir(parents=True, exist_ok=True)
    refs_sorted = sorted(refs, key=lambda x: (x[0], x[1]))
    n = len(refs_sorted)

    voxels = np.lib.format.open_memmap(
        split_dir / "voxels.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, 10, 12, 12),
    )
    tof_paddles = np.lib.format.open_memmap(
        split_dir / "tof_paddles.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, N_TOF_PADDLE),
    )
    tof_primary = np.lib.format.open_memmap(
        split_dir / "tof_primary.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, layout.tof_primary_dim),
    )
    labels = np.lib.format.open_memmap(
        split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    betas = np.lib.format.open_memmap(
        split_dir / "betas.npy", mode="w+", dtype=np.float32, shape=(n,)
    )

    by_path: dict[str, list[tuple[int, int, int]]] = {}
    for out_idx, (path, row_idx, label) in enumerate(refs_sorted):
        by_path.setdefault(path, []).append((row_idx, out_idx, label))

    bad = 0
    for path, targets in tqdm(by_path.items(), desc=f"export {name}", dynamic_ncols=True):
        target_map = {row_idx: out_idx for row_idx, out_idx, _ in targets}
        with Path(path).open() as f:
            for row_idx, row in enumerate(csv.reader(f)):
                out_idx = target_map.get(row_idx)
                if out_idx is None:
                    continue
                try:
                    voxel, tof_paddle, tof_primary_arr, label, beta = row_to_arrays(row, layout)
                except Exception:
                    bad += 1
                    continue
                voxels[out_idx] = voxel
                tof_paddles[out_idx] = tof_paddle
                tof_primary[out_idx] = tof_primary_arr
                labels[out_idx] = label
                betas[out_idx] = beta

    for arr in (voxels, tof_paddles, tof_primary, labels, betas):
        arr.flush()

    label_counts = {
        "0": int((np.asarray(labels) == 0).sum()),
        "1": int((np.asarray(labels) == 1).sum()),
    }
    summary = {
        "split": name,
        "n_events": n,
        "label_counts": label_counts,
        "voxel_shape": [10, 12, 12],
        "tof_paddles": N_TOF_PADDLE,
        "tof_primary": layout.tof_primary_dim,
        "bad_rows_during_export": bad,
    }
    with (split_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("saved", split_dir, summary, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files, directories, or glob patterns.",
    )
    ap.add_argument("--glob", default="CNN*Atrest*.csv")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="renew_topiso")
    ap.add_argument("--events-per-class", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-volid", action="store_true", default=True)
    ap.add_argument("--exclude-inflight", action="store_true", default=True)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    layout = LAYOUTS[args.layout]
    files = list_files(args)
    if not files:
        raise FileNotFoundError("no CSV files matched")

    print("layout:", args.layout)
    print("files:", len(files))
    print("first:", files[0])

    if args.count_only:
        total, bad, labels = count_rows(files, layout)
        print({"rows": total, "bad": bad, "labels": labels})
        return

    refs, bad, label_counts = collect_refs(
        files, layout, args.seed, args.events_per_class
    )
    print("collected refs:", len(refs), "bad:", bad, "labels:", label_counts)
    splits = split_refs(refs, args.train_frac, args.val_frac, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "Nakagami fixed-length Atrest CSV",
        "layout": args.layout,
        "inputs": [str(p) for p in files],
        "events_per_class": args.events_per_class,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "splits": {k: len(v) for k, v in splits.items()},
        "note": "This is not VolID hit/step CSV; Si(Li) fixed grid is converted to sparse voxel nodes by the training dataset.",
    }
    with (args.output_dir / "export_manifest.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for split, split_refs_ in splits.items():
        export_split(split, split_refs_, args.output_dir, layout)

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
