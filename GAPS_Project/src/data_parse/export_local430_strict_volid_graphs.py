#!/usr/bin/env python3
"""Convert local430 strict VolID CSV files to PyG graph caches.

The input CSV is produced by export_local430_strict_volid_csv.cc and keeps the
same 10-column VolID layout used in Nakagami's old CSV:

  file_id,entry,label,stoplayer,volume_id,x,y,z,tof,hit_edep

Unlike export_nakagami_volid_graphs.py, this converter does not require an
Atrest metadata CSV. Labels are read directly from the VolID rows. Beta is not
available in this CSV format, so a dummy beta=-1 is stored until a separate
event-level metadata export is added.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


HitRow = tuple[int, int, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anti-p-csv", required=True, type=Path)
    p.add_argument("--anti-d-csv", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=0, help="cap events per class; 0 keeps all")
    p.add_argument("--train-per-class", type=int, default=0)
    p.add_argument("--val-per-class", type=int, default=0)
    p.add_argument("--test-per-class", type=int, default=0)
    p.add_argument("--split-fracs", default="0.8,0.1,0.1", help="train,val,test fractions")
    return p.parse_args()


def load_hits(csv_path: Path, expected_label: int) -> dict[tuple[str, str], list[HitRow]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    hits: dict[tuple[str, str], list[HitRow]] = defaultdict(list)
    mismatched = 0
    with csv_path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 10:
                raise ValueError(f"{csv_path}: expected 10 columns, got {len(row)}: {row}")

            label = int(float(row[2]))
            if label != expected_label:
                mismatched += 1
                continue

            key = (row[0], row[1])
            stoplayer = int(float(row[3]))
            volume_id = int(float(row[4]))
            x = float(row[5])
            y = float(row[6])
            z = float(row[7])
            tof = float(row[8])
            hit_edep = float(row[9])
            hits[key].append((stoplayer, volume_id, x, y, z, tof, hit_edep))

    if mismatched:
        print(f"warning: skipped {mismatched:,} rows with labels not equal to {expected_label} in {csv_path}")
    return hits


def graph_from_rows(key: tuple[str, str], label: int, rows: list[HitRow], knn_k: int) -> Data | None:
    if len(rows) < 2:
        return None

    stoplayer = torch.tensor([r[0] for r in rows], dtype=torch.float32).view(-1, 1)
    volume_id = torch.tensor([r[1] for r in rows], dtype=torch.long)
    pos = torch.tensor([[r[2], r[3], r[4]] for r in rows], dtype=torch.float32) / 1000.0
    tof = torch.tensor([r[5] for r in rows], dtype=torch.float32).view(-1, 1)
    edep = torch.tensor([r[6] for r in rows], dtype=torch.float32).view(-1, 1)

    x = torch.cat(
        [
            torch.log1p(torch.clamp(edep, min=0.0)),
            tof / 50.0,
            stoplayer / 10.0,
            volume_id.to(torch.float32).view(-1, 1) / 1.0e8,
        ],
        dim=1,
    )

    k_eff = min(knn_k, pos.size(0) - 1)
    edge_index = knn_graph(pos, k=k_eff, loop=False)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        volume_id=volume_id,
        y=torch.tensor(label, dtype=torch.long),
        beta=torch.tensor(-1.0, dtype=torch.float32),
        file_id=key[0],
        event_id=key[1],
    )


def build_graphs(csv_path: Path, label: int, knn_k: int, max_events: int) -> list[Data]:
    grouped = load_hits(csv_path, expected_label=label)
    graphs: list[Data] = []
    too_small = 0

    for key in sorted(grouped):
        graph = graph_from_rows(key, label, grouped[key], knn_k)
        if graph is None:
            too_small += 1
            continue
        graphs.append(graph)
        if max_events > 0 and len(graphs) >= max_events:
            break

    print(
        f"{csv_path.name}: events={len(grouped):,} graphs={len(graphs):,} "
        f"too_small={too_small:,} label={label}"
    )
    return graphs


def resolve_split_counts(args: argparse.Namespace, n_per_class: int) -> tuple[int, int, int]:
    explicit = [args.train_per_class, args.val_per_class, args.test_per_class]
    if any(v > 0 for v in explicit):
        if not all(v > 0 for v in explicit):
            raise ValueError("train/val/test per-class counts must all be provided together")
        need = sum(explicit)
        if need > n_per_class:
            raise ValueError(f"requested {need} per class, but only {n_per_class} available")
        return tuple(explicit)  # type: ignore[return-value]

    fracs = [float(x.strip()) for x in args.split_fracs.split(",")]
    if len(fracs) != 3:
        raise ValueError("--split-fracs must contain train,val,test fractions")
    if not 0.999 <= sum(fracs) <= 1.001:
        raise ValueError("--split-fracs must sum to 1")

    train = int(n_per_class * fracs[0])
    val = int(n_per_class * fracs[1])
    test = n_per_class - train - val
    return train, val, test


def split_balanced(args: argparse.Namespace, anti_p: list[Data], anti_d: list[Data]) -> dict[str, list[Data]]:
    rng = random.Random(args.seed)
    rng.shuffle(anti_p)
    rng.shuffle(anti_d)

    n_per_class = min(len(anti_p), len(anti_d))
    train_n, val_n, test_n = resolve_split_counts(args, n_per_class)

    def cut(xs: list[Data]) -> tuple[list[Data], list[Data], list[Data]]:
        train = xs[:train_n]
        val = xs[train_n : train_n + val_n]
        test = xs[train_n + val_n : train_n + val_n + test_n]
        return train, val, test

    p_train, p_val, p_test = cut(anti_p)
    d_train, d_val, d_test = cut(anti_d)

    splits = {
        "train": p_train + d_train,
        "val": p_val + d_val,
        "test": p_test + d_test,
    }

    for name, graphs in splits.items():
        rng.shuffle(graphs)
        labels = torch.tensor([int(g.y) for g in graphs], dtype=torch.long)
        print(f"{name}: {len(graphs):,} labels={torch.bincount(labels, minlength=2).tolist()}")

    return splits


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anti_p = build_graphs(args.anti_p_csv, label=0, knn_k=args.knn_k, max_events=args.max_per_class)
    anti_d = build_graphs(args.anti_d_csv, label=1, knn_k=args.knn_k, max_events=args.max_per_class)

    splits = split_balanced(args, anti_p=anti_p, anti_d=anti_d)
    for name, graphs in splits.items():
        out_path = args.output_dir / f"{name}.pt"
        torch.save(graphs, out_path)
        print(f"saved {out_path}")

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
