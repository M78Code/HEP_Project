#!/usr/bin/env python3
"""Export Nakagami old VolID CSV files as PyG graph caches.

Input CSV layout confirmed from ExtractDataforNN.cc:
VolID rows:
  file_id,event_id,event_category,stoplayer,volume_id,x,y,z,tof,hit_edep

Atrest rows:
  file_id,event_id,particle_label,event_category,beta,...

This script uses Atrest CSV for antiP/antiD labels and beta, and VolID CSV
for hit-level graph nodes.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--file-id", default="000")
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=50000)
    p.add_argument("--train-per-class", type=int, default=40000)
    p.add_argument("--val-per-class", type=int, default=5000)
    p.add_argument("--test-per-class", type=int, default=5000)
    return p.parse_args()


def csv_name(particle: str, kind: str, file_id: str) -> str:
    return f"CNN211118_{particle}_isot_50K_beta02to05_{kind}_{file_id}.csv"


def load_atrest(atrest_path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    events: dict[tuple[str, str], dict[str, float | int]] = {}
    with atrest_path.open() as f:
        for row in csv.reader(f):
            key = (row[0], row[1])
            events[key] = {
                "label": int(float(row[2])),
                "beta": float(row[4]),
            }
    return events


def load_hits(
    volid_path: Path,
    valid_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], list[tuple[int, int, float, float, float, float, float]]]:
    hits: dict[tuple[str, str], list[tuple[int, int, float, float, float, float, float]]] = defaultdict(list)
    with volid_path.open() as f:
        for row in csv.reader(f):
            key = (row[0], row[1])
            if key not in valid_keys:
                continue
            stoplayer = int(float(row[3]))
            volume_id = int(float(row[4]))
            x = float(row[5])
            y = float(row[6])
            z = float(row[7])
            tof = float(row[8])
            hit_edep = float(row[9])
            hits[key].append((stoplayer, volume_id, x, y, z, tof, hit_edep))
    return hits


def graph_from_event(
    key: tuple[str, str],
    meta: dict[str, float | int],
    rows: list[tuple[int, int, float, float, float, float, float]],
    knn_k: int,
) -> Data | None:
    if len(rows) < 2:
        return None

    stoplayer = torch.tensor([r[0] for r in rows], dtype=torch.float32).view(-1, 1)
    volume_id = torch.tensor([r[1] for r in rows], dtype=torch.long)
    pos = torch.tensor([[r[2], r[3], r[4]] for r in rows], dtype=torch.float32) / 1000.0
    tof = torch.tensor([r[5] for r in rows], dtype=torch.float32).view(-1, 1)
    edep = torch.tensor([r[6] for r in rows], dtype=torch.float32).view(-1, 1)

    # Lightweight first-pass features. Keep volume_id separately for later embedding work.
    x = torch.cat(
        [
            torch.log1p(edep),
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
        y=torch.tensor(int(meta["label"]), dtype=torch.long),
        beta=torch.tensor(float(meta["beta"]), dtype=torch.float32),
        file_id=key[0],
        event_id=key[1],
    )


def build_graphs(args: argparse.Namespace, particle: str) -> list[Data]:
    atrest_path = args.csv_dir / csv_name(particle, "Atrest", args.file_id)
    volid_path = args.csv_dir / csv_name(particle, "VolID", args.file_id)

    if not atrest_path.exists():
        raise FileNotFoundError(atrest_path)
    if not volid_path.exists():
        raise FileNotFoundError(volid_path)

    meta = load_atrest(atrest_path)
    hits = load_hits(volid_path, set(meta))

    graphs: list[Data] = []
    missing = 0
    for key in sorted(meta):
        rows = hits.get(key)
        if not rows:
            missing += 1
            continue
        graph = graph_from_event(key, meta[key], rows, args.knn_k)
        if graph is not None:
            graphs.append(graph)
        if len(graphs) >= args.max_per_class:
            break

    print(f"{particle}: graphs={len(graphs):,} missing_hits={missing:,}")
    return graphs


def split_balanced(args: argparse.Namespace, anti_p: list[Data], anti_d: list[Data]) -> dict[str, list[Data]]:
    rng = random.Random(args.seed)
    rng.shuffle(anti_p)
    rng.shuffle(anti_d)

    need = args.train_per_class + args.val_per_class + args.test_per_class
    if len(anti_p) < need or len(anti_d) < need:
        raise RuntimeError(f"not enough graphs per class: antiP={len(anti_p)}, antiD={len(anti_d)}, need={need}")

    def cut(xs: list[Data]) -> tuple[list[Data], list[Data], list[Data]]:
        train = xs[: args.train_per_class]
        val = xs[args.train_per_class : args.train_per_class + args.val_per_class]
        test = xs[
            args.train_per_class + args.val_per_class :
            args.train_per_class + args.val_per_class + args.test_per_class
        ]
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

    anti_d = build_graphs(args, "Dbar")
    anti_p = build_graphs(args, "Pbar")
    splits = split_balanced(args, anti_p=anti_p, anti_d=anti_d)

    for name, graphs in splits.items():
        out = args.output_dir / f"{name}.pt"
        torch.save(graphs, out)
        print(f"saved {out}")

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
