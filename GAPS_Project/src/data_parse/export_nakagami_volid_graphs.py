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
    p.add_argument("--file-id", default="000", help="single file id, e.g. 000")
    p.add_argument("--file-ids", default=None, help="comma-separated file ids, e.g. 000,001,002")
    p.add_argument("--file-id-start", default=None, help="inclusive first file id, e.g. 000")
    p.add_argument("--file-id-end", default=None, help="inclusive last file id, e.g. 039")
    p.add_argument("--auto-file-ids", action="store_true", help="use all common Dbar/Pbar Atrest/VolID file ids")
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=50000, help="cap graphs per class; use 0 to keep all")
    p.add_argument("--train-per-class", type=int, default=40000)
    p.add_argument("--val-per-class", type=int, default=5000)
    p.add_argument("--test-per-class", type=int, default=5000)
    p.add_argument("--split-fracs", default=None, help="train,val,test fractions, e.g. 0.8,0.1,0.1")
    return p.parse_args()


def csv_name(particle: str, kind: str, file_id: str) -> str:
    return f"CNN211118_{particle}_isot_50K_beta02to05_{kind}_{file_id}.csv"


def resolve_file_ids(args: argparse.Namespace) -> list[str]:
    if args.file_ids:
        ids = [x.strip() for x in args.file_ids.split(",") if x.strip()]
        if not ids:
            raise ValueError("--file-ids was provided but no ids were parsed")
        return ids

    if args.file_id_start is not None or args.file_id_end is not None:
        if args.file_id_start is None or args.file_id_end is None:
            raise ValueError("--file-id-start and --file-id-end must be provided together")
        start = int(args.file_id_start)
        end = int(args.file_id_end)
        if end < start:
            raise ValueError("--file-id-end must be >= --file-id-start")
        width = max(len(args.file_id_start), len(args.file_id_end), 3)
        return [f"{i:0{width}d}" for i in range(start, end + 1)]

    if args.auto_file_ids:
        def ids_for(particle: str, kind: str) -> set[str]:
            ids = set()
            for csv_path in args.csv_dir.glob(csv_name(particle, kind, "*")):
                ids.add(csv_path.stem.rsplit("_", 1)[-1])
            return ids

        common = (
            ids_for("Dbar", "Atrest")
            & ids_for("Dbar", "VolID")
            & ids_for("Pbar", "Atrest")
            & ids_for("Pbar", "VolID")
        )
        ids = sorted(common)
        if not ids:
            raise FileNotFoundError(f"no common Dbar/Pbar Atrest/VolID file ids found in {args.csv_dir}")
        return ids

    return [args.file_id]


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


def build_graphs(args: argparse.Namespace, particle: str, file_ids: list[str]) -> list[Data]:
    max_graphs = None if args.max_per_class <= 0 else args.max_per_class
    graphs: list[Data] = []
    total_missing = 0

    for file_id in file_ids:
        atrest_path = args.csv_dir / csv_name(particle, "Atrest", file_id)
        volid_path = args.csv_dir / csv_name(particle, "VolID", file_id)

        if not atrest_path.exists():
            raise FileNotFoundError(atrest_path)
        if not volid_path.exists():
            raise FileNotFoundError(volid_path)

        meta = load_atrest(atrest_path)
        hits = load_hits(volid_path, set(meta))

        before = len(graphs)
        missing = 0
        for key in sorted(meta):
            rows = hits.get(key)
            if not rows:
                missing += 1
                continue
            graph = graph_from_event(key, meta[key], rows, args.knn_k)
            if graph is not None:
                graphs.append(graph)
            if max_graphs is not None and len(graphs) >= max_graphs:
                break

        total_missing += missing
        added = len(graphs) - before
        print(f"{particle} {file_id}: added={added:,} cumulative={len(graphs):,} missing_hits={missing:,}")

        if max_graphs is not None and len(graphs) >= max_graphs:
            break

    print(f"{particle}: graphs={len(graphs):,} missing_hits_total={total_missing:,}")
    return graphs


def split_counts(args: argparse.Namespace, anti_p: list[Data], anti_d: list[Data]) -> tuple[int, int, int]:
    if args.split_fracs:
        fracs = [float(x.strip()) for x in args.split_fracs.split(",")]
        if len(fracs) != 3:
            raise ValueError("--split-fracs must contain train,val,test fractions")
        if not 0.999 <= sum(fracs) <= 1.001:
            raise ValueError("--split-fracs must sum to 1")
        n_per_class = min(len(anti_p), len(anti_d))
        train_per_class = int(n_per_class * fracs[0])
        val_per_class = int(n_per_class * fracs[1])
        test_per_class = n_per_class - train_per_class - val_per_class
        return train_per_class, val_per_class, test_per_class

    return args.train_per_class, args.val_per_class, args.test_per_class


def split_balanced(args: argparse.Namespace, anti_p: list[Data], anti_d: list[Data]) -> dict[str, list[Data]]:
    rng = random.Random(args.seed)
    rng.shuffle(anti_p)
    rng.shuffle(anti_d)

    train_per_class, val_per_class, test_per_class = split_counts(args, anti_p, anti_d)
    need = train_per_class + val_per_class + test_per_class
    if len(anti_p) < need or len(anti_d) < need:
        raise RuntimeError(f"not enough graphs per class: antiP={len(anti_p)}, antiD={len(anti_d)}, need={need}")

    def cut(xs: list[Data]) -> tuple[list[Data], list[Data], list[Data]]:
        train = xs[:train_per_class]
        val = xs[train_per_class : train_per_class + val_per_class]
        test = xs[train_per_class + val_per_class : train_per_class + val_per_class + test_per_class]
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

    file_ids = resolve_file_ids(args)
    preview = ", ".join(file_ids[:10])
    suffix = " ..." if len(file_ids) > 10 else ""
    print(f"file ids ({len(file_ids)}): {preview}{suffix}")

    anti_d = build_graphs(args, "Dbar", file_ids)
    anti_p = build_graphs(args, "Pbar", file_ids)
    splits = split_balanced(args, anti_p=anti_p, anti_d=anti_d)

    for name, graphs in splits.items():
        out = args.output_dir / f"{name}.pt"
        torch.save(graphs, out)
        print(f"saved {out}")

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
