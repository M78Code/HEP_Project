#!/usr/bin/env python3
"""Create a beta-matched subset from Nakagami old VolID PyG graph caches."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bins", default="0.25,0.30,0.35,0.40,0.45")
    p.add_argument("--min-per-class-bin", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_all(input_dir: Path) -> list:
    graphs = []
    for split in ["train", "val", "test"]:
        graphs.extend(torch.load(input_dir / f"{split}.pt", map_location="cpu", weights_only=False))
    return graphs


def bin_index(beta: float, bins: list[float]) -> int | None:
    for i in range(len(bins) - 1):
        if bins[i] <= beta < bins[i + 1]:
            return i
    return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    bins = [float(x) for x in args.bins.split(",")]

    graphs = load_all(args.input_dir)
    buckets: dict[tuple[int, int], list] = defaultdict(list)

    for g in graphs:
        label = int(g.y)
        beta = float(g.beta)
        idx = bin_index(beta, bins)
        if idx is not None:
            buckets[(label, idx)].append(g)

    selected = []
    print("beta bins:", bins)
    for idx in range(len(bins) - 1):
        p = buckets[(0, idx)]
        d = buckets[(1, idx)]
        n = min(len(p), len(d))
        print(f"[{bins[idx]:.2f},{bins[idx+1]:.2f}) antiP={len(p)} antiD={len(d)} use={n}")
        if n < args.min_per_class_bin:
            print("  skip")
            continue
        rng.shuffle(p)
        rng.shuffle(d)
        selected.extend(p[:n])
        selected.extend(d[:n])

    rng.shuffle(selected)
    n = len(selected)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": selected[:n_train],
        "val": selected[n_train : n_train + n_val],
        "test": selected[n_train + n_val :],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, xs in splits.items():
        labels = torch.tensor([int(g.y) for g in xs], dtype=torch.long)
        betas = torch.tensor([float(g.beta) for g in xs], dtype=torch.float32)
        print(
            name,
            len(xs),
            "labels=",
            torch.bincount(labels, minlength=2).tolist(),
            "beta_mean=",
            [float(betas[labels == i].mean()) for i in [0, 1]],
        )
        torch.save(xs, args.output_dir / f"{name}.pt")

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
