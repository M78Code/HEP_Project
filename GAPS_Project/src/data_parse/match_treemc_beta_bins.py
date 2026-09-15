#!/usr/bin/env python3
"""Create equal beta-bin samples for two or more TreeMc provenance groups."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


PARTICLES = ("antiP", "antiD")
ARRAY_NAMES = (
    "labels.npy",
    "betas.npy",
    "random_seeds.npy",
    "chain_entries.npy",
    "source_file_indices.npy",
    "source_entries.npy",
)


def parse_group(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "--group must have the form NAME=CANDIDATE_GROUP_DIRECTORY"
        )
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--group", type=parse_group, action="append", required=True)
    parser.add_argument("--beta-min", type=float, default=0.20)
    parser.add_argument("--beta-max", type=float, default=0.50)
    parser.add_argument("--beta-bins", type=int, default=60)
    return parser.parse_args()


def load_pool(group: str, particle: str, directory: Path) -> dict:
    source = directory / particle
    if not (source / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete candidate provenance: {source}")
    manifest = json.loads((source / "export_manifest.json").read_text())
    arrays = {name: np.load(source / name, mmap_mode="r") for name in ARRAY_NAMES}
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"array length mismatch: {source}")
    return {"group": group, "particle": particle, "source": source,
            "manifest": manifest, "arrays": arrays}


def split_ranges(events_per_class: int) -> dict[str, tuple[int, int]]:
    train_stop = events_per_class * 8 // 10
    val_stop = train_stop + events_per_class // 10
    return {
        "train": (0, train_stop),
        "val": (train_stop, val_stop),
        "test": (val_stop, events_per_class),
    }


def main() -> None:
    args = parse_args()
    if args.events_per_class < 1 or args.beta_bins < 1:
        raise ValueError("events per class and beta bins must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    groups = dict(args.group)
    if len(groups) != len(args.group) or len(groups) < 2:
        raise ValueError("provide at least two uniquely named --group values")
    pools = {
        (group, particle): load_pool(group, particle, directory)
        for group, directory in groups.items()
        for particle in PARTICLES
    }
    edges = np.linspace(args.beta_min, args.beta_max, args.beta_bins + 1)
    bins: dict[tuple[str, str], list[np.ndarray]] = {}
    for key, pool in pools.items():
        beta = np.asarray(pool["arrays"]["betas.npy"], dtype=np.float64)
        bins[key] = [
            np.flatnonzero(
                (beta >= low) & (beta <= high if index == args.beta_bins - 1 else beta < high)
            )
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
        ]

    capacity = np.asarray(
        [min(len(bins[key][index]) for key in pools) for index in range(args.beta_bins)],
        dtype=np.int64,
    )
    if int(capacity.sum()) < args.events_per_class:
        raise RuntimeError(
            f"only {int(capacity.sum()):,} common beta-matched events/class; "
            "increase candidate pools"
        )
    raw = capacity.astype(np.float64) * args.events_per_class / capacity.sum()
    allocation = np.minimum(np.floor(raw).astype(np.int64), capacity)
    remaining = args.events_per_class - int(allocation.sum())
    for index in np.argsort(-(raw - allocation)):
        if remaining == 0:
            break
        if allocation[index] < capacity[index]:
            allocation[index] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not allocate the requested matched sample")

    slots = np.repeat(np.arange(args.beta_bins), allocation)
    np.random.default_rng(args.seed).shuffle(slots)
    args.output_dir.mkdir(parents=True)
    for population_index, (key, pool) in enumerate(pools.items()):
        rng = np.random.default_rng(args.seed + 1000 + population_index)
        selected = np.empty(args.events_per_class, dtype=np.int64)
        for bin_index, count in enumerate(allocation):
            if count:
                chosen = rng.choice(bins[key][bin_index], int(count), replace=False)
                rng.shuffle(chosen)
                selected[np.flatnonzero(slots == bin_index)] = chosen
        destination = args.output_dir / key[0] / key[1]
        destination.mkdir(parents=True)
        for name, array in pool["arrays"].items():
            np.save(destination / name, np.asarray(array[selected]))
        shutil.copy2(pool["source"] / "source_files.txt", destination)
        manifest = dict(pool["manifest"])
        manifest.update({
            "source": "multi-group beta-bin-matched TreeMc provenance",
            "events": args.events_per_class,
            "candidate_directory": str(pool["source"]),
            "matching_seed": args.seed,
            "beta_bin_edges": edges.tolist(),
            "beta_bin_counts": allocation.tolist(),
            "matching_groups": list(groups),
            "split_events_per_class": {
                split: stop - start for split, (start, stop) in split_ranges(args.events_per_class).items()
            },
        })
        (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
        (destination / "_SUCCESS").write_text("ok\n")

    ranges = split_ranges(args.events_per_class)
    for split, (start, stop) in ranges.items():
        reference = None
        for group in groups:
            for particle in PARTICLES:
                beta = np.load(args.output_dir / group / particle / "betas.npy")
                counts, _ = np.histogram(beta[start:stop], bins=edges)
                if reference is None:
                    reference = counts
                elif not np.array_equal(counts, reference):
                    raise RuntimeError(f"{split}: beta-bin counts differ")
        print(f"[{split}] events/class={stop - start:,}; six populations match")

    summary = {
        "groups": list(groups),
        "events_per_class_per_group": args.events_per_class,
        "beta_bin_edges": edges.tolist(),
        "beta_bin_counts": allocation.tolist(),
        "seed": args.seed,
    }
    (args.output_dir / "matching_manifest.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    print("MULTI-GROUP BETA-BIN MATCHING: VALID")


if __name__ == "__main__":
    main()
