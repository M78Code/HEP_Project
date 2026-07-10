#!/usr/bin/env python3
"""Create an exact class-balanced subset from manifest + graph-cache files.

This is intended for the Aohba 50M graph cache used by train_aohba.py with
--manifest and --cache-dir. The output is a sharded split-cache directory:

  train_antiD_000.pt, train_antiP_000.pt, ...
  val_antiD_000.pt,   val_antiP_000.pt, ...
  test_antiD_000.pt,  test_antiP_000.pt, ...

It can be consumed directly by:

  src/scripts/train_aohba.py --split-cache-dir OUTPUT_DIR
  src/scripts/evaluate_aohba_split_cache.py --cache-dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import torch


PARTICLES = ("antiD", "antiP")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-per-class", type=int, default=1_600_000)
    p.add_argument("--val-per-class", type=int, default=200_000)
    p.add_argument("--test-per-class", type=int, default=200_000)
    p.add_argument("--shard-size", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify-output",
        action="store_true",
        help="Reload each written shard and verify its graph count.",
    )
    return p.parse_args()


def pkl_to_pt(pkl_path: str, cache_dir: Path) -> Path:
    p = Path(pkl_path)
    particle = "antiD" if "antiD" in p.stem else "antiP"
    return cache_dir / particle / (p.stem + ".pt")


def graph_label(graph: Any) -> int | None:
    if not hasattr(graph, "y"):
        return None
    y = graph.y
    if hasattr(y, "view"):
        return int(y.view(-1)[0].item())
    return int(y)


def write_shard(
    graphs: list,
    output_dir: Path,
    split: str,
    particle: str,
    shard_index: int,
) -> dict:
    output_path = output_dir / f"{split}_{particle}_{shard_index:03d}.pt"
    if output_path.exists():
        raise FileExistsError(output_path)
    torch.save(graphs, output_path)

    label_counts: dict[str, int] = {}
    for graph in graphs:
        label = graph_label(graph)
        if label is None:
            continue
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1

    summary = {
        "split": split,
        "particle": particle,
        "shard_index": shard_index,
        "n_graphs": len(graphs),
        "label_counts": label_counts,
        "file": output_path.name,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def verify_shard(path: Path, expected_count: int) -> None:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or len(graphs) != expected_count:
        raise RuntimeError(
            f"invalid shard {path}: expected {expected_count:,} graphs, "
            f"got {type(graphs).__name__} with len="
            f"{len(graphs) if hasattr(graphs, '__len__') else 'N/A'}"
        )
    del graphs
    gc.collect()


def collect_split_particle(
    manifest: dict,
    cache_dir: Path,
    output_dir: Path,
    split: str,
    particle: str,
    target_count: int,
    shard_size: int,
    seed: int,
    verify_output: bool,
) -> dict:
    source_pkls = list(manifest[split][particle])
    source_files = [pkl_to_pt(path, cache_dir) for path in source_pkls]
    missing = [path for path in source_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing cache files, first={missing[0]}")

    rng = random.Random(seed)
    rng.shuffle(source_files)

    rows = []
    buffer = []
    kept = 0
    shard_index = 0
    source_files_used = 0

    for source_index, source_path in enumerate(source_files, 1):
        if kept >= target_count:
            break

        graphs = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(graphs, list) or not graphs:
            raise RuntimeError(f"invalid graph cache: {source_path}")
        rng.shuffle(graphs)
        source_files_used += 1

        for graph in graphs:
            if kept >= target_count:
                break
            buffer.append(graph)
            kept += 1
            if len(buffer) == shard_size:
                row = write_shard(buffer, output_dir, split, particle, shard_index)
                rows.append(row)
                if verify_output:
                    verify_shard(output_dir / row["file"], shard_size)
                print(
                    f"{split}/{particle}: wrote {row['file']} "
                    f"kept={kept:,}/{target_count:,} "
                    f"source_files_used={source_files_used:,}",
                    flush=True,
                )
                buffer = []
                shard_index += 1

        del graphs
        gc.collect()

    if kept < target_count:
        raise RuntimeError(
            f"{split}/{particle}: target={target_count:,}, available after scan={kept:,}"
        )

    if buffer:
        row = write_shard(buffer, output_dir, split, particle, shard_index)
        rows.append(row)
        if verify_output:
            verify_shard(output_dir / row["file"], len(buffer))
        print(
            f"{split}/{particle}: wrote {row['file']} "
            f"kept={kept:,}/{target_count:,} source_files_used={source_files_used:,}",
            flush=True,
        )

    return {
        "split": split,
        "particle": particle,
        "target_count": target_count,
        "kept": kept,
        "source_files_available": len(source_files),
        "source_files_used": source_files_used,
        "n_shards": len(rows),
        "shards": rows,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8") as f:
        manifest = json.load(f)

    targets = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }

    summary = {
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "shard_size": args.shard_size,
        "targets_per_class": targets,
        "splits": {},
    }

    for split_index, split in enumerate(SPLITS):
        summary["splits"][split] = {}
        for particle_index, particle in enumerate(PARTICLES):
            row = collect_split_particle(
                manifest=manifest,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                split=split,
                particle=particle,
                target_count=targets[split],
                shard_size=args.shard_size,
                seed=args.seed + split_index * 100_000 + particle_index * 10_000,
                verify_output=args.verify_output,
            )
            summary["splits"][split][particle] = row

    with (args.output_dir / "subset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== complete ==========")
    total = 0
    for split in SPLITS:
        split_total = 0
        for particle in PARTICLES:
            kept = summary["splits"][split][particle]["kept"]
            split_total += kept
        total += split_total
        print(f"{split:5s}: {split_total:,} events")
    print(f"total: {total:,} events")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
