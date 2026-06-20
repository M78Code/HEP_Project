"""Create a reproducible class-balanced subset from class-pure graph shards.

The source train/val/test split is preserved. Only full-size shards are
selected, so the requested per-class event counts must be divisible by the
source shard size. Selected .pt and .json files are referenced by relative
symbolic links; graph data is not copied.
"""
import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch


PARTICLES = ("antiD", "antiP")
SPLITS = ("train", "val", "test")


def load_graph_count(summary_path: Path) -> int:
    with open(summary_path, encoding="utf-8") as file:
        return int(json.load(file)["n_graphs"])


def candidate_shards(
    source_dir: Path,
    split: str,
    particle: str,
    shard_size: int,
) -> list[Path]:
    candidates = []
    for path in sorted(source_dir.glob(f"{split}_{particle}_*.pt")):
        summary_path = path.with_suffix(".json")
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        if load_graph_count(summary_path) == shard_size:
            candidates.append(path)
    return candidates


def verify_shard(path: Path, expected_size: int):
    try:
        graphs = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(
            f"failed to load selected shard {path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(graphs, list) or len(graphs) != expected_size:
        raise RuntimeError(
            f"invalid selected shard {path}: expected a list of "
            f"{expected_size:,} graphs, got {type(graphs).__name__} "
            f"with length {len(graphs) if hasattr(graphs, '__len__') else 'N/A'}"
        )
    del graphs
    gc.collect()


def relative_symlink(source: Path, destination: Path):
    destination.symlink_to(os.path.relpath(source, destination.parent))


def select_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    events_per_class: int,
    shard_size: int,
    seed: int,
    verify: bool,
) -> dict:
    if events_per_class % shard_size:
        raise ValueError(
            f"{split} events per class ({events_per_class:,}) must be "
            f"divisible by shard size ({shard_size:,})"
        )
    required_shards = events_per_class // shard_size
    result = {
        "split": split,
        "events_per_class": events_per_class,
        "total_events": events_per_class * 2,
        "shard_size": shard_size,
        "particles": {},
    }

    for particle_index, particle in enumerate(PARTICLES):
        candidates = candidate_shards(
            source_dir, split, particle, shard_size)
        if len(candidates) < required_shards:
            raise RuntimeError(
                f"{split}/{particle}: need {required_shards} full shards, "
                f"found {len(candidates)}"
            )

        rng = random.Random(seed + particle_index * 10_000)
        selected = rng.sample(candidates, required_shards)
        selected.sort(key=lambda path: path.name)

        linked_files = []
        for index, source_path in enumerate(selected, 1):
            if verify:
                print(
                    f"verify {split}/{particle} "
                    f"[{index:02d}/{required_shards:02d}] "
                    f"{source_path.name}",
                    flush=True,
                )
                verify_shard(source_path, shard_size)

            for suffix in (".pt", ".json"):
                current_source = source_path.with_suffix(suffix).resolve()
                destination = output_dir / current_source.name
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(
                        f"output already exists: {destination}")
                relative_symlink(current_source, destination)
            linked_files.append(source_path.name)

        result["particles"][particle] = {
            "n_shards": required_shards,
            "n_events": events_per_class,
            "files": linked_files,
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=1_400_000)
    parser.add_argument("--val-per-class", type=int, default=300_000)
    parser.add_argument("--test-per-class", type=int, default=300_000)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="load every selected shard before creating its links",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paddle_ids_path = args.source_dir / "paddle_ids.json"
    if paddle_ids_path.exists():
        relative_symlink(
            paddle_ids_path.resolve(),
            args.output_dir / "paddle_ids.json",
        )

    counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "selection": "random full-size shards within the existing split",
        "splits": {},
    }

    try:
        for split_index, split in enumerate(SPLITS):
            manifest["splits"][split] = select_split(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                split=split,
                events_per_class=counts[split],
                shard_size=args.shard_size,
                seed=args.seed + split_index * 100_000,
                verify=args.verify,
            )
    except Exception:
        print(
            "\nSubset creation failed. Remove the incomplete output directory "
            "before retrying.",
            flush=True,
        )
        raise

    with open(
        args.output_dir / "subset_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    total_events = sum(
        row["total_events"] for row in manifest["splits"].values())
    print("\n========== complete ==========")
    for split, row in manifest["splits"].items():
        print(
            f"{split:5s}: {row['total_events']:,} events "
            f"({row['events_per_class']:,} per class)"
        )
    print(f"total: {total_events:,} events")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
