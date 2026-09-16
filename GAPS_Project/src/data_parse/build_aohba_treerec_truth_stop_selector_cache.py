#!/usr/bin/env python3
"""Build a source-file-held-out TreeRec cache for truth strict-stop selection.

TreeMc strict-stop flags supply labels only.  Every graph input is made from
TreeRec hitseries, exactly as for the ordinary GravNet classifier.  Each ROOT
source file belongs to one split only, and every split is balanced separately
over particle type (antiP/antiD) and target (non-strict/strict).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from GAPS_Project.src.data_parse.build_aohba_fixedgrid_selected_treerec_cache import (
    FEATURE_NAMES,
    GraphBuilder,
    read_event_chunk,
)


PARTICLES = {
    "antiP": {"raw_name": "antip", "target_label": 0},
    "antiD": {"raw_name": "antid", "target_label": 1},
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--train-events-per-cell", type=int, default=10_000)
    parser.add_argument("--val-events-per-cell", type=int, default=2_000)
    parser.add_argument("--test-events-per-cell", type=int, default=3_000)
    return parser.parse_args()


def load_pool(directory: Path, particle: str) -> dict:
    source = directory / particle
    required = (
        "source_file_indices.npy", "source_entries.npy", "source_files.txt",
        "truth_strict_stop.npy", "_SUCCESS",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"{source}: missing {', '.join(missing)}")
    file_indices = np.asarray(np.load(source / "source_file_indices.npy", mmap_mode="r"), dtype=np.int64)
    entries = np.asarray(np.load(source / "source_entries.npy", mmap_mode="r"), dtype=np.int64)
    truth = np.asarray(np.load(source / "truth_strict_stop.npy", mmap_mode="r"), dtype=bool)
    if not (len(file_indices) == len(entries) == len(truth)) or len(truth) == 0:
        raise RuntimeError(f"{source}: inconsistent or empty provenance")
    files = [Path(line) for line in (source / "source_files.txt").read_text().splitlines()
             if line.strip()]
    if np.any(file_indices < 0) or np.any(file_indices >= len(files)):
        raise RuntimeError(f"{source}: invalid source file index")
    return {
        "directory": source,
        "file_indices": file_indices,
        "entries": entries,
        "truth": truth,
        "files": files,
    }


def assign_source_files(file_indices: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Allocate whole ROOT source files to 60/20/20 train/val/test splits."""
    unique = np.unique(file_indices)
    if len(unique) < 3:
        raise RuntimeError("need at least three source ROOT files per particle")
    shuffled = np.random.default_rng(seed).permutation(unique)
    test_count = max(1, round(len(unique) * 0.20))
    val_count = max(1, round(len(unique) * 0.20))
    train_count = len(unique) - test_count - val_count
    if train_count < 1:
        train_count, val_count, test_count = 1, 1, len(unique) - 2
    split_files = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }
    return {split: np.isin(file_indices, values) for split, values in split_files.items()}


def choose_balanced_indices(
    pool: dict, split_masks: dict[str, np.ndarray], counts: dict[str, int],
    seed: int, particle: str,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    selected, audit = {}, {}
    for split in SPLITS:
        pieces = []
        per_target = {}
        for target in (0, 1):
            available = np.flatnonzero(split_masks[split] & (pool["truth"] == bool(target)))
            required = counts[split]
            if len(available) < required:
                raise RuntimeError(
                    f"{particle}/{split}/target={target}: only {len(available):,} "
                    f"source-held-out events, need {required:,}"
                )
            chosen = rng.choice(available, required, replace=False)
            pieces.append((chosen, np.full(required, target, dtype=np.int64)))
            per_target[str(target)] = int(len(available))
        indices = np.concatenate([part[0] for part in pieces])
        targets = np.concatenate([part[1] for part in pieces])
        order = rng.permutation(len(indices))
        selected[split] = np.column_stack((indices[order], targets[order]))
        audit[split] = {
            "available_per_target": per_target,
            "selected_per_target": counts[split],
            "source_file_indices": np.unique(pool["file_indices"][indices]).tolist(),
        }
    return selected, audit


def make_provenance(pool: dict, selected: np.ndarray) -> dict:
    indices = selected[:, 0].astype(np.int64)
    return {
        "file_indices": pool["file_indices"][indices],
        "entries": pool["entries"][indices],
        "files": pool["files"],
        "target_labels": selected[:, 1].astype(np.int64),
    }


def iter_events(provenance: dict, raw_particle: str, chunk_size: int):
    total = len(provenance["entries"])
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        events = read_event_chunk(provenance, raw_particle, start, stop)
        targets = provenance["target_labels"][start:stop]
        for event, target in zip(events, targets):
            event["truth_stop_target"] = int(target)
        yield events


def fit_normalizer(
    selected: dict[str, dict[str, dict]], counts: dict[str, int], args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    builder = GraphBuilder(k=args.k, normalize=False)
    sums = np.zeros(6, dtype=np.float64)
    sums_squared = np.zeros(6, dtype=np.float64)
    n_nodes = 0
    for particle, detail in PARTICLES.items():
        provenance = selected["train"][particle]
        for events in iter_events(provenance, detail["raw_name"], args.chunk_size):
            for event in events:
                values = builder.raw_node_features_from_dict(event)[:, :6].astype(np.float64)
                values[:, 3:6] = np.log1p(np.clip(values[:, 3:6], 0.0, None))
                if not np.isfinite(values).all():
                    raise RuntimeError("non-finite training node feature")
                sums += values.sum(axis=0)
                sums_squared += np.square(values).sum(axis=0)
                n_nodes += len(values)
        print(f"[STATS] {particle}: {counts['train'] * 2:,} train events", flush=True)
    expected = 2 * 2 * counts["train"]
    if n_nodes == 0:
        raise RuntimeError("no train nodes")
    mean = sums / n_nodes
    variance = np.maximum(sums_squared / n_nodes - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    print(f"[NORMALIZER] expected train events={expected:,}")
    return mean.astype(np.float32), std.astype(np.float32), n_nodes


def build_cache(
    selected: dict[str, dict[str, dict]], counts: dict[str, int], args: argparse.Namespace,
    mean: np.ndarray, std: np.ndarray,
) -> dict:
    builder = GraphBuilder(
        k=args.k, normalize=True, normalization_mode="global_log",
        global_feature_mean=mean, global_feature_std=std,
    )
    summary = {}
    for split in SPLITS:
        labels = {0: 0, 1: 0}
        shards = 0
        for particle, detail in PARTICLES.items():
            provenance = selected[split][particle]
            shard_index = 0
            for events in iter_events(provenance, detail["raw_name"], args.chunk_size):
                graphs = []
                for event in events:
                    graph = builder.build_from_dict(event)
                    graph.y = torch.tensor([event["truth_stop_target"]], dtype=torch.long)
                    graph.source_file_index = torch.tensor([event["source_file_index"]], dtype=torch.long)
                    graph.source_root_entry = torch.tensor([event["source_root_entry"]], dtype=torch.long)
                    if not torch.isfinite(graph.x).all():
                        raise RuntimeError("non-finite normalized graph")
                    labels[int(graph.y.item())] += 1
                    graphs.append(graph)
                path = args.output_dir / f"{split}_{particle}_{shard_index:03d}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save(graphs, temporary)
                temporary.replace(path)
                path.with_suffix(".json").write_text(json.dumps({
                    "split": split,
                    "source_particle": particle,
                    "target": "truth strict-stop",
                    "n_graphs": len(graphs),
                    "normalization": "train-only global_log",
                }, indent=2))
                print(f"[SAVE] {path.name}: {len(graphs):,}")
                shard_index += 1
                shards += 1
        expected_per_label = 2 * counts[split]
        if labels != {0: expected_per_label, 1: expected_per_label}:
            raise RuntimeError(f"{split}: target label counts {labels}")
        summary[split] = {
            "events": 4 * counts[split],
            "events_per_target": expected_per_label,
            "source_particles_per_target": counts[split],
            "target_label_counts": labels,
            "shards": shards,
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    counts = {
        "train": args.train_events_per_cell,
        "val": args.val_events_per_cell,
        "test": args.test_events_per_cell,
    }
    if args.k < 1 or args.chunk_size < 1 or any(value < 1 for value in counts.values()):
        raise ValueError("k, chunk size, and split counts must be positive")
    pools = {particle: load_pool(args.provenance_dir, particle) for particle in PARTICLES}
    selected: dict[str, dict[str, dict]] = {split: {} for split in SPLITS}
    source_audit = {}
    for particle, pool in pools.items():
        split_masks = assign_source_files(pool["file_indices"], args.seed + (0 if particle == "antiP" else 1))
        selections, audit = choose_balanced_indices(
            pool, split_masks, counts, args.seed + (100 if particle == "antiP" else 200), particle)
        source_audit[particle] = audit
        for split, values in selections.items():
            selected[split][particle] = make_provenance(pool, values)

    args.output_dir.mkdir(parents=True)
    mean, std, train_nodes = fit_normalizer(selected, counts, args)
    (args.output_dir / "node_feature_normalizer.json").write_text(json.dumps({
        "mode": "global_log",
        "fit_split": "train",
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(), "std": std.tolist(), "train_nodes": train_nodes,
    }, indent=2))
    print("normalizer:")
    for name, feature_mean, feature_std in zip(FEATURE_NAMES, mean, std):
        print(f"  {name:14s} mean={feature_mean:.7g} std={feature_std:.7g}")
    splits = build_cache(selected, counts, args, mean, std)
    manifest = {
        "purpose": "TreeRec-only classifier for TreeMc strict-stop audit label",
        "input": "TreeRec hitseries graph; no TreeMc input feature",
        "target": "TreeMc summary-at-rest + K=0 in tracker + zero-step",
        "split": "whole source ROOT files held out across train/val/test",
        "balance": "antiP/antiD and strict/non-strict balanced independently in every split",
        "source_audit": source_audit,
        "splits": splits,
        "k": args.k,
    }
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print(json.dumps(splits, indent=2))
    print(f"complete: {args.output_dir}")
    print("AOHBA TREEREC TRUTH-STOP SELECTOR CACHE: VALID")


if __name__ == "__main__":
    main()
