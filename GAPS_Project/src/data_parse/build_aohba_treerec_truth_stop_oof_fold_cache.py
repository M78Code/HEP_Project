#!/usr/bin/env python3
"""Build one source-file-held-out fold for the TreeRec strict-stop selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from GAPS_Project.src.data_parse.build_aohba_treerec_truth_stop_selector_cache import (
    PARTICLES,
    build_cache,
    choose_balanced_indices,
    fit_normalizer,
    load_pool,
    make_provenance,
)


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--train-events-per-cell", type=int, default=10_000)
    parser.add_argument("--val-events-per-cell", type=int, default=2_000)
    parser.add_argument("--test-events-per-cell", type=int, default=3_000)
    return parser.parse_args()


def source_fold_masks(
    file_indices: np.ndarray, folds: int, fold_index: int, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Assign complete ROOT files; test and validation are adjacent folds."""
    unique = np.unique(file_indices)
    if len(unique) < folds:
        raise RuntimeError(
            f"only {len(unique)} source ROOT files, cannot form {folds} folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    partitions = [np.asarray(part, dtype=np.int64) for part in np.array_split(shuffled, folds)]
    test_files = partitions[fold_index]
    val_files = partitions[(fold_index + 1) % folds]
    train_files = np.concatenate([
        part for index, part in enumerate(partitions)
        if index not in (fold_index, (fold_index + 1) % folds)
    ])
    groups = {"train": train_files, "val": val_files, "test": test_files}
    masks = {name: np.isin(file_indices, values) for name, values in groups.items()}
    if not all(mask.any() for mask in masks.values()):
        raise RuntimeError(f"fold {fold_index}: an empty source-file split was produced")
    return masks, {name: values.tolist() for name, values in groups.items()}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.folds < 3 or not 0 <= args.fold_index < args.folds:
        raise ValueError("require at least three folds and a valid --fold-index")
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
        particle_seed = args.seed + (0 if particle == "antiP" else 1)
        masks, files = source_fold_masks(
            pool["file_indices"], args.folds, args.fold_index, particle_seed)
        choices, choice_audit = choose_balanced_indices(
            pool, masks, counts, args.seed + 1000 * args.fold_index + particle_seed, particle)
        source_audit[particle] = {"source_file_indices": files, "sampling": choice_audit}
        for split, values in choices.items():
            selected[split][particle] = make_provenance(pool, values)

    args.output_dir.mkdir(parents=True)
    mean, std, train_nodes = fit_normalizer(selected, counts, args)
    (args.output_dir / "node_feature_normalizer.json").write_text(json.dumps({
        "mode": "global_log", "fit_split": "train", "mean": mean.tolist(),
        "std": std.tolist(), "train_nodes": train_nodes,
    }, indent=2))
    print("normalizer:")
    for name, value, scale in zip(("x_mm", "y_mm", "z_mm", "log1p_energy", "log1p_time", "log1p_dedx"), mean, std):
        print(f"  {name:14s} mean={value:.7g} std={scale:.7g}")
    splits = build_cache(selected, counts, args, mean, std)
    manifest = {
        "purpose": "folded TreeRec-only classifier for TreeMc strict-stop audit label",
        "input": "TreeRec hitseries graph; no TreeMc input feature",
        "target": "TreeMc summary-at-rest + K=0 in tracker + zero-step",
        "fold": {"index": args.fold_index, "count": args.folds},
        "source_file_holdout": source_audit,
        "balance": "antiP/antiD and strict/non-strict balanced independently",
        "splits": splits,
        "k": args.k,
    }
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print(json.dumps(splits, indent=2))
    print(f"complete: {args.output_dir}")
    print("AOHBA TREEREC TRUTH-STOP OOF FOLD CACHE: VALID")


if __name__ == "__main__":
    main()
