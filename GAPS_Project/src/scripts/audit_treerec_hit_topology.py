#!/usr/bin/env python3
"""Audit truth-free hit-level topology features in a TreeRec graph cache.

The TreeRec ROOT files available for this study do not persist a usable
high-level reconstruction (tracks, fitted beta, vertex, etc.).  This script
therefore derives only observable event descriptors from cached hit position,
energy, time, and detector type.  It is an audit step: it identifies whether
the candidate descriptors carry class information before they are appended to
the GravNet graph-level input.

The cache must use the ``global_log`` node representation produced by
``rebuild_treerec_subset_global_norm.py``.  Raw energy/time values are
recovered from the recorded train-set normalization constants solely to form
physical, deterministic per-event summaries; no TreeMC quantity is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from GAPS_Project.src.data_parse.treerec_hit_topology import (
    ALL_FEATURE_NAMES as FEATURE_NAMES,
    load_node_normalizer,
    topology_features,
)


def scalar(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def direction_independent_auc(labels: np.ndarray, values: np.ndarray) -> float:
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def summarize(labels: np.ndarray, features: np.ndarray) -> list[dict]:
    output = []
    signal = labels == 1
    background = labels == 0
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        output.append({
            'feature': name,
            'direction_independent_auc': direction_independent_auc(labels, values),
            'antiD_median': float(np.median(values[signal])),
            'antiP_median': float(np.median(values[background])),
        })
    return sorted(output, key=lambda row: row['direction_independent_auc'], reverse=True)


def audit_split(
        cache_dir: Path, split: str, mean: np.ndarray, std: np.ndarray,
        max_graphs_per_shard: int | None) -> tuple[np.ndarray, np.ndarray]:
    shard_paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not shard_paths:
        raise FileNotFoundError(f'no {split}_*.pt shards under {cache_dir}')

    features = []
    labels = []
    for shard_index, path in enumerate(shard_paths, start=1):
        graphs = torch.load(path, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        for graph in graphs:
            features.append(topology_features(graph, mean, std))
            labels.append(scalar(graph.y))
        print(
            f'[{split} {shard_index:02d}/{len(shard_paths):02d}] '
            f'{path.name}: {len(graphs):,} events',
            flush=True,
        )
    return np.asarray(labels, dtype=np.int64), np.stack(features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--max-graphs-per-shard', type=int,
                        help='limit each shard for a fast audit')
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    normalizer_path = args.node_normalizer or args.cache_dir / 'node_feature_normalizer.json'
    mean, std = load_node_normalizer(normalizer_path)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.out_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'cache_dir': str(args.cache_dir.resolve()),
        'node_normalizer': str(normalizer_path.resolve()),
        'feature_names': list(FEATURE_NAMES),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'splits': {},
    }
    for split in args.splits:
        labels, features = audit_split(
            args.cache_dir, split, mean, std, args.max_graphs_per_shard)
        np.savez_compressed(args.out_dir / f'{split}_topology_features.npz',
                            labels=labels, features=features,
                            feature_names=np.asarray(FEATURE_NAMES))
        rows = summarize(labels, features)
        report['splits'][split] = {
            'events': int(len(labels)),
            'label_counts': {str(key): int(value) for key, value in Counter(labels).items()},
            'feature_summary': rows,
        }
        print(f'\n[{split}] events={len(labels):,} labels={dict(Counter(labels))}')
        for row in rows:
            print(
                f"{row['feature']:38s} auc={row['direction_independent_auc']:.5f} "
                f"antiD_median={row['antiD_median']:.5g} "
                f"antiP_median={row['antiP_median']:.5g}")

    (args.out_dir / 'topology_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nsaved: {args.out_dir}')


if __name__ == '__main__':
    main()
