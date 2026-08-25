#!/usr/bin/env python3
"""Audit truth-free Si(Li) cluster and vertex candidates in a TreeRec cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from GAPS_Project.src.data_parse.treerec_cluster_vertex import (
    FEATURE_NAMES,
    cluster_vertex_features,
)
from GAPS_Project.src.data_parse.treerec_hit_topology import load_node_normalizer


def _label(graph) -> int:
    return int(graph.y.reshape(-1)[0].item())


def _direction_independent_auc(labels: np.ndarray, values: np.ndarray) -> float:
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def _read_split(cache_dir: Path, split: str, mean: np.ndarray, std: np.ndarray,
                limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt shards under {cache_dir}')
    labels, rows = [], []
    for index, path in enumerate(paths, start=1):
        graphs = torch.load(path, map_location='cpu', weights_only=False)
        if limit is not None:
            graphs = graphs[:limit]
        labels.extend(_label(graph) for graph in graphs)
        rows.extend(cluster_vertex_features(graph, mean, std) for graph in graphs)
        print(f'[{split} {index:02d}/{len(paths):02d}] {path.name}: {len(graphs):,} events', flush=True)
    return np.asarray(labels, dtype=np.int64), np.stack(rows)


def _summary(labels: np.ndarray, features: np.ndarray) -> list[dict]:
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        rows.append({
            'feature': name,
            'direction_independent_auc': _direction_independent_auc(labels, values),
            'antiD_median': float(np.median(values[labels == 1])),
            'antiP_median': float(np.median(values[labels == 0])),
            'minimum': float(values.min()),
            'maximum': float(values.max()),
        })
    return sorted(rows, key=lambda row: row['direction_independent_auc'], reverse=True)


def _threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> tuple[float, float, float]:
    fpr, tpr, cut = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    index = candidates[np.argmin(fpr[candidates])]
    return float(cut[index]), float(tpr[index]), float(fpr[index])


def _hard_background(labels: np.ndarray, scores: np.ndarray, features: np.ndarray,
                     target: float) -> dict:
    cut, efficiency, fpr = _threshold(labels, scores, target)
    hard = (labels == 0) & (scores >= cut)
    rejected = (labels == 0) & ~hard
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        hard_values, rejected_values = features[hard, index], features[rejected, index]
        pooled_std = np.sqrt(0.5 * (np.var(hard_values) + np.var(rejected_values)))
        effect = float((np.mean(hard_values) - np.mean(rejected_values)) / pooled_std) if pooled_std > 1e-12 else 0.0
        rows.append({
            'feature': name,
            'hard_antiP_median': float(np.median(hard_values)),
            'rejected_antiP_median': float(np.median(rejected_values)),
            'standardized_mean_difference': effect,
            'absolute_standardized_mean_difference': abs(effect),
        })
    return {
        'target_signal_efficiency': target,
        'threshold': cut,
        'actual_signal_efficiency': efficiency,
        'fpr': fpr,
        'hard_antiP': int(hard.sum()),
        'rejected_antiP': int(rejected.sum()),
        'feature_summary': sorted(rows, key=lambda row: row['absolute_standardized_mean_difference'], reverse=True),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    parser.add_argument('--max-graphs-per-shard', type=int)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--signal-efficiencies', type=float, nargs='+', default=[0.95, 0.98])
    args = parser.parse_args()
    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.evaluation_dir is not None and 'test' not in args.splits:
        raise ValueError('--evaluation-dir requires test in --splits')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.out_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    normalizer = args.node_normalizer or args.cache_dir / 'node_feature_normalizer.json'
    mean, std = load_node_normalizer(normalizer)
    report = {
        'cache_dir': str(args.cache_dir.resolve()),
        'node_normalizer': str(normalizer.resolve()),
        'feature_names': list(FEATURE_NAMES),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'note': 'Candidates use only observed TreeRec Si(Li) hit positions and energies; no truth or reconstructed ROOT fields.',
        'splits': {},
    }
    test = None
    for split in args.splits:
        labels, features = _read_split(args.cache_dir, split, mean, std, args.max_graphs_per_shard)
        if not np.isfinite(features).all():
            raise RuntimeError(f'non-finite features in {split}')
        rows = _summary(labels, features)
        report['splits'][split] = {'events': int(len(labels)), 'feature_summary': rows}
        np.savez_compressed(args.out_dir / f'{split}_cluster_vertex_features.npz', labels=labels, features=features, feature_names=np.asarray(FEATURE_NAMES))
        _write_csv(args.out_dir / f'{split}_feature_summary.csv', rows)
        print(f'\n[{split}] events={len(labels):,}')
        for row in rows:
            print(f"{row['feature']:52s} auc={row['direction_independent_auc']:.5f} antiD={row['antiD_median']:.5g} antiP={row['antiP_median']:.5g}")
        if split == 'test':
            test = labels, features

    if args.evaluation_dir is not None:
        if args.max_graphs_per_shard is not None:
            raise ValueError('hard-background audit requires complete test split')
        labels_eval = np.load(args.evaluation_dir / 'labels.npy').astype(np.int64)
        scores = np.load(args.evaluation_dir / 'scores.npy').astype(np.float64)
        labels, features = test
        if not np.array_equal(labels_eval, labels) or scores.shape != labels.shape:
            raise RuntimeError('evaluation labels/scores are not aligned with test cache')
        rows = [_hard_background(labels, scores, features, target) for target in args.signal_efficiencies]
        report['hard_background'] = rows
        for row in rows:
            suffix = str(row['target_signal_efficiency']).replace('.', 'p')
            _write_csv(args.out_dir / f'hard_antip_eff_{suffix}.csv', row['feature_summary'])
            print(f"\n[hard antiP @ eff={row['target_signal_efficiency']:.2f}] n={row['hard_antiP']}, fpr={row['fpr']:.6g}")
            for feature in row['feature_summary'][:8]:
                print(f"{feature['feature']:52s} effect={feature['standardized_mean_difference']:+.3f}")

    (args.out_dir / 'cluster_vertex_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nsaved: {args.out_dir}')


if __name__ == '__main__':
    main()
