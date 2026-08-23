#!/usr/bin/env python3
"""Audit truth-free track/terminal/star candidates in a TreeRec graph cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from GAPS_Project.src.data_parse.treerec_hit_topology import load_node_normalizer
from GAPS_Project.src.data_parse.treerec_track_star import (
    FEATURE_NAMES,
    track_star_features,
)


def scalar_label(graph) -> int:
    return int(graph.y.reshape(-1)[0].item())


def direction_independent_auc(labels: np.ndarray, values: np.ndarray) -> float:
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def read_split(
        cache_dir: Path, split: str, mean: np.ndarray, std: np.ndarray,
        max_graphs_per_shard: int | None) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt shards under {cache_dir}')
    labels, rows = [], []
    for shard_index, path in enumerate(paths, start=1):
        graphs = torch.load(path, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        for graph in graphs:
            labels.append(scalar_label(graph))
            rows.append(track_star_features(graph, mean, std))
        print(f'[{split} {shard_index:02d}/{len(paths):02d}] '
              f'{path.name}: {len(graphs):,} events', flush=True)
    return np.asarray(labels, dtype=np.int64), np.stack(rows)


def summary(labels: np.ndarray, features: np.ndarray) -> list[dict]:
    anti_d = labels == 1
    anti_p = labels == 0
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index]
        rows.append({
            'feature': name,
            'direction_independent_auc': direction_independent_auc(labels, values),
            'antiD_median': float(np.median(values[anti_d])),
            'antiP_median': float(np.median(values[anti_p])),
            'minimum': float(values.min()),
            'maximum': float(values.max()),
        })
    return sorted(rows, key=lambda row: row['direction_independent_auc'], reverse=True)


def threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> tuple[float, float, float]:
    fpr, tpr, cut = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    index = candidates[np.argmin(fpr[candidates])]
    return float(cut[index]), float(tpr[index]), float(fpr[index])


def hard_background_summary(
        labels: np.ndarray, scores: np.ndarray, features: np.ndarray,
        target: float) -> dict:
    cut, eff, fpr = threshold(labels, scores, target)
    hard = (labels == 0) & (scores >= cut)
    rejected = (labels == 0) & ~hard
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        hard_values = features[hard, index]
        rejected_values = features[rejected, index]
        pooled_std = np.sqrt(0.5 * (
            np.var(hard_values) + np.var(rejected_values)))
        effect = (
            float((np.mean(hard_values) - np.mean(rejected_values)) / pooled_std)
            if pooled_std > 1e-12 else 0.0)
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
        'actual_signal_efficiency': eff,
        'fpr': fpr,
        'hard_antiP': int(hard.sum()),
        'rejected_antiP': int(rejected.sum()),
        'feature_summary': sorted(
            rows, key=lambda row: row['absolute_standardized_mean_difference'],
            reverse=True),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max-graphs-per-shard', type=int)
    parser.add_argument('--evaluation-dir', type=Path,
                        help='optional baseline evaluation directory for hard antiP audit')
    parser.add_argument('--signal-efficiencies', type=float, nargs='+',
                        default=[0.95, 0.98])
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if any(not 0.0 < value < 1.0 for value in args.signal_efficiencies):
        raise ValueError('--signal-efficiencies must be in (0, 1)')
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
        'note': (
            'Features are deterministic candidates from observed TreeRec hits. '
            'They do not use TreeMc metadata, labels, MC beta, or ROOT tracks.'),
        'splits': {},
    }
    saved_test = None
    for split in args.splits:
        labels, features = read_split(
            args.cache_dir, split, mean, std, args.max_graphs_per_shard)
        if not np.isfinite(features).all():
            raise RuntimeError(f'non-finite features in {split}')
        rows = summary(labels, features)
        report['splits'][split] = {
            'events': int(len(labels)),
            'label_counts': {
                'antiP': int(np.count_nonzero(labels == 0)),
                'antiD': int(np.count_nonzero(labels == 1)),
            },
            'feature_summary': rows,
        }
        np.savez_compressed(args.out_dir / f'{split}_track_star_features.npz',
                            labels=labels, features=features,
                            feature_names=np.asarray(FEATURE_NAMES))
        write_csv(args.out_dir / f'{split}_feature_summary.csv', rows)
        print(f'\n[{split}] events={len(labels):,}')
        for row in rows:
            print(f"{row['feature']:48s} auc={row['direction_independent_auc']:.5f} "
                  f"antiD={row['antiD_median']:.5g} antiP={row['antiP_median']:.5g}")
        if split == 'test':
            saved_test = labels, features

    if args.evaluation_dir is not None:
        labels_path = args.evaluation_dir / 'labels.npy'
        scores_path = args.evaluation_dir / 'scores.npy'
        labels_eval = np.load(labels_path).astype(np.int64)
        scores = np.load(scores_path).astype(np.float64)
        test_labels, test_features = saved_test
        if args.max_graphs_per_shard is not None:
            raise ValueError('hard-background audit requires the complete test split')
        if not np.array_equal(labels_eval, test_labels) or scores.shape != test_labels.shape:
            raise RuntimeError('evaluation labels/scores are not aligned with test cache')
        hard_rows = [hard_background_summary(
            test_labels, scores, test_features, target)
            for target in args.signal_efficiencies]
        report['hard_background'] = hard_rows
        for row in hard_rows:
            suffix = str(row['target_signal_efficiency']).replace('.', 'p')
            write_csv(args.out_dir / f'hard_antip_eff_{suffix}.csv', row['feature_summary'])
            print(f"\n[hard antiP @ eff={row['target_signal_efficiency']:.2f}] "
                  f"n={row['hard_antiP']}, fpr={row['fpr']:.6g}")
            for feature in row['feature_summary'][:8]:
                print(f"{feature['feature']:48s} effect="
                      f"{feature['standardized_mean_difference']:+.3f}")

    (args.out_dir / 'track_star_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nsaved: {args.out_dir}')


if __name__ == '__main__':
    main()
