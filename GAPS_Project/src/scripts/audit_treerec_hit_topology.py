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


FEATURE_NAMES = (
    'log_spatial_eig_small',
    'log_spatial_eig_middle',
    'log_spatial_eig_large',
    'log_energy_weighted_spatial_rms',
    'log_time_span_ns',
    'log_time_iqr_ns',
    'energy_weighted_time_std_ns_log',
    'energy_top1_fraction',
    'energy_top3_fraction',
    'late_quartile_energy_fraction',
    'log_energy_mean',
    'log_energy_std',
    'log_energy_q10',
    'log_energy_q50',
    'log_energy_q90',
    'log_energy_q90_minus_q10',
    'tof_sili_centroid_distance_log',
    'tof_sili_time_gap_ns_log',
)


def scalar(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def load_global_log_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('mode') != 'global_log':
        raise ValueError(f'{path} is not a global_log node normalizer')
    mean = np.asarray(payload['mean'], dtype=np.float64)
    std = np.asarray(payload['std'], dtype=np.float64)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError('expected six continuous node-feature constants')
    if np.any(std <= 0.0):
        raise ValueError('node normalizer has non-positive standard deviation')
    return mean, std


def recovered_raw_features(graph, mean: np.ndarray, std: np.ndarray):
    """Recover observable energy/time from the cached global-log node tensor."""
    x = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
    pos = graph.pos.detach().cpu().numpy().astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[1] != 8 or pos.shape != (len(x), 3):
        raise ValueError(f'unexpected graph shapes: x={x.shape}, pos={pos.shape}')

    log_energy = x[:, 3] * std[3] + mean[3]
    log_time = x[:, 4] * std[4] + mean[4]
    energy = np.expm1(np.clip(log_energy, 0.0, 50.0))
    time = np.expm1(np.clip(log_time, 0.0, 50.0))
    det_type = x[:, 6]
    return pos, energy, time, det_type


def topology_features(graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    pos, energy, time, det_type = recovered_raw_features(graph, mean, std)
    n_hits = len(energy)
    if n_hits == 0:
        raise ValueError('empty graph')

    weights = np.maximum(energy, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        weights = np.ones(n_hits, dtype=np.float64)
    weights /= weights.sum()

    center = np.sum(pos * weights[:, None], axis=0)
    delta = pos - center
    covariance = (delta * weights[:, None]).T @ delta
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    spatial_rms = float(np.sqrt(np.sum(weights * np.sum(delta * delta, axis=1))))

    sorted_time = np.sort(time)
    time_span = float(sorted_time[-1] - sorted_time[0]) if n_hits > 1 else 0.0
    time_iqr = float(np.quantile(time, 0.75) - np.quantile(time, 0.25))
    weighted_time_mean = float(np.sum(weights * time))
    weighted_time_std = float(np.sqrt(np.sum(weights * (time - weighted_time_mean) ** 2)))

    ranked_energy = np.sort(energy)[::-1]
    total_energy = max(float(energy.sum()), 1e-12)
    top1_fraction = float(ranked_energy[:1].sum() / total_energy)
    top3_fraction = float(ranked_energy[:3].sum() / total_energy)
    time_q75 = float(np.quantile(time, 0.75))
    late_energy_fraction = float(energy[time >= time_q75].sum() / total_energy)

    # TreeMc strict studies found the event-level energy-deposition shape to
    # be highly informative.  These are detector-level TreeRec hit summaries,
    # not primary-track quantities and not MC truth.
    energy_q10, energy_q50, energy_q90 = np.quantile(energy, (0.10, 0.50, 0.90))
    energy_mean = float(np.mean(energy))
    energy_std = float(np.std(energy))

    tof_mask = det_type < 0.5
    sili_mask = ~tof_mask
    if tof_mask.any() and sili_mask.any():
        tof_weight = weights[tof_mask]
        sili_weight = weights[sili_mask]
        tof_weight /= tof_weight.sum()
        sili_weight /= sili_weight.sum()
        tof_center = np.sum(pos[tof_mask] * tof_weight[:, None], axis=0)
        sili_center = np.sum(pos[sili_mask] * sili_weight[:, None], axis=0)
        centroid_distance = float(np.linalg.norm(tof_center - sili_center))
        tof_time = float(np.sum(time[tof_mask] * tof_weight))
        sili_time = float(np.sum(time[sili_mask] * sili_weight))
        tof_sili_time_gap = abs(tof_time - sili_time)
    else:
        centroid_distance = 0.0
        tof_sili_time_gap = 0.0

    return np.asarray([
        np.log1p(eigenvalues[0]),
        np.log1p(eigenvalues[1]),
        np.log1p(eigenvalues[2]),
        np.log1p(spatial_rms),
        np.log1p(max(time_span, 0.0)),
        np.log1p(max(time_iqr, 0.0)),
        np.log1p(max(weighted_time_std, 0.0)),
        top1_fraction,
        top3_fraction,
        late_energy_fraction,
        np.log1p(max(energy_mean, 0.0)),
        np.log1p(max(energy_std, 0.0)),
        np.log1p(max(float(energy_q10), 0.0)),
        np.log1p(max(float(energy_q50), 0.0)),
        np.log1p(max(float(energy_q90), 0.0)),
        np.log1p(max(float(energy_q90 - energy_q10), 0.0)),
        np.log1p(centroid_distance),
        np.log1p(max(tof_sili_time_gap, 0.0)),
    ], dtype=np.float32)


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
    mean, std = load_global_log_normalizer(normalizer_path)
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
