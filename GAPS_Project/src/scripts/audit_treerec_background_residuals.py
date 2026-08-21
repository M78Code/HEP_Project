"""Compare hard and rejected antiP events using existing TreeRec observables.

This is a diagnostic, not a training script.  It aligns an evaluation's saved
``labels.npy`` and ``scores.npy`` with the test graph cache, then compares
antiP events that pass the antiD classifier threshold with the antiP events
that are correctly rejected.  No TreeMc value is used as a candidate feature.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

from GAPS_Project.src.models.tree_rec_features import build_base_graph_feat


BASE_FEATURE_NAMES = [
    'n_hits',
    'total_energy',
    *[f'sili_energy_layer_{index:02d}' for index in range(16)],
    *[f'tof_energy_layer_{index:02d}' for index in range(16)],
    'outer_tof_energy_scaled',
    'inner_tof_energy_scaled',
    'outer_tof_n_hits_scaled',
    'inner_tof_n_hits_scaled',
    'tof_dt_scaled',
    'outer_entry_x_scaled',
    'outer_entry_y_scaled',
    'outer_entry_z_scaled',
    'inner_entry_x_scaled',
    'inner_entry_y_scaled',
    'inner_entry_z_scaled',
]


@dataclass(frozen=True)
class GroupSummary:
    name: str
    count: int
    median: float
    mean: float


def rejection_threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> dict:
    """Use exactly the same ROC selection rule as evaluation script."""
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    index = candidates[np.argmin(fpr[candidates])]
    return {
        'target_signal_efficiency': float(target),
        'actual_signal_efficiency': float(tpr[index]),
        'fpr': float(fpr[index]),
        'threshold': float(thresholds[index]),
    }


def load_test_graphs(cache_dir: Path) -> list:
    files = sorted(cache_dir.glob('test_*.pt'))
    if not files:
        raise FileNotFoundError(f'no test_*.pt under {cache_dir}')
    graphs = []
    for index, path in enumerate(files, start=1):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        graphs.extend(shard)
        print(f'[{index:02d}/{len(files):02d}] {path.name}: {len(shard):,} events')
    return graphs


def graph_feature_matrix(graphs: list) -> tuple[np.ndarray, np.ndarray]:
    rows, mc_beta = [], []
    for graph in graphs:
        base = build_base_graph_feat(graph).view(-1).detach().cpu().numpy()
        if base.shape != (len(BASE_FEATURE_NAMES),):
            raise RuntimeError(f'unexpected base graph feature shape: {base.shape}')

        tof = base[34:45]
        outer_energy, inner_energy = tof[0], tof[1]
        profile_sili = float(base[2:18].sum())
        profile_tof = float(base[18:34].sum())
        total_profile_energy = profile_sili + profile_tof
        path_mm = float(np.linalg.norm(tof[8:11] - tof[5:8]) * 1000.0)
        derived = np.asarray([
            profile_sili,
            profile_tof,
            profile_sili / max(total_profile_energy, 1e-8),
            outer_energy / max(outer_energy + inner_energy, 1e-8),
            tof[4] * 50.0,
            path_mm,
        ], dtype=np.float64)
        rows.append(np.concatenate([base.astype(np.float64), derived]))
        mc_beta.append(float(graph.mc_beta.view(-1)[0]))

    names = BASE_FEATURE_NAMES + [
        'sili_profile_energy_sum',
        'tof_profile_energy_sum',
        'sili_profile_energy_fraction',
        'outer_tof_energy_fraction',
        'tof_dt_ns',
        'tof_entry_path_mm',
    ]
    return np.stack(rows), np.asarray(mc_beta, dtype=np.float64), names


def summarize(values: np.ndarray, name: str) -> GroupSummary:
    return GroupSummary(
        name=name,
        count=int(len(values)),
        median=float(np.median(values)),
        mean=float(np.mean(values)),
    )


def feature_comparison(
        feature_matrix: np.ndarray, feature_names: list[str],
        hard_mask: np.ndarray, rejected_mask: np.ndarray) -> list[dict]:
    hard = feature_matrix[hard_mask]
    rejected = feature_matrix[rejected_mask]
    rows = []
    for index, name in enumerate(feature_names):
        hard_values = hard[:, index]
        rejected_values = rejected[:, index]
        pooled_std = np.sqrt(
            0.5 * (np.var(hard_values) + np.var(rejected_values)))
        effect_size = (
            float((np.mean(hard_values) - np.mean(rejected_values)) / pooled_std)
            if pooled_std > 1e-12 else 0.0
        )
        rows.append({
            'feature': name,
            'hard_median': float(np.median(hard_values)),
            'rejected_median': float(np.median(rejected_values)),
            'hard_mean': float(np.mean(hard_values)),
            'rejected_mean': float(np.mean(rejected_values)),
            'standardized_mean_difference': effect_size,
            'absolute_standardized_mean_difference': abs(effect_size),
        })
    return sorted(rows, key=lambda row: row['absolute_standardized_mean_difference'], reverse=True)


def tof_dt_matched_signal_mask(
        labels: np.ndarray, feature_matrix: np.ndarray, hard_mask: np.ndarray,
        tof_dt_index: int, bins: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Select antiD events with the hard antiP TOF-time distribution.

    The raw TreeRec TOF flight time is the dominant hard-background difference.
    Matching it before comparing antiP with antiD prevents that one variable
    from hiding secondary observables.  Events are sampled without replacement
    inside quantile bins whenever sufficient antiD events are available.
    """
    hard_dt = feature_matrix[hard_mask, tof_dt_index]
    signal_indices = np.flatnonzero(labels == 1)
    signal_dt = feature_matrix[signal_indices, tof_dt_index]
    edges = np.quantile(hard_dt, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        raise RuntimeError('hard antiP TOF-time distribution has no usable spread')

    hard_bin = np.digitize(hard_dt, edges[1:-1], right=False)
    signal_bin = np.digitize(signal_dt, edges[1:-1], right=False)
    selected = []
    bin_rows = []
    for index in range(len(edges) - 1):
        n_hard = int(np.count_nonzero(hard_bin == index))
        candidates = signal_indices[signal_bin == index]
        n_selected = min(n_hard, len(candidates))
        if n_selected:
            selected.extend(rng.choice(candidates, size=n_selected, replace=False).tolist())
        bin_rows.append({
            'bin': index,
            'dt_min_ns': float(edges[index] * 50.0),
            'dt_max_ns': float(edges[index + 1] * 50.0),
            'hard_antiP': n_hard,
            'available_antiD': int(len(candidates)),
            'selected_antiD': int(n_selected),
        })

    matched = np.zeros(len(labels), dtype=bool)
    matched[np.asarray(selected, dtype=np.int64)] = True
    return matched, {
        'requested_hard_antiP': int(hard_mask.sum()),
        'matched_antiD': int(matched.sum()),
        'bins': bin_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument(
        '--evaluation-dir', type=Path, required=True,
        help='directory produced by evaluate_aohba_split_cache.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--signal-efficiencies', type=float, nargs='+',
                        default=[0.95, 0.98])
    parser.add_argument('--top-features', type=int, default=15)
    parser.add_argument('--tof-match-bins', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260821)
    args = parser.parse_args()

    if not args.signal_efficiencies or any(not 0.0 < value < 1.0 for value in args.signal_efficiencies):
        raise ValueError('--signal-efficiencies must be in (0, 1)')
    if args.top_features < 1:
        raise ValueError('--top-features must be positive')
    if args.tof_match_bins < 2:
        raise ValueError('--tof-match-bins must be at least 2')

    labels_path = args.evaluation_dir / 'labels.npy'
    scores_path = args.evaluation_dir / 'scores.npy'
    if not labels_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            'evaluation directory must contain labels.npy and scores.npy; '
            f'got {args.evaluation_dir}')
    labels = np.load(labels_path)
    scores = np.load(scores_path)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError(f'bad labels/scores shapes: {labels.shape}, {scores.shape}')

    graphs = load_test_graphs(args.cache_dir)
    if len(graphs) != len(labels):
        raise RuntimeError(
            f'cache has {len(graphs):,} test graphs but evaluation has '
            f'{len(labels):,} labels')
    cache_labels = np.asarray(
        [int(graph.y.view(-1)[0]) for graph in graphs], dtype=np.int64)
    if not np.array_equal(cache_labels, labels.astype(np.int64)):
        raise RuntimeError(
            'cached test labels are not aligned with the saved evaluation labels')

    feature_matrix, mc_beta, feature_names = graph_feature_matrix(graphs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'cache_dir': str(args.cache_dir),
        'evaluation_dir': str(args.evaluation_dir),
        'n_events': int(len(labels)),
        'n_antiP': int(np.count_nonzero(labels == 0)),
        'n_antiD': int(np.count_nonzero(labels == 1)),
        'note': (
            'All ranked quantities are existing TreeRec graph observables. '
            'mc_beta below is diagnostic only and is not a candidate input.'),
        'thresholds': [],
    }
    rng = np.random.default_rng(args.seed)
    tof_dt_index = BASE_FEATURE_NAMES.index('tof_dt_scaled')

    for target in args.signal_efficiencies:
        threshold = rejection_threshold(labels, scores, target)
        anti_p = labels == 0
        hard_mask = anti_p & (scores >= threshold['threshold'])
        rejected_mask = anti_p & ~hard_mask
        if not hard_mask.any() or not rejected_mask.any():
            raise RuntimeError(f'empty antiP comparison group at target={target}')

        comparisons = feature_comparison(
            feature_matrix, feature_names, hard_mask, rejected_mask)
        matched_signal_mask, matching = tof_dt_matched_signal_mask(
            labels, feature_matrix, hard_mask, tof_dt_index,
            args.tof_match_bins, rng)
        signal_comparisons = feature_comparison(
            feature_matrix, feature_names, hard_mask, matched_signal_mask)
        row = {
            **threshold,
            'hard_antiP_count': int(hard_mask.sum()),
            'rejected_antiP_count': int(rejected_mask.sum()),
            'hard_score': summarize(scores[hard_mask], 'hard').__dict__,
            'rejected_score': summarize(scores[rejected_mask], 'rejected').__dict__,
            'mc_beta_diagnostic': {
                'hard': summarize(mc_beta[hard_mask], 'hard').__dict__,
                'rejected': summarize(mc_beta[rejected_mask], 'rejected').__dict__,
            },
            'feature_comparison': comparisons,
            'tof_dt_matched_antiD_comparison': {
                **matching,
                'feature_comparison': signal_comparisons,
            },
        }
        report['thresholds'].append(row)

        print(f'\nSignal efficiency target: {target:.2f}')
        print(
            f"threshold={threshold['threshold']:.6g} "
            f"actual_eff={threshold['actual_signal_efficiency']:.6f} "
            f"FPR={threshold['fpr']:.8g} "
            f"hard antiP={hard_mask.sum():,}/{anti_p.sum():,}")
        print('Top TreeRec feature differences (hard antiP minus rejected antiP):')
        for feature in comparisons[:args.top_features]:
            print(
                f"  {feature['feature']:34s} "
                f"effect={feature['standardized_mean_difference']:+.3f} "
                f"median={feature['hard_median']:.5g} vs "
                f"{feature['rejected_median']:.5g}")
        print(
            'MC beta diagnostic only: '
            f"median={row['mc_beta_diagnostic']['hard']['median']:.5f} vs "
            f"{row['mc_beta_diagnostic']['rejected']['median']:.5f}")
        print(
            'Top differences after matching hard antiP and antiD in TOF dt: '
            f"matched antiD={matched_signal_mask.sum():,}/{hard_mask.sum():,}")
        for feature in signal_comparisons[:args.top_features]:
            print(
                f"  {feature['feature']:34s} "
                f"effect={feature['standardized_mean_difference']:+.3f} "
                f"median={feature['hard_median']:.5g} vs "
                f"{feature['rejected_median']:.5g}")

        suffix = f"eff_{target:.2f}".replace('.', 'p')
        with (args.output_dir / f'feature_comparison_{suffix}.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)
        with (args.output_dir / f'hard_antip_vs_tofmatched_antid_{suffix}.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=signal_comparisons[0].keys())
            writer.writeheader()
            writer.writerows(signal_comparisons)

    with (args.output_dir / 'residual_report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f'\nsaved: {args.output_dir}')


if __name__ == '__main__':
    main()
