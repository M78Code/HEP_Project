"""Compare saved classifier scores on beta-matched test samples."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


DEFAULT_BINS = [0.25, 0.30, 0.35, 0.40, 0.50]
TARGET_EFFICIENCIES = [0.90, 0.95, 0.98]


def parse_result(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError(
            '--result must use NAME=RESULT_DIR format')
    name, path = value.split('=', 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(
            '--result must use NAME=RESULT_DIR format')
    return name, Path(path)


def rejection_at_efficiency(labels, scores, target):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    idx = candidates[np.argmin(fpr[candidates])]
    return float('inf') if fpr[idx] == 0 else float(1.0 / fpr[idx])


def evaluate(labels, scores):
    metrics = {
        'accuracy': float(accuracy_score(labels, scores >= 0.5)),
        'auc': float(roc_auc_score(labels, scores)),
    }
    for target in TARGET_EFFICIENCIES:
        metrics[f'rejection_{target:.2f}'] = rejection_at_efficiency(
            labels, scores, target)
    return metrics


def make_matched_indices(labels, betas, bins, rng):
    matched = []
    bin_indices = []

    for bin_index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        is_last = bin_index == len(bins) - 2
        beta_mask = (
            (betas >= low)
            & ((betas <= high) if is_last else (betas < high))
        )
        anti_p = np.flatnonzero(beta_mask & (labels == 0))
        anti_d = np.flatnonzero(beta_mask & (labels == 1))
        n_per_class = min(len(anti_p), len(anti_d))
        if n_per_class == 0:
            continue

        selected = np.concatenate([
            rng.choice(anti_p, n_per_class, replace=False),
            rng.choice(anti_d, n_per_class, replace=False),
        ])
        rng.shuffle(selected)
        matched.append(selected)
        bin_indices.append({
            'beta_low': low,
            'beta_high': high,
            'n_per_class': n_per_class,
            'indices': selected,
        })

    if not matched:
        raise ValueError('no beta bin contains both antiP and antiD')
    return np.concatenate(matched), bin_indices


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {'mean': float('inf'), 'std': 0.0}
    return {
        'mean': float(finite.mean()),
        'std': float(finite.std(ddof=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result',
        action='append',
        type=parse_result,
        required=True,
        metavar='NAME=RESULT_DIR',
        help='saved evaluation directory; repeat for each model',
    )
    parser.add_argument('--bins', type=float, nargs='+', default=DEFAULT_BINS)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/beta_matched_comparison.json'),
    )
    args = parser.parse_args()

    if len(args.bins) < 2 or np.any(np.diff(args.bins) <= 0):
        raise ValueError('--bins must be strictly increasing')
    if args.repeats < 1:
        raise ValueError('--repeats must be at least 1')

    result_scores = {}
    reference_labels = None
    reference_betas = None

    for name, result_dir in args.result:
        labels = np.load(result_dir / 'labels.npy')
        scores = np.load(result_dir / 'scores.npy')
        betas = np.load(result_dir / 'betas.npy')
        if not (len(labels) == len(scores) == len(betas)):
            raise ValueError(f'{name}: labels/scores/betas length mismatch')

        if reference_labels is None:
            reference_labels = labels
            reference_betas = betas
        else:
            if not np.array_equal(labels, reference_labels):
                raise ValueError(f'{name}: labels differ from the first result')
            if not np.array_equal(betas, reference_betas):
                raise ValueError(f'{name}: betas differ from the first result')
        result_scores[name] = scores

    metric_names = [
        'accuracy', 'auc',
        'rejection_0.90', 'rejection_0.95', 'rejection_0.98',
    ]
    collected = {
        name: {
            'overall': {metric: [] for metric in metric_names},
            'bins': {},
        }
        for name in result_scores
    }
    sample_description = None

    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        matched, matched_bins = make_matched_indices(
            reference_labels, reference_betas, args.bins, rng)

        if sample_description is None:
            sample_description = {
                'total_events': int(len(matched)),
                'events_per_class': int(len(matched) // 2),
                'bins': [
                    {
                        'beta_low': row['beta_low'],
                        'beta_high': row['beta_high'],
                        'events_per_class': row['n_per_class'],
                    }
                    for row in matched_bins
                ],
            }

        for name, scores in result_scores.items():
            overall = evaluate(
                reference_labels[matched], scores[matched])
            for metric, value in overall.items():
                collected[name]['overall'][metric].append(value)

            for row in matched_bins:
                key = f'{row["beta_low"]:.2f}-{row["beta_high"]:.2f}'
                collected[name]['bins'].setdefault(
                    key, {metric: [] for metric in metric_names})
                bin_metrics = evaluate(
                    reference_labels[row['indices']],
                    scores[row['indices']],
                )
                for metric, value in bin_metrics.items():
                    collected[name]['bins'][key][metric].append(value)

    output = {
        'seed': args.seed,
        'repeats': args.repeats,
        'sample': sample_description,
        'models': {},
    }
    for name, sections in collected.items():
        output['models'][name] = {
            'overall': {
                metric: summarize(values)
                for metric, values in sections['overall'].items()
            },
            'bins': {
                key: {
                    metric: summarize(values)
                    for metric, values in metrics.items()
                }
                for key, metrics in sections['bins'].items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(
        f'beta-matched sample: {sample_description["total_events"]:,} events '
        f'({sample_description["events_per_class"]:,} per class)')
    print(f'repeats: {args.repeats}, seed: {args.seed}')
    print()
    print(
        f'{"model":<18} {"AUC":>16} {"Rej@.90":>16} '
        f'{"Rej@.95":>16} {"Rej@.98":>16}')
    for name, model_result in output['models'].items():
        overall = model_result['overall']

        def format_metric(key):
            row = overall[key]
            return f'{row["mean"]:.4f}+/-{row["std"]:.4f}'

        print(
            f'{name:<18} {format_metric("auc"):>16} '
            f'{format_metric("rejection_0.90"):>16} '
            f'{format_metric("rejection_0.95"):>16} '
            f'{format_metric("rejection_0.98"):>16}')
    print(f'\noutput: {args.output}')


if __name__ == '__main__':
    main()
