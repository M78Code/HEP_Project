"""Analyze saved classifier scores in beta intervals."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def rejection_at_efficiency(labels, scores, target):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    idx = candidates[np.argmin(fpr[candidates])]
    return float('inf') if fpr[idx] == 0 else float(1.0 / fpr[idx])


def evaluate_bin(labels, scores):
    counts = {
        str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
    }
    result = {
        'n_events': int(len(labels)),
        'label_counts': counts,
        'accuracy': float(accuracy_score(labels, scores >= 0.5)),
    }
    if len(np.unique(labels)) == 2:
        result['auc'] = float(roc_auc_score(labels, scores))
        result['rejection_0.90'] = rejection_at_efficiency(
            labels, scores, 0.90)
        result['rejection_0.95'] = rejection_at_efficiency(
            labels, scores, 0.95)
        result['rejection_0.98'] = rejection_at_efficiency(
            labels, scores, 0.98)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument(
        '--bins', type=float, nargs='+',
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    )
    args = parser.parse_args()

    labels = np.load(args.result_dir / 'labels.npy')
    scores = np.load(args.result_dir / 'scores.npy')
    betas = np.load(args.result_dir / 'betas.npy')
    if not (len(labels) == len(scores) == len(betas)):
        raise ValueError('labels/scores/betas length mismatch')

    rows = []
    print(
        f'{"beta range":>14} {"events":>9} {"antiP":>9} {"antiD":>9} '
        f'{"AUC":>9} {"Rej@.90":>10} {"Rej@.95":>10} {"Rej@.98":>10}'
    )
    for low, high in zip(args.bins[:-1], args.bins[1:]):
        is_last = high == args.bins[-1]
        mask = (betas >= low) & ((betas <= high) if is_last else (betas < high))
        row = {
            'beta_low': low,
            'beta_high': high,
            **evaluate_bin(labels[mask], scores[mask]),
        }
        rows.append(row)
        counts = row['label_counts']
        print(
            f'[{low:.2f},{high:.2f}{"]" if is_last else ")":>1} '
            f'{row["n_events"]:9,d} {counts.get("0", 0):9,d} '
            f'{counts.get("1", 0):9,d} {row.get("auc", float("nan")):9.5f} '
            f'{row.get("rejection_0.90", float("nan")):10.2f} '
            f'{row.get("rejection_0.95", float("nan")):10.2f} '
            f'{row.get("rejection_0.98", float("nan")):10.2f}'
        )

    with open(args.result_dir / 'metrics_by_beta.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
