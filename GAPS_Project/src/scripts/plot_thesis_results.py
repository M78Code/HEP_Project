"""Create thesis figures from saved GravNet evaluation arrays."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


COLORS = {
    'antiP': '#d1495b',
    'antiD': '#2878b5',
    'gravnet': '#4c566a',
    'tof172': '#00798c',
    'shuffle': '#e69500',
    'zero': '#9b59b6',
}
DEFAULT_BETA_BINS = [0.25, 0.30, 0.35, 0.40, 0.50]


def load_result(path: Path) -> dict:
    arrays = {
        'labels': np.load(path / 'labels.npy'),
        'scores': np.load(path / 'scores.npy'),
    }
    beta_path = path / 'betas.npy'
    if beta_path.exists():
        arrays['betas'] = np.load(beta_path)
    if len(arrays['labels']) != len(arrays['scores']):
        raise ValueError(f'{path}: labels/scores length mismatch')
    if 'betas' in arrays and len(arrays['labels']) != len(arrays['betas']):
        raise ValueError(f'{path}: labels/betas length mismatch')
    return arrays


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def style_axis(axis):
    axis.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.45)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def plot_score_distribution(gravnet, tof172, output_dir):
    bins = np.linspace(0.0, 1.0, 81)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    panels = [
        ('GravNet', gravnet),
        ('GravNet + TOF172', tof172),
    ]

    for axis, (title, result) in zip(axes, panels):
        labels = result['labels']
        scores = result['scores']
        axis.hist(
            scores[labels == 0], bins=bins, histtype='step',
            linewidth=1.5, color=COLORS['antiP'], label='antiP',
        )
        axis.hist(
            scores[labels == 1], bins=bins, histtype='step',
            linewidth=1.5, color=COLORS['antiD'], label='antiD',
        )
        axis.set_yscale('log')
        axis.set_xlim(0.0, 1.0)
        axis.set_title(title)
        axis.set_xlabel('Anti-deuteron prediction score')
        style_axis(axis)

    axes[0].set_ylabel('Events')
    axes[1].legend(frameon=False, loc='upper center')
    fig.tight_layout()
    save_figure(fig, output_dir, 'score_distribution')


def plot_threshold_example(result, output_dir, threshold):
    labels = result['labels']
    scores = result['scores']
    bins = np.linspace(0.0, 1.0, 81)

    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    axis.hist(
        scores[labels == 0], bins=bins, histtype='step',
        linewidth=1.5, color=COLORS['antiP'], label='antiP',
    )
    axis.hist(
        scores[labels == 1], bins=bins, histtype='step',
        linewidth=1.5, color=COLORS['antiD'], label='antiD',
    )
    axis.axvline(
        threshold, color='#202020', linewidth=1.8,
        label=f'Threshold = {threshold:.2f}',
    )
    axis.annotate(
        'Classified as antiP',
        xy=(threshold * 0.50, 0.93),
        xycoords=('data', 'axes fraction'),
        ha='center',
    )
    axis.annotate(
        'Classified as antiD',
        xy=(threshold + (1.0 - threshold) * 0.50, 0.93),
        xycoords=('data', 'axes fraction'),
        ha='center',
    )
    axis.set_yscale('log')
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel('Anti-deuteron prediction score')
    axis.set_ylabel('Events')
    axis.legend(
        frameon=False,
        loc='lower center',
        ncol=3,
        fontsize=9,
    )
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, 'score_threshold_example')


def rejection_curve(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    n_background = max(1, int((labels == 0).sum()))
    rejection = 1.0 / np.maximum(fpr, 1.0 / n_background)
    return tpr, rejection


def plot_rejection_comparison(results, output_dir):
    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    labels_and_styles = {
        'gravnet': ('GravNet', '-', 2.0),
        'tof172': ('GravNet + TOF172', '-', 2.0),
        'shuffle': ('TOF172 shuffled', '--', 1.6),
        'zero': ('TOF172 zeroed', ':', 1.8),
    }

    for key, result in results.items():
        label, linestyle, linewidth = labels_and_styles[key]
        tpr, rejection = rejection_curve(result['labels'], result['scores'])
        auc = roc_auc_score(result['labels'], result['scores'])
        axis.plot(
            tpr, rejection, color=COLORS[key], linestyle=linestyle,
            linewidth=linewidth, label=f'{label} (AUC={auc:.4f})',
        )

    axis.set_yscale('log')
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(bottom=1.0)
    axis.set_xlabel('Signal efficiency (antiD recall)')
    axis.set_ylabel('Background rejection (1 / FPR)')
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, 'rejection_comparison')


def rejection_at_efficiency(labels, scores, target=0.90):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    index = candidates[np.argmin(fpr[candidates])]
    return np.inf if fpr[index] == 0 else float(1.0 / fpr[index])


def beta_rejections(result, bins):
    labels = result['labels']
    scores = result['scores']
    betas = result.get('betas')
    if betas is None:
        raise ValueError('betas.npy is required for the beta plot')

    values = []
    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        is_last = index == len(bins) - 2
        mask = (
            (betas >= low)
            & ((betas <= high) if is_last else (betas < high))
        )
        if len(np.unique(labels[mask])) != 2:
            values.append(np.nan)
        else:
            values.append(rejection_at_efficiency(
                labels[mask], scores[mask], target=0.90))
    return np.asarray(values)


def plot_rejection_by_beta(gravnet, tof172, output_dir, bins):
    gravnet_values = beta_rejections(gravnet, bins)
    tof_values = beta_rejections(tof172, bins)
    labels = [
        f'{low:.2f}-{high:.2f}'
        for low, high in zip(bins[:-1], bins[1:])
    ]
    positions = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.bar(
        positions - width / 2, gravnet_values, width,
        color=COLORS['gravnet'], label='GravNet',
    )
    axis.bar(
        positions + width / 2, tof_values, width,
        color=COLORS['tof172'], label='GravNet + TOF172',
    )
    axis.set_yscale('log')
    axis.set_xticks(positions, labels)
    axis.set_xlabel(r'$\beta$ interval')
    axis.set_ylabel('Background rejection at 90% signal efficiency')
    axis.legend(frameon=False)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, output_dir, 'rejection_by_beta')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gravnet-dir', type=Path, required=True)
    parser.add_argument('--tof172-dir', type=Path, required=True)
    parser.add_argument('--shuffle-dir', type=Path)
    parser.add_argument('--zero-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--beta-bins', type=float, nargs='+', default=DEFAULT_BETA_BINS)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError('--threshold must be within [0, 1]')
    if len(args.beta_bins) < 2 or np.any(np.diff(args.beta_bins) <= 0):
        raise ValueError('--beta-bins must be strictly increasing')

    results = {
        'gravnet': load_result(args.gravnet_dir),
        'tof172': load_result(args.tof172_dir),
    }
    if args.shuffle_dir is not None:
        results['shuffle'] = load_result(args.shuffle_dir)
    if args.zero_dir is not None:
        results['zero'] = load_result(args.zero_dir)

    reference_labels = results['gravnet']['labels']
    for name, result in results.items():
        if not np.array_equal(result['labels'], reference_labels):
            raise ValueError(f'{name}: labels differ from GravNet result')

    plot_score_distribution(
        results['gravnet'], results['tof172'], args.output_dir)
    plot_threshold_example(
        results['tof172'], args.output_dir, args.threshold)
    plot_rejection_comparison(results, args.output_dir)
    plot_rejection_by_beta(
        results['gravnet'], results['tof172'],
        args.output_dir, args.beta_bins,
    )
    print(f'figures saved under: {args.output_dir}')


if __name__ == '__main__':
    main()
