"""Evaluate Nakagami onlyPrimary three-input reproduction model."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

NAKAGAMI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NAKAGAMI_ROOT))

from data_parse.three_input_dataset import ThreeInputDataset
from models.nakagami_three_input import NakagamiThreeInputNet


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    labels_list, logits_list, probs_list = [], [], []
    for voxel, tof_paddle, tof_primary, label in tqdm(loader, desc='eval', dynamic_ncols=True):
        voxel = voxel.to(device, non_blocking=True)
        tof_paddle = tof_paddle.to(device, non_blocking=True)
        tof_primary = tof_primary.to(device, non_blocking=True)
        logits = model(voxel, tof_paddle, tof_primary)
        probs = torch.sigmoid(logits)
        labels_list.append(label.numpy())
        logits_list.append(logits.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
    return np.concatenate(labels_list), np.concatenate(logits_list), np.concatenate(probs_list)


def print_metrics(name, labels, probs):
    preds = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"\n{'=' * 70}")
    print(f'Model: {name}')
    print(f'  Accuracy : {accuracy_score(labels, preds):.6f}')
    print(f'  Precision: {precision_score(labels, preds, zero_division=0):.6f}')
    print(f'  Recall   : {recall_score(labels, preds, zero_division=0):.6f}')
    print(f'  F1 Score : {f1_score(labels, preds, zero_division=0):.6f}')
    print(f'  ROC AUC  : {roc_auc_score(labels, probs):.6f}')
    print(f'  Confusion: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}')


def print_rejection_at_efficiency(labels, probs, targets=(0.50, 0.80, 0.90, 0.95, 0.98, 0.99)):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    n_background = int((labels == 0).sum())
    print(f"\n{'=' * 70}")
    print('Background Rejection = 1 / FPR')
    print(f'Background events: {n_background:,}')
    print(f"{'Signal Eff':>12}  {'Threshold':>12}  {'FPR':>12}  {'Rejection':>14}")
    print('-' * 58)
    for target in targets:
        idx = int(np.argmin(np.abs(tpr - target)))
        f = float(fpr[idx])
        thr = float(thresholds[idx])
        rej_str = f'>{n_background:.2e}' if f == 0 else f'{1.0 / f:.3e}'
        print(f'{target:12.2f}  {thr:12.5g}  {f:12.3e}  {rej_str:>14}')


def plot_rejection_curve(labels, probs, save_path, label):
    fpr, tpr, _ = roc_curve(labels, probs)
    n_background = max(1, int((labels == 0).sum()))
    fpr_safe = np.where(fpr == 0, 1.0 / n_background, fpr)
    plt.figure(figsize=(7, 6))
    plt.semilogy(tpr, 1.0 / fpr_safe, label=label)
    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.xlim(0.5, 1.0)
    plt.ylim(1, 1e6)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend()

    def log_fmt(x, _):
        if x == 1:
            return '1'
        if x == 10:
            return '10'
        return f'$10^{{{int(round(np.log10(x)))}}}$'

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_fmt))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Curve saved: {save_path}')


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    val_set = ThreeInputDataset(Path(args.data_dir) / 'val_onlyprimary_4M',
                                normalize=args.normalize,
                                max_events=args.max_val_events)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    model = NakagamiThreeInputNet(dropout_res=0.1, dropout_dense=0.2).to(device)
    state = torch.load(args.model_path, map_location=device)
    clean_state = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    print(f'loaded model: {args.model_path}')
    print(f'validation events: {len(val_set):,}')

    labels, logits, probs = run_inference(model, val_loader, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{args.tag}_labels.npy', labels)
    np.save(out_dir / f'{args.tag}_logits.npy', logits)
    np.save(out_dir / f'{args.tag}_probs.npy', probs)
    print(f'inference arrays saved to: {out_dir}')
    print_metrics(args.tag, labels, probs)
    print_rejection_at_efficiency(labels, probs)
    plot_rejection_curve(labels, probs, out_dir / f'rejection_{args.tag}.png', args.tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/mnt/ynakagami3/nakagami_data/data_4M_onlyprimary')
    parser.add_argument('--model-path', default=str(NAKAGAMI_ROOT / 'results' / 'onlyprimary_3input' / 'nakagami_onlyprimary_3input_best.pth'))
    parser.add_argument('--out-dir', default=str(NAKAGAMI_ROOT / 'results' / 'evaluation_onlyprimary_3input'))
    parser.add_argument('--tag', default='Nakagami4M_onlyPrimary_3input_11d')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-val-events', type=int, default=None)
    parser.add_argument('--normalize', action='store_true', help='must match training setting')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
