"""Evaluate GravNet on sharded train/val/test graph caches."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.models.gravnet import GravNetClassifier
from src.models.gravnet_tof import GravNetTOFClassifier
from src.models.dgcnn import DGCNNClassifier


class ShardedGraphDataset(IterableDataset):
    def __init__(self, files):
        self.files = list(files)

    def __iter__(self):
        for path in self.files:
            for data in torch.load(path, map_location='cpu', weights_only=False):
                yield data

    def approx_len(self):
        total = 0
        for path in self.files:
            summary = path.with_suffix('.json')
            if summary.exists():
                with open(summary, encoding='utf-8') as f:
                    total += int(json.load(f).get('n_graphs', 0))
        return total


def rejection_at_efficiency(labels, scores, target):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    idx = candidates[np.argmin(fpr[candidates])]
    rejection = float('inf') if fpr[idx] == 0 else 1.0 / fpr[idx]
    return {
        'target_efficiency': target,
        'actual_efficiency': float(tpr[idx]),
        'fpr': float(fpr[idx]),
        'rejection': float(rejection),
        'threshold': float(thresholds[idx]),
    }


@torch.no_grad()
def infer(model, loader, device, total_batches, model_name, tof_mode='normal'):
    labels, scores, betas = [], [], []
    model.eval()
    for batch in tqdm(loader, total=total_batches, desc='test', dynamic_ncols=True):
        batch = batch.to(device)
        graph_feat = torch.cat([
            batch.n_hits.view(-1, 1),
            batch.total_energy.view(-1, 1),
            batch.sili_profile.view(-1, 16),
            batch.tof_profile.view(-1, 16),
            batch.tof_feat.view(-1, 11),
        ], dim=1)
        if model_name == 'gravnet_tof':
            tof_paddle_energy = batch.tof_paddle_energy.view(-1, 172)

            if tof_mode == 'zero':
                tof_paddle_energy = torch.zeros_like(tof_paddle_energy)
            elif tof_mode == 'shuffle':
                permutation = torch.randperm(
                    tof_paddle_energy.size(0),
                    device=tof_paddle_energy.device,
                )
                tof_paddle_energy = tof_paddle_energy[permutation]

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                graph_feat=graph_feat,
                tof_paddle_energy=tof_paddle_energy,
            )
        else:
            logits = model(
                batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels.append(batch.y.view(-1).cpu().numpy())
        scores.append(probs.cpu().numpy())
        if hasattr(batch, 'mc_beta'):
            betas.append(batch.mc_beta.view(-1).cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(betas) if betas else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dim for DGCNN')
    parser.add_argument('--model', choices=['gravnet', 'gravnet_tof', 'dgcnn'],
                        default='gravnet')
    parser.add_argument(
        '--tof-mode',
        choices=['normal', 'zero', 'shuffle'],
        default='normal',
        help='TOF172 ablation mode',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model != 'gravnet_tof' and args.tof_mode != 'normal':
        raise ValueError(
            '--tof-mode zero/shuffle is only valid with --model gravnet_tof'
        )

    files = sorted(args.cache_dir.glob('test_*.pt'))
    if not files:
        raise FileNotFoundError(f'no test_*.pt under {args.cache_dir}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = ShardedGraphDataset(files)
    n_events = dataset.approx_len()
    n_batches = (n_events + args.batch_size - 1) // args.batch_size
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    if args.model == 'gravnet_tof':
        model = GravNetTOFClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=45, num_blocks=6)
    elif args.model == 'dgcnn':
        model = DGCNNClassifier(
            in_channels=8, hidden_dim=args.hidden_dim, k=8, graph_feat_dim=45)
    else:
        model = GravNetClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=45, num_blocks=6)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    state = {
        key.replace('_orig_mod.', '').replace('module.', ''): value
        for key, value in state.items()
    }
    model.load_state_dict(state)
    model.to(device)

    print(f'device     : {device}')
    print(f'test files : {len(files)}')
    print(f'test events: {n_events:,}')
    print(f'model      : {args.model_path}')
    print(f'TOF mode   : {args.tof_mode}')

    labels, scores, betas = infer(
        model,
        loader,
        device,
        n_batches,
        args.model,
        tof_mode=args.tof_mode,
    )

    predictions = (scores >= 0.5).astype(np.int64)
    metrics = {
        'n_events': int(len(labels)),
        'label_counts': {
            str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
        },
        'accuracy': float(accuracy_score(labels, predictions)),
        'auc': float(roc_auc_score(labels, scores)),
        'rejection': [
            rejection_at_efficiency(labels, scores, target)
            for target in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
        ],
        'tof_mode': args.tof_mode,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'labels.npy', labels)
    np.save(args.output_dir / 'scores.npy', scores)
    if betas is not None:
        np.save(args.output_dir / 'betas.npy', betas)
    with open(args.output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(7, 6))
    plt.plot(tpr, 1.0 / np.maximum(fpr, 1.0 / (labels == 0).sum()),
             label=(
                 f'GravNetTOF ({args.tof_mode}, '
                 f'AUC={metrics["auc"]:.4f})'
             ))
    plt.yscale('log')
    plt.xlim(0.5, 1.0)
    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / 'rejection_curve.png', dpi=300)
    plt.close()

    print(f'accuracy: {metrics["accuracy"]:.6f}')
    print(f'AUC     : {metrics["auc"]:.6f}')
    for row in metrics['rejection']:
        print(
            f'Rej@{row["target_efficiency"]:.2f}: '
            f'{row["rejection"]:.3f} '
            f'(actual eff={row["actual_efficiency"]:.6f}, '
            f'FPR={row["fpr"]:.8g})'
        )
    print(f'output: {args.output_dir}')


if __name__ == '__main__':
    main()
