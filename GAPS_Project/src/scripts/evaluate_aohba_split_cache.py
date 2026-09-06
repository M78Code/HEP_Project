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

from GAPS_Project.src.models.gravnet import (
    ClusterVertexTokenClassifier,
    GravNetAttentionClassifier,
    GravNetClusterTokenClassifier,
    DetectorAwareGravNetClassifier,
    GravNetClassifier,
    GravNetPhysicsEdgeClassifier,
    GravNetMultiTaskClassifier,
    GravNetSoftObjectClassifier,
)
from GAPS_Project.src.models.gravnet_tof import GravNetTOFClassifier
from GAPS_Project.src.models.dgcnn import DGCNNClassifier
from GAPS_Project.src.models.tree_rec_features import (
    HIT_TOPOLOGY_FEATURE_DIM,
    INPUT_ABLATION_CHOICES,
    TRACK_STAR_FEATURE_DIM,
    apply_input_ablation,
    append_hit_topology,
    append_track_star,
    build_base_graph_feat,
    cluster_vertex_token_inputs,
    fit_mc_beta_normalizer,
    load_graph_feature_normalizer,
    normalize_base_graph_feat,
    reconstruct_tof_beta,
)


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


def build_graph_feat(
        batch, use_mc_beta=False, use_tof_beta=False,
        use_hit_topology=False,
        use_track_star=False,
        graph_feature_mean=None, graph_feature_std=None):
    if use_mc_beta and use_tof_beta:
        raise ValueError('MC beta and TOF-reconstructed beta are mutually exclusive')
    graph_feat = build_base_graph_feat(batch)
    if (graph_feature_mean is None) != (graph_feature_std is None):
        raise ValueError('graph-feature mean/std must be provided together')
    if graph_feature_mean is not None:
        graph_feat = normalize_base_graph_feat(
            graph_feat, graph_feature_mean, graph_feature_std)
    if use_mc_beta:
        if not hasattr(batch, 'mc_beta'):
            raise ValueError('--use-mc-beta requires mc_beta in graph cache')
        graph_feat = torch.cat(
            [graph_feat, batch.mc_beta.view(-1, 1).float()],
            dim=1,
        )
    elif use_tof_beta:
        graph_feat = torch.cat([
            graph_feat,
            reconstruct_tof_beta(batch.tof_feat.view(-1, 11).float()),
        ], dim=1)
    if use_hit_topology:
        graph_feat = append_hit_topology(graph_feat, batch)
    if use_track_star:
        graph_feat = append_track_star(graph_feat, batch)
    return graph_feat


@torch.no_grad()
def infer(
        model, loader, device, total_batches, model_name,
        tof_mode='normal', use_mc_beta=False, use_tof_beta=False,
        use_hit_topology=False,
        use_track_star=False,
        multi_task_beta=False,
        beta_target_mean=None, beta_target_std=None,
        graph_feature_mean=None, graph_feature_std=None,
        input_ablation='full'):
    labels, scores, betas, predicted_betas = [], [], [], []
    model.eval()
    for batch in tqdm(loader, total=total_batches, desc='test', dynamic_ncols=True):
        batch = batch.to(device)
        graph_feat = build_graph_feat(
            batch,
            use_mc_beta=use_mc_beta,
            use_tof_beta=use_tof_beta,
            use_hit_topology=use_hit_topology,
            use_track_star=use_track_star,
            graph_feature_mean=graph_feature_mean,
            graph_feature_std=graph_feature_std,
        )
        node_features, graph_feat = apply_input_ablation(
            batch.x, graph_feat, input_ablation)
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

            model_output = model(
                node_features,
                batch.edge_index,
                batch.batch,
                graph_feat=graph_feat,
                tof_paddle_energy=tof_paddle_energy,
            )
        elif model_name in ('gravnet_cluster_tokens', 'cluster_tokens_only'):
            vertex_token, prong_tokens, prong_mask = cluster_vertex_token_inputs(batch)
            model_output = model(
                node_features, batch.edge_index, batch.batch, graph_feat=graph_feat,
                vertex_token=vertex_token, prong_tokens=prong_tokens,
                prong_mask=prong_mask)
        elif model_name == 'gravnet_physics_edges':
            if not hasattr(batch, 'edge_attr'):
                raise ValueError(
                    'gravnet_physics_edges requires cached edge_attr')
            model_output = model(
                node_features, batch.edge_index, batch.batch, graph_feat=graph_feat,
                edge_attr=batch.edge_attr)
        elif model_name == 'gravnet_soft_objects':
            logits, _ = model(
                node_features, batch.edge_index, batch.batch, graph_feat=graph_feat)
            model_output = logits
        else:
            model_output = model(
                node_features, batch.edge_index, batch.batch, graph_feat=graph_feat)
        if multi_task_beta:
            logits, beta_prediction = model_output
            predicted_betas.append(
                (beta_prediction * beta_target_std + beta_target_mean).cpu().numpy())
        else:
            logits = model_output
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels.append(batch.y.view(-1).cpu().numpy())
        scores.append(probs.cpu().numpy())
        if hasattr(batch, 'mc_beta'):
            betas.append(batch.mc_beta.view(-1).cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(betas) if betas else None,
        np.concatenate(predicted_betas) if predicted_betas else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dim for DGCNN')
    parser.add_argument(
        '--model',
        choices=['gravnet', 'gravnet_tof', 'gravnet_detector', 'gravnet_attention', 'gravnet_cluster_tokens', 'cluster_tokens_only', 'gravnet_physics_edges', 'gravnet_soft_objects', 'dgcnn'],
                        default='gravnet')
    parser.add_argument(
        '--tof-mode',
        choices=['normal', 'zero', 'shuffle'],
        default='normal',
        help='TOF172 ablation mode',
    )
    parser.add_argument(
        '--use-mc-beta',
        action='store_true',
        help='append TreeMc primary beta to graph-level features',
    )
    parser.add_argument(
        '--use-tof-beta',
        action='store_true',
        help='append beta reconstructed from TreeRec TOF hits and a validity mask',
    )
    parser.add_argument(
        '--use-hit-topology', action='store_true',
        help='append six precomputed TreeRec hit-level topology summaries',
    )
    parser.add_argument(
        '--use-track-star', action='store_true',
        help='append four precomputed TreeRec track/star geometry candidates',
    )
    parser.add_argument('--multi-task-beta', action='store_true',
                        help='evaluate a joint classification and beta-regression model')
    parser.add_argument('--classify-with-predicted-beta', action='store_true',
                        help='classification head consumes the model-predicted beta; requires --multi-task-beta')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--gravnet-normalization', choices=['batch', 'layer'], default='batch',
        help='must match the normalization used during GravNet training',
    )
    parser.add_argument(
        '--graph-feature-normalizer', type=Path, default=None,
        help='same train-only JSON used for training',
    )
    parser.add_argument(
        '--input-ablation', choices=INPUT_ABLATION_CHOICES, default='full',
        help='must match the controlled TreeRec input mask used for training',
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model != 'gravnet_tof' and args.tof_mode != 'normal':
        raise ValueError(
            '--tof-mode zero/shuffle is only valid with --model gravnet_tof'
        )
    if args.use_mc_beta and args.use_tof_beta:
        raise ValueError('--use-mc-beta and --use-tof-beta cannot be used together')
    if args.multi_task_beta and args.use_mc_beta:
        raise ValueError('--multi-task-beta cannot be combined with --use-mc-beta')
    if args.classify_with_predicted_beta and not args.multi_task_beta:
        raise ValueError(
            '--classify-with-predicted-beta requires --multi-task-beta')
    if args.multi_task_beta and args.model in ('gravnet_cluster_tokens', 'cluster_tokens_only'):
        raise ValueError('cluster-token models are classification-only in this A/B')
    if args.input_ablation != 'full':
        if args.model != 'gravnet':
            raise ValueError('--input-ablation currently supports --model gravnet only')
        if any((args.use_mc_beta, args.use_tof_beta, args.use_hit_topology,
                args.use_track_star, args.multi_task_beta)):
            raise ValueError(
                '--input-ablation must be evaluated without optional input branches')

    files = sorted(args.cache_dir.glob('test_*.pt'))
    if not files:
        raise FileNotFoundError(f'no test_*.pt under {args.cache_dir}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph_feature_mean = None
    graph_feature_std = None
    if args.graph_feature_normalizer is not None:
        graph_feature_mean, graph_feature_std = load_graph_feature_normalizer(
            args.graph_feature_normalizer)
        graph_feature_mean = graph_feature_mean.to(device)
        graph_feature_std = graph_feature_std.to(device)
    dataset = ShardedGraphDataset(files)
    n_events = dataset.approx_len()
    n_batches = (n_events + args.batch_size - 1) // args.batch_size
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    graph_feat_dim = 45 + (2 if args.use_tof_beta else (1 if args.use_mc_beta else 0))
    if args.use_hit_topology:
        graph_feat_dim += HIT_TOPOLOGY_FEATURE_DIM
    if args.use_track_star:
        graph_feat_dim += TRACK_STAR_FEATURE_DIM
    if args.input_ablation == 'node_only':
        graph_feat_dim = 0

    beta_target_mean = None
    beta_target_std = None
    if args.multi_task_beta:
        if args.model != 'gravnet':
            raise ValueError('--multi-task-beta currently supports --model gravnet only')
        train_files = sorted(args.cache_dir.glob('train_*.pt'))
        if not train_files:
            raise FileNotFoundError('beta multi-task evaluation requires train_*.pt in cache')
        beta_target_mean, beta_target_std, beta_target_events = \
            fit_mc_beta_normalizer(train_files)
    if args.multi_task_beta:
        model = GravNetMultiTaskClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim,
            num_blocks=6,
            classify_with_predicted_beta=args.classify_with_predicted_beta)
    elif args.model == 'gravnet_tof':
        model = GravNetTOFClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'gravnet_detector':
        model = DetectorAwareGravNetClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'gravnet_attention':
        model = GravNetAttentionClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'gravnet_cluster_tokens':
        model = GravNetClusterTokenClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'gravnet_physics_edges':
        model = GravNetPhysicsEdgeClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'gravnet_soft_objects':
        model = GravNetSoftObjectClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim, num_blocks=6)
    elif args.model == 'cluster_tokens_only':
        model = ClusterVertexTokenClassifier()
    elif args.model == 'dgcnn':
        model = DGCNNClassifier(
            in_channels=8, hidden_dim=args.hidden_dim, k=8,
            graph_feat_dim=graph_feat_dim)
    else:
        model = GravNetClassifier(
            in_channels=8, hidden_dim=128, graph_feat_dim=graph_feat_dim,
            num_blocks=6, normalization=args.gravnet_normalization)
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
    print(f'MC beta    : {"enabled" if args.use_mc_beta else "disabled"}')
    print(f'TOF beta   : {"enabled" if args.use_tof_beta else "disabled"}')
    print(f'hit topology: {"enabled" if args.use_hit_topology else "disabled"}')
    print(f'track/star : {"enabled" if args.use_track_star else "disabled"}')
    print(
        'cluster/vertex tokens: '
        f'{"enabled" if args.model in ("gravnet_cluster_tokens", "cluster_tokens_only") else "disabled"}')
    print(
        'explicit physics edges: '
        f'{"enabled" if args.model == "gravnet_physics_edges" else "disabled"}')
    print(
        'soft object queries: '
        f'{"enabled" if args.model == "gravnet_soft_objects" else "disabled"}')
    if args.multi_task_beta:
        print(
            'beta multi-task: enabled '
            f'| target=(beta-{beta_target_mean:.6g})/{beta_target_std:.6g} '
            f'| train targets={beta_target_events:,} '
            f'| classifier beta input={args.classify_with_predicted_beta}')
    else:
        print('beta multi-task: disabled')
    print(f'graph norm : {args.graph_feature_normalizer or "disabled"}')
    print(f'graph feat : {graph_feat_dim}')
    print(f'GravNet norm: {args.gravnet_normalization}')
    print(f'input ablation: {args.input_ablation}')

    labels, scores, betas, predicted_betas = infer(
        model,
        loader,
        device,
        n_batches,
        args.model,
        tof_mode=args.tof_mode,
        use_mc_beta=args.use_mc_beta,
        use_tof_beta=args.use_tof_beta,
        use_hit_topology=args.use_hit_topology,
        use_track_star=args.use_track_star,
        multi_task_beta=args.multi_task_beta,
        beta_target_mean=beta_target_mean,
        beta_target_std=beta_target_std,
        graph_feature_mean=graph_feature_mean,
        graph_feature_std=graph_feature_std,
        input_ablation=args.input_ablation,
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
        'use_mc_beta': bool(args.use_mc_beta),
        'use_tof_beta': bool(args.use_tof_beta),
        'use_hit_topology': bool(args.use_hit_topology),
        'use_track_star': bool(args.use_track_star),
        'use_cluster_vertex_tokens': bool(
            args.model in ('gravnet_cluster_tokens', 'cluster_tokens_only')),
        'use_physics_edges': bool(args.model == 'gravnet_physics_edges'),
        'use_soft_objects': bool(args.model == 'gravnet_soft_objects'),
        'multi_task_beta': bool(args.multi_task_beta),
        'classify_with_predicted_beta': bool(
            args.classify_with_predicted_beta),
        'graph_feat_dim': int(graph_feat_dim),
        'input_ablation': args.input_ablation,
    }
    if args.multi_task_beta:
        if betas is None or predicted_betas is None:
            raise ValueError('beta multi-task evaluation requires mc_beta metadata')
        beta_error = predicted_betas - betas
        metrics['beta_regression'] = {
            'mae': float(np.mean(np.abs(beta_error))),
            'rmse': float(np.sqrt(np.mean(np.square(beta_error)))),
            'pearson_correlation': float(np.corrcoef(betas, predicted_betas)[0, 1]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'labels.npy', labels)
    np.save(args.output_dir / 'scores.npy', scores)
    if betas is not None:
        np.save(args.output_dir / 'betas.npy', betas)
    if predicted_betas is not None:
        np.save(args.output_dir / 'predicted_betas.npy', predicted_betas)
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
    if args.multi_task_beta:
        beta_metrics = metrics['beta_regression']
        print(f'beta MAE: {beta_metrics["mae"]:.6f}')
        print(f'beta RMSE: {beta_metrics["rmse"]:.6f}')
        print(f'beta Pearson: {beta_metrics["pearson_correlation"]:.6f}')
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
