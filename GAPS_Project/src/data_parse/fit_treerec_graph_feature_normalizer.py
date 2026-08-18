#!/usr/bin/env python3
"""Fit a train-only global-log normalizer for the 45-D TreeRec graph summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from GAPS_Project.src.models.tree_rec_features import (
    BASE_GRAPH_FEATURE_DIM,
    build_base_graph_feat,
    transform_base_graph_feat,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.cache_dir.glob('train_*.pt'))
    if not shards:
        raise FileNotFoundError(f'no train_*.pt under {args.cache_dir}')
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite: {args.output}')

    sum_x = torch.zeros(BASE_GRAPH_FEATURE_DIM, dtype=torch.float64)
    sum_x2 = torch.zeros_like(sum_x)
    events = 0

    for i, shard in enumerate(shards, start=1):
        graphs = torch.load(shard, map_location='cpu', weights_only=False)
        rows = []
        for graph in graphs:
            rows.append(build_base_graph_feat(graph))
        features = transform_base_graph_feat(torch.cat(rows, dim=0)).double()
        sum_x += features.sum(dim=0)
        sum_x2 += features.square().sum(dim=0)
        events += features.size(0)
        print(f'[{i:02d}/{len(shards):02d}] {shard.name}: {features.size(0):,} events', flush=True)

    mean = sum_x / events
    variance = (sum_x2 / events - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-6)
    payload = {
        'transform': 'log1p_first_38_then_global_zscore',
        'graph_feature_dim': BASE_GRAPH_FEATURE_DIM,
        'log1p_indices': list(range(38)),
        'events': events,
        'source_cache_dir': str(args.cache_dir.resolve()),
        'mean': mean.float().tolist(),
        'std': std.float().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'saved: {args.output}')


if __name__ == '__main__':
    main()
