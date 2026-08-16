"""Audit whether cached TreeRec node feature 6 can separate TOF and Si(Li)."""
import argparse
import json
from pathlib import Path

import torch


def graph_count(path: Path) -> int:
    summary = path.with_suffix('.json')
    if not summary.exists():
        return 0
    with summary.open(encoding='utf-8') as file:
        return int(json.load(file).get('n_graphs', 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--max-graphs-per-shard', type=int, default=None)
    args = parser.parse_args()

    for split in args.splits:
        files = sorted(args.cache_dir.glob(f'{split}_*.pt'))
        if not files:
            raise FileNotFoundError(f'no {split}_*.pt under {args.cache_dir}')

        stats = {
            'graphs': 0,
            'both_types': 0,
            'tof_only': 0,
            'sili_only': 0,
            'all_zero_type': 0,
            'nodes_tof': 0,
            'nodes_sili': 0,
            'nodes_zero': 0,
        }
        label_counts = {0: 0, 1: 0}

        for path in files:
            graphs = torch.load(path, map_location='cpu', weights_only=False)
            if args.max_graphs_per_shard is not None:
                graphs = graphs[:args.max_graphs_per_shard]
            for graph in graphs:
                detector_type = graph.x[:, 6]
                n_tof = int((detector_type < 0).sum())
                n_sili = int((detector_type > 0).sum())
                n_zero = int((detector_type == 0).sum())
                stats['graphs'] += 1
                stats['nodes_tof'] += n_tof
                stats['nodes_sili'] += n_sili
                stats['nodes_zero'] += n_zero
                label_counts[int(graph.y.view(-1)[0])] += 1
                if n_tof and n_sili:
                    stats['both_types'] += 1
                elif n_tof:
                    stats['tof_only'] += 1
                elif n_sili:
                    stats['sili_only'] += 1
                else:
                    stats['all_zero_type'] += 1

        total = max(stats['graphs'], 1)
        print(f'\n[{split}]')
        print(f'files: {len(files)}  declared graphs: {sum(graph_count(p) for p in files):,}')
        print(f'audited graphs: {stats["graphs"]:,}  labels: {label_counts}')
        print(
            'both TOF+Si(Li): '
            f'{stats["both_types"]:,} ({stats["both_types"] / total:.4%})')
        print(f'TOF-only      : {stats["tof_only"]:,}')
        print(f'Si(Li)-only   : {stats["sili_only"]:,}')
        print(f'all-zero type : {stats["all_zero_type"]:,}')
        print(
            'nodes (TOF / Si(Li) / zero): '
            f'{stats["nodes_tof"]:,} / {stats["nodes_sili"]:,} / {stats["nodes_zero"]:,}')


if __name__ == '__main__':
    main()
