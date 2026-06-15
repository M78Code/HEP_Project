"""
大場さん新データ (GAPS_Sim_2tof) の TreeMc / TreeRec branch 構造を確認。
- GGeometry はスキップ(uproot 非対応)
- 真の branch 名を把握してから前処理を設計する。
"""
import uproot
from pathlib import Path

ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')


def list_branches(tree, label, max_show=80):
    print(f'\n--- {label}: {tree.num_entries} entries, {len(tree.keys())} branches ---')
    for i, k in enumerate(tree.keys()):
        if i >= max_show:
            print(f'  ... ({len(tree.keys()) - max_show} more)')
            break
        try:
            interp = tree[k].interpretation
            print(f'  [{i:3d}] {k:60s}  {interp}')
        except Exception as e:
            print(f'  [{i:3d}] {k:60s}  (interp err: {e.__class__.__name__})')


def inspect_first_file(particle):
    files = sorted((ROOT_DIR / particle).glob('*.root'))
    print(f'\n========== {particle}: {len(files)} files ==========')
    f = uproot.open(files[0])

    for key in ['TreeMc', 'TreeRec']:
        matches = [k for k in f.keys() if k.startswith(key + ';')]
        if not matches:
            print(f'  {key}: NOT FOUND')
            continue
        latest = sorted(matches)[-1]
        tree = f[latest]
        list_branches(tree, f'{particle} / {latest}')


def estimate_total(particle):
    files = sorted((ROOT_DIR / particle).glob('*.root'))
    n_sample = min(5, len(files))
    total = 0
    for fp in files[:n_sample]:
        with uproot.open(fp) as f:
            matches = [k for k in f.keys() if k.startswith('TreeMc;')]
            if matches:
                total += f[sorted(matches)[-1]].num_entries
    avg = total / n_sample
    print(f'\n[{particle}] avg = {avg:,.0f} ev/file  ->  projected total {avg * len(files):,.0f}')


if __name__ == '__main__':
    for p in ['antiD', 'antiP']:
        inspect_first_file(p)
        estimate_total(p)
