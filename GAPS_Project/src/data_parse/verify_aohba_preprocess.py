"""
大場 2tof データの小サンプル(各粒子 1 file × 1000 events)で前処理パイプラインを end-to-end 検証。

確認項目:
  1. ROOT → dict (volume_id, energy, positions, times, label, beta) の変換が動く
  2. voxelizer.build_sili_voxel() の shape / nonzero / max
  3. voxelizer.build_tof_features() の dim / NaN・Inf / value range
  4. GraphBuilder.build_from_dict() の x (N,8) / graph_feat (45) / tof_feat (11)
  5. label 分布(antiD ファイル → 全 1, antiP ファイル → 全 0 を期待)
  6. β 分布(metadata 用)

NG なら preprocess 全体を直す。OK ならそのまま chunked preprocess に進む。
"""
import sys
from pathlib import Path

import numpy as np
import uproot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel, build_tof_features
from GAPS_Project.src.data_parse.graph_builder import GraphBuilder

ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')
N_EVENTS = 1000

PDG_ANTIPROTON   = -2212
PDG_ANTIDEUTERON = -1000010020


def load_events(particle, n=N_EVENTS):
    """1 file から先頭 n events を dict list に変換する。"""
    fp = sorted((ROOT_DIR / particle).glob('*.root'))[0]
    print(f'\n[{particle}] file: {fp.name}')

    with uproot.open(fp) as f:
        mc_key  = sorted(k for k in f.keys() if k.startswith('TreeMc;'))[-1]
        rec_key = sorted(k for k in f.keys() if k.startswith('TreeRec;'))[-1]
        mc, rec = f[mc_key], f[rec_key]

        pdg   = mc['Mc/primaryPdg_'].array(entry_stop=n, library='np')
        beta  = mc['Mc/CEventBase/primaryBetaGenerated_'].array(entry_stop=n, library='np')

        vol   = rec['Rec/hitseries_/hitseries_.volume_id_'].array(entry_stop=n, library='np')
        edep  = rec['Rec/hitseries_/hitseries_.energydep_'].array(entry_stop=n, library='np')
        hpos  = rec['Rec/hitseries_/hitseries_.hit_position_'].array(entry_stop=n, library='np')
        htime = rec['Rec/hitseries_/hitseries_.hit_time_'].array(entry_stop=n, library='np')

    events = []
    for i in range(len(pdg)):
        v = np.asarray(vol[i], dtype=np.int64)
        e = np.asarray(edep[i], dtype=np.float32)
        # TVector3 jagged → (N, 3) ndarray
        p_raw = hpos[i]
        if len(p_raw) > 0:
            pos = np.stack([p_raw['fX'], p_raw['fY'], p_raw['fZ']], axis=1).astype(np.float32)
        else:
            pos = np.zeros((0, 3), dtype=np.float32)
        t = np.asarray(htime[i], dtype=np.float32)

        events.append({
            'volume_id': v,
            'energy':    e,
            'positions': pos,
            'times':     t,
            'label':     int(pdg[i]),
            'beta':      float(beta[i]),
        })
    print(f'  loaded {len(events)} events')
    return events


def check_hits(events, particle):
    """生 hit の基本統計を確認。"""
    hit_counts = np.array([len(e['energy']) for e in events])
    print(f'  hits/event  min={hit_counts.min()}  max={hit_counts.max()}  '
          f'mean={hit_counts.mean():.2f}  median={np.median(hit_counts):.0f}')

    # zero-hit events (GraphBuilder の knn_graph で死ぬので preprocess で skip 対象)
    zero_hit = int((hit_counts == 0).sum())
    print(f'  zero-hit events: {zero_hit}/{len(events)}')

    pdgs = np.array([e['label'] for e in events])
    uniq, cnt = np.unique(pdgs, return_counts=True)
    print(f'  PDG distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}')

    betas = np.array([e['beta'] for e in events])
    print(f'  beta  min={betas.min():.3f}  max={betas.max():.3f}  '
          f'mean={betas.mean():.3f}')

    # 全 hit を平坦化して位置レンジを見る
    all_pos = np.concatenate([e['positions'] for e in events if len(e['positions']) > 0])
    if len(all_pos) > 0:
        print(f'  hit_position range:')
        for i, ax in enumerate(['x', 'y', 'z']):
            print(f'    {ax}: [{all_pos[:, i].min():.1f}, {all_pos[:, i].max():.1f}]')

        # Si(Li) hit の voxel 範囲 (-700~700) 外がどれだけ落ちるか
        all_vid = np.concatenate([e['volume_id'] for e in events if len(e['volume_id']) > 0])
        is_sili = (all_vid // 1_000_000) >= 200
        sili_pos = all_pos[is_sili]
        n_sili = len(sili_pos)
        if n_sili > 0:
            out_x = ((sili_pos[:, 0] < -700) | (sili_pos[:, 0] > 700)).sum()
            out_y = ((sili_pos[:, 1] < -700) | (sili_pos[:, 1] > 700)).sum()
            out_xy = (((sili_pos[:, 0] < -700) | (sili_pos[:, 0] > 700))
                      | ((sili_pos[:, 1] < -700) | (sili_pos[:, 1] > 700))).sum()
            print(f'  Si(Li) hits: {n_sili}  '
                  f'out_x={out_x} ({100 * out_x / n_sili:.2f}%)  '
                  f'out_y={out_y} ({100 * out_y / n_sili:.2f}%)  '
                  f'out_xy={out_xy} ({100 * out_xy / n_sili:.2f}%)')

    all_t = np.concatenate([e['times'] for e in events if len(e['times']) > 0])
    print(f'  hit_time range: [{all_t.min():.3f}, {all_t.max():.3f}]  '
          f'nan_frac={np.isnan(all_t).mean():.3f}')


def check_voxel(events):
    """build_sili_voxel の出力を 12×12 と 20×20 の両方で確認。
    20×20 は論文の主 CNN / FusedGravNet 用。
    """
    for grid in (12, 20):
        shapes_ok = 0
        nonzero_frac = []
        max_vals = []
        for e in events:
            v = build_sili_voxel(e, grid_x=grid, grid_y=grid)
            if v.shape == (10, grid, grid):
                shapes_ok += 1
            nonzero_frac.append((v > 0).mean())
            max_vals.append(v.max())
        print(f'  [10×{grid}×{grid}] shape OK: {shapes_ok}/{len(events)}  '
              f'nonzero  mean={np.mean(nonzero_frac):.4f}  max={np.max(nonzero_frac):.4f}  '
              f'voxel_max  mean={np.mean(max_vals):.3e}  max={np.max(max_vals):.3e}')


def check_tof(events):
    """build_tof_features の出力を確認。"""
    feats = np.stack([build_tof_features(e) for e in events])
    print(f'  tof_feat shape: {feats.shape}  (期待 (N,11))')
    print(f'  has NaN: {np.isnan(feats).any()}  has Inf: {np.isinf(feats).any()}')
    print(f'  per-dim stats:')
    names = ['outer_e', 'inner_e', 'outer_n', 'inner_n', 'tof',
             'oeX', 'oeY', 'oeZ', 'ieX', 'ieY', 'ieZ']
    for i, n in enumerate(names):
        col = feats[:, i]
        print(f'    [{i:2d}] {n:8s}  min={col.min():.3e}  max={col.max():.3e}  '
              f'mean={col.mean():.3e}')


def check_graph(events):
    """GraphBuilder の出力を確認。N<=1 の event は knn_graph が不安定なので skip。"""
    gb = GraphBuilder(k=8, normalize=True)
    x_shapes, y_vals, graph_dims = [], [], []
    tof_has_bad = 0
    skipped_small = 0

    for e in events:
        if len(e['energy']) <= 1:
            skipped_small += 1
            continue
        d = gb.build_from_dict(e)
        x_shapes.append(d.x.shape)
        y_vals.append(int(d.y.item()))
        gdim = (d.n_hits.numel() + d.total_energy.numel()
                + d.sili_profile.numel() + d.tof_profile.numel() + d.tof_feat.numel())
        graph_dims.append(gdim)
        if torch_any_bad(d.tof_feat):
            tof_has_bad += 1

    print(f'  skipped N<=1 events: {skipped_small}/{len(events)}')
    x1 = [s[1] for s in x_shapes]
    print(f'  x dim (期待 8):  unique={set(x1)}')
    print(f'  graph_feat dim (期待 45):  unique={set(graph_dims)}')
    yu, yc = np.unique(y_vals, return_counts=True)
    print(f'  y distribution:  {dict(zip(yu.tolist(), yc.tolist()))}')
    print(f'  tof_feat has NaN/Inf: {tof_has_bad}/{len(x_shapes)}')


def torch_any_bad(t):
    import torch
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


if __name__ == '__main__':
    for particle in ['antiD', 'antiP']:
        print(f'\n{"=" * 60}\n{particle}\n{"=" * 60}')
        events = load_events(particle)
        print('\n--- raw hits ---')
        check_hits(events, particle)
        print('\n--- voxelizer ---')
        check_voxel(events)
        print('\n--- tof features ---')
        check_tof(events)
        print('\n--- graph builder ---')
        check_graph(events)
