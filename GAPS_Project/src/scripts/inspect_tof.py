"""分析TOF layer_idx编码规则，判断inner/outer TOF"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import uproot
import matplotlib.pyplot as plt

import GAPS_Project


# 项目根目录（GAPS_Project/），所有路径基于此
PROJECT_ROOT = Path(GAPS_Project.__file__).parent
MAX_EVENTS = 5000
ROOT_PATH = PROJECT_ROOT / 'dataset' / 'test_sample' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.root'


def inspect_tof():

    with uproot.open(ROOT_PATH) as f:
        tree = f["TreeRec;1"]

        volume_all = tree["Rec/hitseries_/hitseries_.volume_id_"].array(
            entry_stop=MAX_EVENTS, library="np"
        )
        energy_all = tree["Rec/hitseries_/hitseries_.energydep_"].array(
            entry_stop=MAX_EVENTS, library="np"
        )
        time_all = tree["Rec/hitseries_/hitseries_.hit_time_"].array(
            entry_stop=MAX_EVENTS, library="np"
        )
        pos_all = tree["Rec/hitseries_/hitseries_.hit_position_"].array(
            entry_stop=MAX_EVENTS, library="ak"
        )

    stats = defaultdict(lambda: {
        "count": 0,
        "energy_sum": 0.0,
        "times": [],
        "xs": [], "ys": [], "zs": [], "rxys": [], "abs_zs": [],
        "volume_ids": set(),
    })

    # 同时收集画图数据
    plot_xs, plot_ys, plot_zs, plot_layers = [], [], [], []

    for i in range(len(volume_all)):
        vids = np.asarray(volume_all[i], dtype=np.int64)
        ens = np.asarray(energy_all[i], dtype=np.float32)
        ts = np.asarray(time_all[i], dtype=np.float32)

        pos_raw = pos_all[i]
        xs = np.asarray(pos_raw["fX"], dtype=np.float32)
        ys = np.asarray(pos_raw["fY"], dtype=np.float32)
        zs = np.asarray(pos_raw["fZ"], dtype=np.float32)
        rxys = np.sqrt(xs**2 + ys**2)
        abs_zs = np.abs(zs)

        layer_idx = vids // 1000000
        is_tof = layer_idx < 200

        # 画图数据
        plot_xs.append(xs[is_tof])
        plot_ys.append(ys[is_tof])
        plot_zs.append(zs[is_tof])
        plot_layers.append(layer_idx[is_tof])

        for vid, li, e, t, x, y, z, rxy, az in zip(
            vids[is_tof], layer_idx[is_tof], ens[is_tof], ts[is_tof],
            xs[is_tof], ys[is_tof], zs[is_tof], rxys[is_tof], abs_zs[is_tof],
        ):
            d = stats[int(li)]
            d["count"] += 1
            d["energy_sum"] += float(e)
            d["volume_ids"].add(int(vid))
            if not np.isnan(t):
                d["times"].append(float(t))
            d["xs"].append(float(x))
            d["ys"].append(float(y))
            d["zs"].append(float(z))
            d["rxys"].append(float(rxy))
            d["abs_zs"].append(float(az))

    # ── 统计表 ──────────────────────────────
    print(f"\n分析了 {len(volume_all)} 个event")
    print(f"\n{'layer':>6} {'count':>8} {'n_vid':>6} "
          f"{'rxy_mean':>9} {'rxy_min':>9} {'rxy_max':>9} "
          f"{'abs_z_m':>9} {'z_mean':>9} "
          f"{'t_mean':>9} {'t_min':>9} {'t_max':>9} "
          f"{'E_sum':>12}")
    print("-" * 120)

    for li in sorted(stats.keys()):
        d = stats[li]
        times = np.array(d["times"], dtype=np.float32)
        rxy_arr = np.array(d["rxys"], dtype=np.float32)
        zs_arr = np.array(d["zs"], dtype=np.float32)
        az_arr = np.array(d["abs_zs"], dtype=np.float32)

        if len(times) > 0:
            t_min, t_mean, t_max = np.nanmin(times), np.nanmean(times), np.nanmax(times)
        else:
            t_min = t_mean = t_max = np.nan

        print(f"{li:>6d} {d['count']:>8d} {len(d['volume_ids']):>6d} "
              f"{np.mean(rxy_arr):>9.1f} {np.min(rxy_arr):>9.1f} {np.max(rxy_arr):>9.1f} "
              f"{np.mean(az_arr):>9.1f} {np.mean(zs_arr):>9.1f} "
              f"{t_mean:>9.2f} {t_min:>9.2f} {t_max:>9.2f} "
              f"{d['energy_sum']:>12.2f}")

    print(f"\n{'='*60}")
    print("Example volume_ids per TOF layer:")
    for li in sorted(stats.keys()):
        examples = sorted(list(stats[li]["volume_ids"]))[:15]
        print(f"  layer_idx={li}: {examples}")

    # ── 也统计Si(Li)的layer_idx做对照 ──────────
    print(f"\n{'='*60}")
    print("Si(Li) layer_idx 分布（对照）:")
    sili_stats = defaultdict(int)
    for i in range(len(volume_all)):
        vids = np.asarray(volume_all[i], dtype=np.int64)
        layer_idx = vids // 1000000
        for li in layer_idx[layer_idx >= 200]:
            sili_stats[int(li)] += 1
    for li in sorted(sili_stats.keys()):
        print(f"  layer_idx={li}: count={sili_stats[li]}")

    # ── 画图 ──────────────────────────────────
    plot_xs = np.concatenate(plot_xs)
    plot_ys = np.concatenate(plot_ys)
    plot_zs = np.concatenate(plot_zs)
    plot_layers = np.concatenate(plot_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # X-Y 俯视图
    sc1 = axes[0].scatter(plot_xs, plot_ys, c=plot_layers, s=2, alpha=0.4, cmap='tab10')
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("TOF hits: X-Y view (colored by layer_idx)")
    axes[0].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[0], label="layer_idx")

    # Z-R 侧视图
    r_all = np.sqrt(plot_xs**2 + plot_ys**2)
    sc2 = axes[1].scatter(plot_zs, r_all, c=plot_layers, s=2, alpha=0.4, cmap='tab10')
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("r = sqrt(x²+y²) [mm]")
    axes[1].set_title("TOF hits: Z-R view (colored by layer_idx)")
    plt.colorbar(sc2, ax=axes[1], label="layer_idx")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "results" / "tof_layer_analysis.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    inspect_tof()


"""
分析了 57 个event

 layer    count  n_vid  rxy_mean   rxy_min   rxy_max   abs_z_m    z_mean    t_mean     t_min     t_max        E_sum
------------------------------------------------------------------------------------------------------------------------
   100      255     45     616.9      74.5    2255.3    1055.1    1055.1     62.96      5.30   2547.14      1915.90
   102       45     15    1182.2    1056.8    1482.3     442.9    -420.9    274.17     13.72   5288.08       155.98
   103       49     15    1159.4    1052.0    1462.2     450.4    -405.8     97.85     14.51   3028.33        89.51
   104       54     16    1203.4    1052.8    1476.1     415.1    -389.4    125.39     18.35   4680.27       126.38
   105       57     16    1171.1    1052.1    1406.2     456.1    -442.1    140.39     14.35   2487.64       130.27
   110      203     12     370.1      74.5    1055.6     134.8     134.8    147.92     10.49   4780.70      1814.05
   111      110     12     556.4      75.8    1051.6    1110.2   -1110.2    179.24     16.89   4224.11       263.84
   112       54      8     901.4     824.4    1136.2     433.7    -433.7    601.59     12.63   5286.89       240.74
   113       46      8     917.6     830.9    1139.4     485.2    -485.2    163.92     15.15   3028.21        88.97
   114       51      8     934.1     824.1    1135.2     470.4    -470.4     81.33     16.42   1827.28       186.19
   115       76      8     928.1     824.3    1100.5     471.6    -471.6     83.02     12.13   2487.06       220.23
   116        6      4    1101.4    1101.4    1101.4     577.6    -577.6     32.16     16.31     58.81        11.27

============================================================
Example volume_ids per TOF layer:
  layer_idx=100: [100000000, 100001000, 100002000, 100003000, 100004000, 100005000, 100050000, 100051000, 100052000, 100053000, 100054000, 100055000, 100200000, 100201000, 100202000]
  layer_idx=102: [102000000, 102001000, 102002000, 102003000, 102050000, 102051000, 102052000, 102053000, 102500000, 102501000, 102503000, 102550000, 102551000, 102552000, 102553000]
  layer_idx=103: [103000000, 103001000, 103002000, 103003000, 103050000, 103051000, 103052000, 103053000, 103500000, 103501000, 103503000, 103550000, 103551000, 103552000, 103553000]
  layer_idx=104: [104000000, 104001000, 104002000, 104003000, 104050000, 104051000, 104052000, 104053000, 104500000, 104501000, 104502000, 104503000, 104550000, 104551000, 104552000]
  layer_idx=105: [105000000, 105001000, 105002000, 105003000, 105050000, 105051000, 105052000, 105053000, 105500000, 105501000, 105502000, 105503000, 105550000, 105551000, 105552000]
  layer_idx=110: [110000000, 110001000, 110002000, 110003000, 110004000, 110005000, 110050000, 110051000, 110052000, 110053000, 110054000, 110055000]
  layer_idx=111: [111000000, 111001000, 111002000, 111003000, 111004000, 111005000, 111050000, 111051000, 111052000, 111053000, 111054000, 111055000]
  layer_idx=112: [112000000, 112001000, 112002000, 112003000, 112050000, 112051000, 112052000, 112053000]
  layer_idx=113: [113000000, 113001000, 113002000, 113003000, 113050000, 113051000, 113052000, 113053000]
  layer_idx=114: [114000000, 114001000, 114002000, 114003000, 114050000, 114051000, 114052000, 114053000]
  layer_idx=115: [115000000, 115001000, 115002000, 115003000, 115050000, 115051000, 115052000, 115053000]
  layer_idx=116: [116000000, 116100000, 116200000, 116300000]

============================================================
Si(Li) layer_idx 分布（对照）:
  layer_idx=200: count=87
  layer_idx=201: count=88
  layer_idx=202: count=84
  layer_idx=203: count=87
  layer_idx=204: count=83
  layer_idx=205: count=61
  layer_idx=206: count=80
  layer_idx=207: count=75
  layer_idx=208: count=69
  layer_idx=209: count=50

图已保存: /Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/results/tof_layer_analysis.png
"""
