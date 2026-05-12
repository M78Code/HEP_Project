import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import Scintillator_Project
PROJECT_ROOT = Path(Scintillator_Project.__file__).parent

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def check_single_waveform(split_dir):
    with open(split_dir / "test.json") as f:
        data = json.load(f)

    # 按位置分组
    from collections import defaultdict
    events_by_pos = defaultdict(list)
    for ev in data['events']:
        events_by_pos[ev['position_label']].append(ev)

    print("数据集中的位置标签：", sorted(events_by_pos.keys()))

    # ── 选择要查看的位置 ──────────────────────────
    TARGET_POS = 45.0  # ← 改这里可以换位置
    N_SAMPLES = 3  # ← 每个位置展示几个事件
    # ─────────────────────────────────────────────

    evs = events_by_pos[TARGET_POS][:N_SAMPLES]
    fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(14, N_SAMPLES * 4))
    fig.suptitle(f"位置 = {TARGET_POS} cm，前 {N_SAMPLES} 个事件", fontsize=14)
    for i, ev in enumerate(evs):
        ch0 = np.array(ev['CH0'])
        ch1 = np.array(ev['CH1'])
        x = np.arange(1024)

        for ch, ax, name in [(ch0, axes[i][0], "CH0（左端）"),
                             (ch1, axes[i][1], "CH1（右端）")]:
            baseline = ch[:100].mean()
            peak_idx = ch.argmin()
            peak_val = ch[peak_idx]

            # 10% 幅度阈值对应的脉冲区间
            thresh = baseline - (baseline - peak_val) * 0.1
            below = np.where(ch < thresh)[0]
            pulse_start = below[0] if len(below) > 0 else peak_idx
            pulse_end = below[-1] if len(below) > 0 else peak_idx

            ax.plot(x, ch, color='steelblue', linewidth=0.8, label='波形')
            ax.axhline(baseline, color='gray', linestyle='--', linewidth=0.8, label=f'基线 {baseline:.4f}')
            ax.axhline(thresh, color='orange', linestyle='--', linewidth=0.8, label=f'10%阈值 {thresh:.4f}')
            ax.axvline(peak_idx, color='red', linestyle='-', linewidth=1.0, label=f'峰值 idx={peak_idx}')
            ax.axvspan(pulse_start, pulse_end, alpha=0.15, color='green', label=f'脉冲区间[{pulse_start}~{pulse_end}]，共{pulse_end - pulse_start}点')

            # 候选固定窗口
            ax.axvspan(450, 800, alpha=0.07, color='purple', label='候选固定窗口 [450~800]')

            ax.set_title(f"事件 {i + 1} — {name}  (peak idx={peak_idx}, 脉冲宽={pulse_end - pulse_start})")
            ax.set_xlabel("样本点 index")
            ax.set_ylabel("幅度")
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlim(0, 1024)

    plt.tight_layout()
    # plt.savefig(f"waveform_pos{int(TARGET_POS)}cm.png", dpi=150)
    plt.show()
    # print(f"已保存：waveform_pos{int(TARGET_POS)}cm.png")


def draw_ppt_waveform():
    """
    PPT用波形图：15 / 80 / 140cm 的代表事例
    :return:
    """
    SPLIT_DIR = PROJECT_ROOT / 'dataset' / 'split'
    SAVE_PATH = PROJECT_ROOT / 'results' / 'plots' / 'ppt_waveform_comparsion.png'
    TARGET_POSITIONS = [15.0, 80.0, 140.0]
    ROI_START, ROI_END = 450, 800

    with open(SPLIT_DIR / "test.json") as f:
        data = json.load(f)

    events_by_pos = defaultdict(list)
    for ev in data['events']:
        events_by_pos[ev['position_label']].append(ev)

    fig, axes = plt.subplots(len(TARGET_POSITIONS), 1, figsize=(12, 9), sharex = True)
    fig.suptitle('Scintillator Waveforms by Position (CH0: Left PMT / CH1: Right PMT)', fontsize=13)

    x = np.arange(1024)

    for ax, pos in zip(axes, TARGET_POSITIONS):
        ev = events_by_pos[pos][0]
        ch0 = np.array(ev['CH0'])
        ch1 = np.array(ev['CH1'])

        ax.plot(x, ch0, color='#4C9BE8', linewidth=1.0, label='CH0 (Left PMT)')
        ax.plot(x, ch1, color='#E87B4C', linewidth=1.0, label='CH1 (Right PMT)')

        # ROI窗口
        ax.axvspan(ROI_START, ROI_END, alpha=0.08, color='green', label=f'ROI [{ROI_START}~{ROI_END}]')

        # 峰值标记
        peak0 = ch0.argmin()
        peak1 = ch1.argmin()
        ax.axvline(peak0, color='#4C9BE8', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(peak1, color='#E87B4C', linestyle='--', linewidth=0.8, alpha=0.7)

        delta_t = peak1 - peak0
        ax.set_title(f'Position = {pos:.0f} cm | Δt = {delta_t:+d} samples', fontsize=10)
        ax.set_ylabel('Amplitude')
        ax.set_xlim(400, 850)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Sample Index")
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=300)
    plt.show()
    print(f'保存: {SAVE_PATH}')


def check_all_waveforms(split_dir):
    """
    遍历所有事件，记录每个事件的峰值位置（index），按物理位置标签分组，画出分布。
    验证：
    1. 固定窗口 [450~800] 是否覆盖了所有位置的脉冲
    2. CH0和CH1的峰值index差（Δt）是否随位置变化——这直接验证时间差法的物理依据
    :return:
    """
    all_events = []
    for split in ['train', 'val', 'test']:
        with open(split_dir / f"{split}.json") as f:
            data = json.load(f)
        all_events.extend(data['events'])

    print(f"总事件数：{len(all_events)}")
    # 按位置分组，统计峰值index
    stats = defaultdict(lambda: {'peak_ch0': [], 'peak_ch1': [], "delta_t": []})

    for ev in all_events:
        ch0 = np.array(ev['CH0'])
        ch1 = np.array(ev['CH1'])
        pos = ev['position_label']

        peak_ch0 = ch0.argmin()
        peak_ch1 = ch1.argmin()
        delta_t = peak_ch1 - peak_ch0   # 正值 = CH1比CH0晚到

        stats[pos]['peak_ch0'].append(peak_ch0)
        stats[pos]['peak_ch1'].append(peak_ch1)
        stats[pos]['delta_t'].append(delta_t)

    positions = sorted(stats.keys())

    # ── 图1：各位置的峰值index分布（CH0 和 CH1）──────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, ch_key, ch_name, color in [
        (axes[0], "peak_ch0", "CH0（左端）", "steelblue"),
        (axes[1], "peak_ch1", "CH1（右端）", "darkorange"),
    ]:
        data_by_pos = [stats[p][ch_key] for p in positions]
        ax.boxplot(data_by_pos, positions=positions, widths=4,
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   medianprops=dict(color='red', linewidth=2))

        # 候选固定窗口上下界
        ax.axhline(450, color='purple', linestyle='--', linewidth=1, label='ウィンドウ下限 450') # 窗口下界 450
        ax.axhline(800, color='purple', linestyle='--', linewidth=1, label='ウィンドウ上限 800') # 窗口上界 800
        ax.fill_between([10, 145], 450, 800, alpha=0.07, color='purple', label='候補ウィンドウ [450~800]') # 候选窗口 [450, 800]

        ax.set_title(f"{ch_name} ピーク index 分布（位置ラベル別）") # 峰值 index 分布（按位置标签）
        ax.set_xlabel("位置ラベル (cm)") # 位置ラベル
        ax.set_ylabel("ピーク index") # 峰值
        ax.set_xticks(positions)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT / 'results' / 'plots'}/peak_index_by_position.png", dpi=150)
    plt.show()

    # ── 图2：Δt = peak_CH1 - peak_CH0 随位置变化 ──────────────
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    means = [np.mean(stats[p]["delta_t"]) for p in positions]
    stds = [np.std(stats[p]["delta_t"]) for p in positions]
    ax2.errorbar(positions, means, yerr=stds, fmt='o-', color='green', capsize=4, label='Δt 平均値 ± 標準偏差') # Δt 均值 ± 标准差
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title("Δt = peak_idx(CH1) - peak_idx(CH0) の位置依存性\n（正値：CH1 が CH0 より遅い、負値：CH1 が CH0 より早い）") # Δt = peak_idx(CH1) - peak_idx(CH0) 随位置的变化\n（正值=CH1比CH0晚，负值=CH1比CH0早）
    ax2.set_xlabel("位置ラベル (cm)") # 位置标签 (cm)
    ax2.set_ylabel("Δt（サンプル数）") # （样本数）
    ax2.set_xticks(positions)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT / 'results' / 'plots'}/delta_t_by_position.png", dpi=150)
    plt.show()

    # ── 终端输出汇总 ──────────────────────────────────────────
    print(f"\n{'位置':>6} | {'CH0峰值 min~max':>18} | {'CH1峰值 min~max':>18} | {'Δt 均值':>8} | {'事件数':>6}")
    print("-" * 72)
    for p in positions:
        p0 = stats[p]["peak_ch0"]
        p1 = stats[p]["peak_ch1"]
        dt = stats[p]["delta_t"]
        print(f"{p:>6.0f} | {min(p0):>6}~{max(p0):<10} | {min(p1):>6}~{max(p1):<10} | {np.mean(dt):>+8.1f} | {len(p0):>6}")

    # ── 窗口外事件统计 ──────────────────────────────────────
    WIN_START, WIN_END = 450, 800
    print(f"\n窗口 [{WIN_START}~{WIN_END}] 覆盖情况：")
    print(f"{'位置':>6} | {'CH0窗口外':>10} | {'CH1窗口外':>10} | {'总事件':>8} | {'窗口外占比':>10}")
    print("-" * 56)
    for p in positions:
        p0 = stats[p]["peak_ch0"]
        p1 = stats[p]["peak_ch1"]
        out_ch0 = sum(1 for x in p0 if x < WIN_START or x > WIN_END)
        out_ch1 = sum(1 for x in p1 if x < WIN_START or x > WIN_END)
        total = len(p0)
        ratio = (out_ch0 + out_ch1) / (total * 2) * 100
        print(f"{p:>6.0f} | {out_ch0:>10} | {out_ch1:>10} | {total:>8} | {ratio:>9.1f}%")


def parse_log(log_path: Path):
    """
    通过300epoch日志记录，绘制Loss曲线
    :param log_path: 
    :return: 
    """

    import re
    epochs, train_losses, val_losses = [], [], []
    pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+train_loss:\s+([\d.]+)\s+\|\s+val_loss:\s+([\d.]+)')
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
    return np.array(epochs), np.array(train_losses), np.array(val_losses)


def draw_loss_curve_png():
    """
    通过300epoch日志记录，绘制Loss曲线
    :return:
    """
    LOG_PATH = PROJECT_ROOT / 'logs' / '20260506-142427.log'
    SAVE_PATH = PROJECT_ROOT / 'results' / 'plots' / '20260506-142427_loss_curve.png'
    BEST_EPOCH = 276
    BEST_VAL_RMSE = 5.19

    epochs, train_losses, val_losses = parse_log(LOG_PATH)

    # loss -> RMSE
    train_rmse, val_rmse = np.sqrt(train_losses), np.sqrt(val_losses)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(epochs, train_rmse, color='#4C9BE8', linewidth=1.2, label='Train RMSE', alpha=0.9)
    ax.plot(epochs, val_rmse, color='#E87B4C', linewidth=1.2, label='Val RMSE', alpha=0.9)

    # 最优点标记
    ax.axvline(BEST_EPOCH, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(BEST_VAL_RMSE, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter([BEST_EPOCH], [BEST_VAL_RMSE], color='red', zorder=5, s=50, label=f'Best val RMSE = {BEST_VAL_RMSE:.2f} cm (Epoch {BEST_EPOCH})')

    # 学习率变化标记
    ax.axvline(75, color='green', linestyle=':', linewidth=0.8, alpha=0.5, label='LR decay (×0.7)')
    for step in [150, 225]:
        ax.axvline(step, color='green', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_xlim(5, 300)
    ax.set_ylabel('RMSE (cm)')
    ax.set_title('Training & Validation RMSE (300 Epochs)')
    ax.set_ylim(0, 20)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{SAVE_PATH}", dpi=300)
    plt.show()
    print(f'保存: {SAVE_PATH}')




if __name__ == "__main__":
    # 加载数据
    # split_dir = PROJECT_ROOT / 'dataset' / 'split'
    # check_single_waveform(split_dir)
    # check_all_waveforms(split_dir)

    # draw_loss_curve_png()
    draw_ppt_waveform()



"""
check_all_waveforms(split_dir)函数输出结果
总事件数：23602

    位置 |      CH0峰值 min~max |      CH1峰值 min~max |    Δt 均值 |    事件数
------------------------------------------------------------------------
    15 |     57~675        |    293~663        |    +27.7 |   6022
    25 |    568~607        |    588~636        |    +23.1 |    244
    35 |    567~611        |    585~636        |    +18.6 |    251
    45 |    543~697        |    512~708        |    +13.5 |   4509
    55 |    572~615        |    581~627        |     +9.3 |    301
    65 |    555~618        |    569~635        |     +5.3 |    222
    75 |    576~628        |    578~619        |     +0.0 |    318
    80 |      1~693        |    563~627        |     -2.0 |   3828
    85 |    579~633        |    577~617        |     -4.4 |    284
    95 |    567~650        |    569~631        |     -9.4 |    534
   105 |    580~647        |    570~646        |    -13.8 |    390
   115 |    579~643        |    540~646        |    -18.9 |    466
   125 |    397~802        |     56~647        |    -23.9 |   4805
   135 |    590~638        |    566~604        |    -27.8 |    239
   140 |    586~679        |    563~618        |    -30.4 |   1189

窗口 [450~800] 覆盖情况：
    位置 |     CH0窗口外 |     CH1窗口外 |      总事件 |      窗口外占比
--------------------------------------------------------
    15 |          1 |          1 |     6022 |       0.0%
    25 |          0 |          0 |      244 |       0.0%
    35 |          0 |          0 |      251 |       0.0%
    45 |          0 |          0 |     4509 |       0.0%
    55 |          0 |          0 |      301 |       0.0%
    65 |          0 |          0 |      222 |       0.0%
    75 |          0 |          0 |      318 |       0.0%
    80 |          1 |          0 |     3828 |       0.0%
    85 |          0 |          0 |      284 |       0.0%
    95 |          0 |          0 |      534 |       0.0%
   105 |          0 |          0 |      390 |       0.0%
   115 |          0 |          0 |      466 |       0.0%
   125 |          2 |          1 |     4805 |       0.0%
   135 |          0 |          0 |      239 |       0.0%
   140 |          0 |          0 |     1189 |       0.0%
"""