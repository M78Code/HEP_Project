# 波形预处理
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#
# fig, axes = plt.subplots(3, 1, figsize=(12, 7))
# fig.patch.set_facecolor('#FAFAFA')
#
# input_vals = [1, 3, 5, 4, 2, 1, 3]
# kernel = [1, 0, -1]
# n = len(input_vals)
# k = len(kernel)
# output = [sum(input_vals[i+j]*kernel[j] for j in range(k)) for i in range(n-k+1)]
#
# colors_input = ['#AED6F1'] * n
# colors_kernel = ['#A9DFBF'] * k
# colors_output = ['#F9E79F'] * len(output)
#
# def draw_boxes(ax, values, colors, y=0, label='', highlight=None):
#    for i, (v, c) in enumerate(zip(values, colors)):
#        col = '#F1948A' if highlight is not None and i in highlight else c
#        rect = mpatches.FancyBboxPatch((i*1.1, y), 0.9, 0.7,
#            boxstyle="round,pad=0.05", linewidth=1.5,
#            edgecolor='#555', facecolor=col)
#        ax.add_patch(rect)
#        ax.text(i*1.1+0.45, y+0.35, str(v), ha='center', va='center',
#                fontsize=13, fontweight='bold')
#    if label:
#        ax.text(-0.6, y+0.35, label, ha='right', va='center', fontsize=11)
#
# # Step 0
# ax = axes[0]
# ax.set_xlim(-1, 9); ax.set_ylim(-0.3, 2.2); ax.axis('off')
# ax.set_title('Step 1：位置 0　　1×1 + 3×0 + 5×(−1) = −4', fontsize=12, pad=6)
# draw_boxes(ax, input_vals, colors_input, y=1.3, label='入力', highlight=[0,1,2])
# draw_boxes(ax, kernel,     colors_kernel, y=0.4, label='フィルタ')
# ax.annotate('', xy=(0*1.1+0.45, 1.3), xytext=(0*1.1+0.45, 1.1),
#            arrowprops=dict(arrowstyle='->', color='#888'))
# for i in range(3):
#    ax.annotate('', xy=(i*1.1+0.45, 0.4+0.7), xytext=(i*1.1+0.45, 1.3),
#                arrowprops=dict(arrowstyle='-', color='#E74C3C', lw=1.2, linestyle='dashed'))
# ax.text(0.45, 0.05, '−4', ha='center', va='center', fontsize=13,
#        fontweight='bold', color='#C0392B',
#        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#E67E22', lw=1.5))
# ax.text(0.45, -0.2, '出力[0]', ha='center', fontsize=9, color='#888')
#
# # Step 1
# ax = axes[1]
# ax.set_xlim(-1, 9); ax.set_ylim(-0.3, 2.2); ax.axis('off')
# ax.set_title('Step 2：位置 1　　3×1 + 5×0 + 4×(−1) = −1', fontsize=12, pad=6)
# draw_boxes(ax, input_vals, colors_input, y=1.3, label='入力', highlight=[1,2,3])
# draw_boxes(ax, kernel,     colors_kernel, y=0.4, label='フィルタ')
# for i in range(3):
#    ax.annotate('', xy=((i+1)*1.1+0.45, 0.4+0.7), xytext=((i+1)*1.1+0.45, 1.3),
#                arrowprops=dict(arrowstyle='-', color='#E74C3C', lw=1.2, linestyle='dashed'))
# ax.text(1.1+0.45, 0.05, '−1', ha='center', va='center', fontsize=13,
#        fontweight='bold', color='#C0392B',
#        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#E67E22', lw=1.5))
# ax.text(1.1+0.45, -0.2, '出力[1]', ha='center', fontsize=9, color='#888')
#
# # Step 2
# ax = axes[2]
# ax.set_xlim(-1, 9); ax.set_ylim(-0.3, 2.2); ax.axis('off')
# ax.set_title('Step 3：位置 2　　5×1 + 4×0 + 2×(−1) = ＋3', fontsize=12, pad=6)
# draw_boxes(ax, input_vals, colors_input, y=1.3, label='入力', highlight=[2,3,4])
# draw_boxes(ax, kernel,     colors_kernel, y=0.4, label='フィルタ')
# for i in range(3):
#    ax.annotate('', xy=((i+2)*1.1+0.45, 0.4+0.7), xytext=((i+2)*1.1+0.45, 1.3),
#                arrowprops=dict(arrowstyle='-', color='#E74C3C', lw=1.2, linestyle='dashed'))
# ax.text(2*1.1+0.45, 0.05, '+3', ha='center', va='center', fontsize=13,
#        fontweight='bold', color='#C0392B',
#        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9E79F', edgecolor='#E67E22', lw=1.5))
# ax.text(2*1.1+0.45, -0.2, '出力[2]', ha='center', fontsize=9, color='#888')
#
# plt.suptitle('1D 畳み込み（Convolution）— フィルタのスライド', fontsize=14, fontweight='bold', y=1.01)
# plt.tight_layout()
# plt.savefig('conv1d_slide_demo.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
# print("saved")




# plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS',  'AppleGothic']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

input_vals = [1, 3, 5, 4, 2, 1, 3]
kernel = [1, 0, -1]
n = len(input_vals)
k = len(kernel)
output = [sum(input_vals[i+j]*kernel[j] for j in range(k)) for i in range(n-k+1)]
# output = [-4, -1, 3, 3, -1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#FAFAFA')

# ── 左：入力・フィルタ・出力をボックスで表示 ──────────────────
ax = axes[0]
ax.set_xlim(-1.2, 8.5)
ax.set_ylim(-0.5, 3.5)
ax.axis('off')
ax.set_title('入力・フィルタ・出力（全ステップ）', fontsize=13, fontweight='bold', pad=10)

def draw_row(ax, values, y, color, label):
   for i, v in enumerate(values):
       rect = mpatches.FancyBboxPatch(
           (i * 1.05, y), 0.88, 0.65,
           boxstyle="round,pad=0.05", lw=1.5,
           edgecolor='#555', facecolor=color)
       ax.add_patch(rect)
       ax.text(i * 1.05 + 0.44, y + 0.32, str(v),
               ha='center', va='center', fontsize=13, fontweight='bold')
   ax.text(-0.9, y + 0.32, label, ha='right', va='center', fontsize=11)

draw_row(ax, input_vals, 2.5, '#AED6F1', '入力')
draw_row(ax, kernel,     1.4, '#A9DFBF', 'フィルタ')
draw_row(ax, output,     0.2, '#F9E79F', '出力')

# フィルタとラベルの説明
ax.text(3.5 * 1.05 + 0.44, 1.1,
       'フィルタ [1, 0, −1]：差分検出（エッジ検出）',
       ha='center', fontsize=9, color='#555', style='italic')
ax.text(2.5 * 1.05 + 0.44, -0.1,
       '出力長 = 入力長 − フィルタ長 + 1 = 7 − 3 + 1 = 5',
       ha='center', fontsize=9, color='#555')

# 出力の正負に色付け注釈
for i, v in enumerate(output):
   color = '#C0392B' if v < 0 else '#1A5276'
   ax.text(i * 1.05 + 0.44, 0.0, '↑' if v > 0 else '↓',
           ha='center', va='top', fontsize=11, color=color, fontweight='bold')

# ── 右：折れ線グラフで入力・出力の波形を比較 ──────────────────
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')
ax2.set_title('波形で見る：入力 vs 出力', fontsize=13, fontweight='bold', pad=10)

x_in  = np.arange(len(input_vals))
x_out = np.arange(len(output)) + 1  # フィルタ中心に合わせてオフセット

ax2.plot(x_in, input_vals, 'o-', color='#2E86C1', lw=2, ms=8, label='入力信号')
ax2.plot(x_out, output,    's--', color='#E67E22', lw=2, ms=8, label='出力（畳み込み後）')

# 出力の値をラベル表示
for x, v in zip(x_out, output):
   ax2.annotate(str(v), (x, v),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=10, color='#E67E22', fontweight='bold')

ax2.axhline(0, color='#AAA', lw=0.8, linestyle='--')
ax2.fill_between(x_out, output, 0,
                where=[v > 0 for v in output], alpha=0.15, color='#1A5276', label='正（下降検出）')
ax2.fill_between(x_out, output, 0,
                where=[v < 0 for v in output], alpha=0.15, color='#C0392B', label='負（上昇検出）')

ax2.set_xlabel('位置インデックス', fontsize=11)
ax2.set_ylabel('値', fontsize=11)
ax2.set_xticks(range(len(input_vals)))
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('1D 畳み込みの結果まとめ　—　フィルタ [1, 0, −1] による差分検出',
            fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('conv1d_summary.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("saved: conv1d_summary.png")