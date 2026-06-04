"""
FusedGravNet (GravNet_6b_h128 + CNN encoder) 评估脚本
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import GAPS_Project
from GAPS_Project.src.data_parse.data_loader import make_data_loaders_from_split
from GAPS_Project.src.models.fused_model import FusedGravNet

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── 配置 ──────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 64    # FusedGravNet有CNN，显存需求较大
LAZY_LOAD = True
NUM_WORKERS = 0

MODEL_PATH = PROJECT_ROOT / 'results' / '20260529-171008_FusedGravNet' / '20260529-171008_FusedGravNet_best.pth'
OUT_DIR = PROJECT_ROOT / 'results' / 'evaluation'
MODEL_TAG = 'FusedGravNet'


@torch.no_grad()
def run_inference(model, loader, device):
    """返回 (labels, probs, betas) numpy arrays"""
    model.eval()
    all_labels, all_probs, all_betas = [], [], []

    for batch in tqdm(loader, desc=f'{MODEL_TAG} eval'):
        batch = batch.to(device)

        graph_feat = torch.cat([
            batch.n_hits.view(-1, 1),
            batch.total_energy.view(-1, 1),
            batch.sili_profile.view(-1, 16),
            batch.tof_profile.view(-1, 16),
            batch.tof_feat.view(-1, 11),
        ], dim=1)

        voxel = torch.log1p(batch.voxel.view(-1, 1, 10, 20, 20))  # log1p 标准化，与训练一致
        logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat, voxel=voxel)
        probs = torch.softmax(logits, dim=1)[:, 1]  # antiD概率

        all_labels.append(batch.y.squeeze().cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        if hasattr(batch, 'mc_beta'):
            all_betas.append(batch.mc_beta.squeeze().cpu().numpy())

    labels = np.concatenate(all_labels)
    probs_arr = np.concatenate(all_probs)
    betas = np.concatenate(all_betas) if all_betas else np.zeros(len(labels))
    return labels, probs_arr, betas

def print_metrics(name, labels, probs):
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'=' * 55}")
    print(f"模型: {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (antiD识别精确率)")
    print(f"  Recall    : {rec:.4f}  (antiD信号效率)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  混淆矩阵  : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  背景抑制率: {tn / (tn + fp):.4f}  ({tn + fp}个antiP中正确拒绝{tn}个)")
    return auc


def print_rejection_at_efficiency(results, signal_efficiencies=(0.50, 0.80, 0.90, 0.95, 0.98, 0.99)):
    print(f"\n{'=' * 70}")
    print('各信号效率下的背景抑制率 (Background Rejection = 1/FPR)')
    col_w = 22
    print(f"{'Signal Eff':>12}", end="")
    for name, _, _ in results:
        print(f" {name:>{col_w}}", end="")
    print()
    print('-' * (12 + (col_w + 1) * len(results)))

    for target_eff in signal_efficiencies:
        print(f" {target_eff:>12.2f}", end="")
        for name, labels, probs in results:
            fpr, tpr, _ = roc_curve(labels, probs)
            idx = np.argmin(np.abs(tpr - target_eff))
            f = fpr[idx]
            rej_str = f'{1.0 / f:.2e}' if f > 0 else '>1e10'
            print(f"  {rej_str:>{col_w}}", end="")
        print()

def plot_rejection_curve(results, save_path):
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        fpr, tpr, _ = roc_curve(labels, probs)
        fpr_safe = np.where(fpr == 0, 1e-10, fpr)
        plt.semilogy(tpr, 1.0 / fpr_safe, label=name)
    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.legend()
    plt.xlim(0.5, 1.0)
    plt.ylim(1, 2e4)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)

    def _log_fmt(x, _):
        if x == 1:   return '1'
        if x == 10:  return '10'
        exp = int(round(np.log10(x)))
        return f'$10^{{{exp}}}$'

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(_log_fmt))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Rejection曲线已保存: {save_path}")


def evaluate():
    print(f'使用设备: {DEVICE}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ── 加载测试集 ──
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载test数据集...')
    _, _, test_loader = make_data_loaders_from_split(
    split_dir=split_dir, batch_size=BATCH_SIZE,
    lazy=LAZY_LOAD, num_workers=NUM_WORKERS)
    print(f'test batches: {len(test_loader)}')
    # ── 加载模型 ──
    model = FusedGravNet(
        in_channels=8, hidden_dim=128, graph_feat_dim=45,
        num_blocks=6, cnn_feat_dim=128, dropout=0.3,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f'已加载模型: {MODEL_PATH}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

    # ── 推理 ──
    labels, probs, betas = run_inference(model, test_loader, DEVICE)

    # ── 评价指标 ──
    auc = print_metrics(MODEL_TAG, labels, probs)

    # ── 保存 ──
    np.save(OUT_DIR / f'{MODEL_TAG}_labels.npy', labels)
    np.save(OUT_DIR / f'{MODEL_TAG}_probs.npy',  probs)
    np.save(OUT_DIR / f'{MODEL_TAG}_betas.npy',  betas)
    print(f'\n推理结果已保存至: {OUT_DIR}')

    # ── 与其他模型对比 ──
    results = [(MODEL_TAG, labels, probs)]
    for name in ['GravNet_6b_h128_rec', 'DGCNN_rec']:
        lbl_path = OUT_DIR / f'{name}_labels.npy'
        prb_path = OUT_DIR / f'{name}_probs.npy'
        if lbl_path.exists() and prb_path.exists():
            lbl = np.load(lbl_path)
            prb = np.load(prb_path)
            results.append((name, lbl, prb))
            print(f'已加载对比模型: {name}')

    print_rejection_at_efficiency(results)
    plot_rejection_curve(results, OUT_DIR / f'rejection_{MODEL_TAG}_vs_all.png')
    print(f'\n最终AUC: {auc:.4f}')


if __name__ == '__main__':
    evaluate()

