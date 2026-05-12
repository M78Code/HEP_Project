import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime


import Scintillator_Project
from Scintillator_Project.src.data_parse.data_loader import make_dataloaders
from Scintillator_Project.src.models.cnn1d import CNN1D

PROJECT_ROOT = Path(Scintillator_Project.__file__).parent

RMSE_BASELINE = 4.7766
CFD_BASELINE = 4.93

# ===== 设备 =============
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def load_model(model_path: Path):
    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict(model, test_loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = model(x).squeeze(1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


def plot_scatter(y_true, y_pred, save_path):
    """真实 vs 预测 散点图"""
    errors = np.abs(y_pred - y_true)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(y_true, y_pred, c=errors, cmap="RdYlGn_r", alpha=0.4, s=8, vmin=0, vmax=20)
    plt.colorbar(sc, ax=ax, label="|Error| (cm)")

    lims = [14, 141]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Ideal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Position (cm)")
    ax.set_ylabel("Predicted Position (cm)")
    ax.set_title("True vs Predicted Position")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'保存: {save_path}')


def plot_error_vs_position(y_true, y_pred, save_path):
    """各位置的误差箱线图"""
    positions = sorted(set(y_true.astype(int)))
    data_by_pos = []
    labels = []
    for pos in positions:
        mask = y_true.astype(int) == pos
        errs = np.abs(y_pred[mask] - y_true[mask])
        data_by_pos.append(errs)
        labels.append(str(pos))

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data_by_pos, labels=labels, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor('#4C9BE8')
        patch.set_alpha(0.7)

    ax.axhline(RMSE_BASELINE, color='red', linestyle='--', linewidth=1.2, label=f'Overall RMSE = {RMSE_BASELINE:.2f} cm')
    ax.set_xlabel('True Position (cm)')
    ax.set_ylabel('|Error| (cm)')
    ax.set_title('Absolute Error by Position')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'保存: {save_path}')


def plot_error_histogram(y_true, y_pred, save_path):
    """误差直方图（含传统方法对比线）"""
    errors = y_pred - y_true # 含正负方向

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=60, color='#4C9BE8', alpha=0.8, edgecolor='white')

    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean = {errors.mean():.2f} cm')

    # 传统方法对比（大場卒論 表1，CFD融合法）
    ax.axvline(CFD_BASELINE, color='orange', linestyle=':', linewidth=1.2, label=f'Traditional best RMSE = {CFD_BASELINE:.2f} cm (CFD)')
    ax.axvline(-CFD_BASELINE, color='orange', linestyle=':', linewidth=1.2)

    ax.set_xlabel('Prediction Error (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution (CNN)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f'保存: {save_path}')


def main():
    model_path = PROJECT_ROOT / 'results' / '20260506-142427_best_model.pth'
    print(f'使用模型: {model_path.name}')

    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    _, _, test_loader = make_dataloaders(split_dir, batch_size=64)

    model = load_model(model_path)
    y_true, y_pred = predict(model, test_loader)

    # 输出目录
    plots_dir = PROJECT_ROOT / 'results' /'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')

    plot_scatter(y_true, y_pred, plots_dir / f'{ts}_scatter.png')
    plot_error_vs_position(y_true, y_pred, plots_dir / f'{ts}_error_by_position.png')
    plot_error_histogram(y_true, y_pred, plots_dir / f'{ts}_error_histogram.png')

    print(f'\n完成。 图片保存于: {plots_dir}')


if __name__ == '__main__':
    main()

