from pathlib import Path

import GAPS_Project
PROJECT_ROOT = Path(GAPS_Project.__file__).parent

def classification_demo():
    """
        鸢尾花分类任务
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Accuracy:", model.score(X_test, y_test))




def regression_demo():
    """
        鸢尾花回归任务
    :return:
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.predict(X_test[:5]))


def clustering_demo():
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris

    X, _ = load_iris(return_X_y=True)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print(kmeans.labels_)


def cuda_knn_ok():
    import torch, torch_cluster
    from torch_cluster import knn
    x = torch.randn(10, 3).cuda()
    b = torch.zeros(10, dtype=torch.long).cuda()
    idx = knn(x, x, k=3, batch_x=b, batch_y=b)
    print('CUDA knn OK:', idx.shape)

def check_beta():
    import pickle
    import numpy as np
    train_pkl_path = PROJECT_ROOT / 'dataset' / 'split' / 'train.pkl'
    with open(train_pkl_path, 'rb') as f:
        data = pickle.load(f)
    # print(f"类型: {type(data)}")
    # if isinstance(data, list):
    #     print(f"长度: {len(data)}")
    #     print(f"第一个元素类型: {type(data[0])}")
    #     print(f"第一个元素keys: {data[0].keys()}")
    # elif isinstance(data, dict):
    #     print(f"keys: {data.keys()}")
    #     first_key = next(iter(data))
    #     print(f"第一个key: {first_key}")
    #     print(f"第一个value类型: {type(data[first_key])}")

    events = data['events']
    # print(f"事件总数: {len(events)}")
    # print(f"第一个事件类型: {type(events[0])}")
    # print(f"第一个事件keys: {events[0].keys()}")
    # 查看label的实际值
    # labels = [e['label'] for e in events[:5]]
    # print(f"前5个label值: {labels}")
    # print(f"label类型: {type(labels[0])}")

    betas = np.array([e['beta'] for e in events[:10000]])
    print(f"beta 范围: {betas.min():.4f} ~ {betas.max():.4f}")
    print(f"beta 均值: {betas.mean():.4f}  std: {betas.std():.4f}")

    antiP_betas = np.array([e['beta'] for e in events[:10000] if e['label'] == -2212])
    antiD_betas = np.array([e['beta'] for e in events[:10000] if e['label'] == -1000010020])
    print(f"antiP beta: {antiP_betas.mean():.4f} ± {antiP_betas.std():.4f}  (n={len(antiP_betas)})")
    print(f"antiD beta: {antiD_betas.mean():.4f} ± {antiD_betas.std():.4f}  (n={len(antiD_betas)})")

def rec_primary_energy_depositions():
    import uproot
    import numpy as np

    with uproot.open(PROJECT_ROOT / 'dataset' / 'tar_root' / 'antiD' / 'antiD_2tof_FTFP_BERT_1778545887.root') as f:
        tree_rec = f['TreeRec']
        tree_mc = f['TreeMc']

        pri_edep = tree_rec['Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.first'].array()
        mc_edep = tree_mc['Mc/totalEnergyDeposition_'].array()
        mc_vid = tree_mc['Mc/volumeId_'].array()

        print('primaryEnergyDepositions_[0]:', pri_edep[0])
        print('Mc/totalEnergyDeposition_[0]:', mc_edep[0])
        print('Mc/volumeId_[0]:', mc_vid[0])
        print('长度对比:', len(mc_edep[0]), len(mc_vid[0]))


def check_range():
    import pickle
    import numpy as np

    train_pkl_path = PROJECT_ROOT / 'dataset' / 'split' / 'train.pkl'
    with open(train_pkl_path, 'rb') as f:
        payload = pickle.load(f)

    data = payload['events']
    print(f'総事例数: {len(data)}')
    print(f'フィールド一覧: {list(data[0].keys())}')

    x_all, y_all = [], []
    for ev in data[:2000]:
        vids = np.array(ev['volume_id'])
        pos  = np.array(ev['positions'])
        mask = (vids // 1000000) >= 200
        if mask.any():
            x_all.extend(pos[mask, 0].tolist())
            y_all.extend(pos[mask, 1].tolist())

    print(f'X: {min(x_all):.1f} ~ {max(x_all):.1f}')
    print(f'Y: {min(y_all):.1f} ~ {max(y_all):.1f}')


def train_log():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_file = PROJECT_ROOT / 'results' / '20260603-035100_DGCNN' / 'events.out.tfevents.1780458660.1e45c81b8399.363.0'

    ea = EventAccumulator(str(event_file))
    ea.Reload()
    print("Tags:")
    print(ea.Tags())

    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            print(f"{tag}: last step = {last.step}, value = {last.value}")


if __name__ == '__main__':
    # clustering_demo()
    # cuda_knn_ok()
    # check_range()
    # a = 1000 // 6
    # print(a)
    train_log()
    # rec_primary_energy_depositions()



"""
primaryEnergyDepositions_[0]: []
Mc/totalEnergyDeposition_[0]: [11.4, 12.8, 16, 3.09, 5.78, 2.73, ..., 3.31, 3.15, 2.32, 2.29, 0.0729, 0.0188]
Mc/volumeId_[0]: [100052000, 100003000, 110003000, ..., 105552000, 201150306, 201210005]
长度对比: 29 29
"""


cd ~/HEP_Project/GAPS_Project
conda activate naka

python - <<'PY'
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, roc_auc_score

# Japanese font
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
jp_font = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = jp_font.get_name()
mpl.rcParams["axes.unicode_minus"] = False

items = [
    (
        "CNN+DNN",
        Path("results/20260720-183528_CNNDNNFig72_nakagami_fig72_cnndnn_4M_rerun/evaluation_test"),
        "#1f77b4",
    ),
    (
        "GNN",
        Path("results/20260718-001435_SparseVoxelGNN_nakagami_atrest_sparse_voxel_gravnet_4M_std/evaluation_test"),
        "#ff7f0e",
    ),
]

out_dir = Path("results/ppt_figures_large_font")
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "fig_result1_cnndnn_vs_gnn_large_font.png"
out_pdf = out_dir / "fig_result1_cnndnn_vs_gnn_large_font.pdf"

plt.figure(figsize=(11, 7.2), dpi=180)

summary = []

for name, d, color in items:
    labels = np.load(d / "labels.npy")
    scores = np.load(d / "scores.npy")

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)

    # background rejection = 1 / false positive rate
    mask = fpr > 0
    rejection = np.empty_like(fpr)
    rejection[:] = np.nan
    rejection[mask] = 1.0 / fpr[mask]

    plt.plot(
        tpr[mask],
        rejection[mask],
        label=f"{name} AUC={auc:.4f}",
        color=color,
        linewidth=3.4,
    )

    summary.append({
        "model": name,
        "result_dir": str(d),
        "auc": float(auc),
    })

plt.yscale("log")
plt.xlim(0.5, 1.0)
plt.ylim(1, 1e6)

plt.xlabel("反重陽子の信号効率", fontsize=24)
plt.ylabel("反陽子背景の除去性能", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.grid(True, which="both", linestyle=":", linewidth=1.2, alpha=0.75)
plt.legend(fontsize=20, loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)

with open(out_dir / "fig_result1_cnndnn_vs_gnn_large_font_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("saved:", out_png)
print("saved:", out_pdf)
PY


