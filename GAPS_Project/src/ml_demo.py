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


def vec3_array(raw):





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



