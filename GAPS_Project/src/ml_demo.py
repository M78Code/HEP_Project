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
    print(f"事件总数: {len(events)}")
    print(f"第一个事件类型: {type(events[0])}")
    print(f"第一个事件keys: {events[0].keys()}")


if __name__ == '__main__':
    # clustering_demo()
    # cuda_knn_ok()
    check_beta()
