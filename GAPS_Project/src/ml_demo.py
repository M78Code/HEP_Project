


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


if __name__ == '__main__':
    clustering_demo()
