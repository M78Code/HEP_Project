# PyG（PyTorch Geometric）とは

## 1. PyG とは

PyG は **PyTorch Geometric** の略です。

これは、PyTorch をベースにした

**Graph Neural Network（GNN, グラフニューラルネットワーク）**

のためのライブラリです。

通常の CNN が画像（Image）を扱うのに対して、
PyG は

* Graph（グラフ構造データ）

を扱うために使います。

---

## 2. なぜ GAPS で PyG を使うのか

GAPS のデータは、単なる

* CH0 + CH1 の波形

だけではなく、将来的には

* Si detector
* TOF detector
* Tracker
* 複数の hit points

など、多数の detector hit を扱います。

これらのデータは

「点（hit）」と
「点同士の関係」

を持っているため、
自然に **Graph（グラフ）** として表現できます。

そのため、

**Graph Neural Network（GNN）**

が非常に適しています。

これは現在の HEP（高エネルギー物理）でも
非常に注目されている手法です。

---

## 3. Data オブジェクトとは

PyG の最も重要なデータ構造は

```python
from torch_geometric.data import Data
```

で作る

**Data オブジェクト**

です。

これは

**1つの event = 1つの graph**

を表します。

---

## 4. event が graph になる理由

例えば、1つの event に

* 10個の detector hits

があるとします。

各 hit には

* 位置（x, y, z）
* energy deposition
* time
* PDG
* detector ID

などの情報があります。

これらが

## ノード（nodes）

になります。

さらに、

* 空間的な距離
* 時間的な関係
* 物理的な関連性

が

## エッジ（edges）

になります。

つまり

```text
1 event = 1 graph
```

となります。

---

## 5. Data の基本構造

```python
data = Data(
    x=node_features,
    edge_index=edge_connections,
    y=label
)
```

---

## 6. 各パラメータの意味

### x

ノード特徴量（node features）

```python
x.shape = [num_nodes, num_features]
```

例：

* 10 hits
* 各 hit に 5個の特徴

なら

```text
[10, 5]
```

になります。

---

### edge_index

ノード同士の接続関係

```python
edge_index.shape = [2, num_edges]
```

例：

* 0 → 1
* 1 → 2
* 2 → 3

など。

---

### y

教師ラベル（label）

例えば

* antideuteron かどうか（分類）
* 粒子通過位置（回帰）

などです。

---

## 7. この文の意味

> 将单个 event 的 hit 信息转换成 PyG 的 Data 对象

これは

**ROOT ファイルの event データを
GNN が学習できる形式に変換する**

という意味です。

つまり

```python
torch_geometric.data.Data
```

に変換することです。

この前処理は
非常に重要です。

---

## 8. 最も簡単な例

元の event：

```python
event = {
    "hit1": [x, y, z, E],
    "hit2": [x, y, z, E],
    ...
}
```

これを

```python
Data(
    x=tensor(...),
    edge_index=tensor(...),
    y=label
)
```

に変換します。

---

## 9. まとめ

```text
PyG = PyTorch Geometric
```

これは

**Graph Neural Network（GNN）**

を扱うためのライブラリです。

そして

```text
event → Data object
```

とは

**物理イベントをグラフ構造データに変換すること**

を意味します。

これは

**GAPS + Machine Learning + GNN**

において最も重要なステップの一つです。
