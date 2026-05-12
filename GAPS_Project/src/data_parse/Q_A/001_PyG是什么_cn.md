# PyG（PyTorch Geometric）是什么

## 1. PyG 是什么

PyG 是 **PyTorch Geometric** 的简称。

它是基于 PyTorch 的

**图神经网络（Graph Neural Network, GNN）**

专用库。

普通的 CNN 主要处理：

* 图像（Image）
* 序列（Sequence）
* 表格（Table）

而 PyG 主要处理：

* 图结构数据（Graph）

因此，它非常适合高能物理（HEP）中的复杂探测器数据分析。

---

## 2. 为什么 GAPS 会用到 PyG

GAPS 的数据不仅仅是：

* CH0 + CH1 的波形数据

未来更重要的是：

* Si detector
* TOF detector
* Tracker
* 多个 detector hit points

这些数据天然具有：

“点（hit）”
和
“点与点之间的关系”

因此非常适合表示成：

## Graph（图）

而不是普通的一维波形。

所以：

## Graph Neural Network（GNN）

就非常适合这种任务。

这也是当前高能物理（HEP）领域非常热门的研究方向。

---

## 3. Data 对象是什么

PyG 最核心的数据结构就是：

```python
from torch_geometric.data import Data
```

这个：

## Data 对象

它表示：

## 一张图（one graph）

也就是：

```text
一个 event = 一张 graph
```

---

## 4. 为什么一个 event 可以表示成 graph

例如：

一个 event 中有：

* 10 个 detector hits

每个 hit 包含：

* 位置（x, y, z）
* 能量沉积（energy deposition）
* 时间（time）
* PDG
* detector ID

这些就是：

## 节点（nodes）

而节点之间：

* 空间距离
* 时间关系
* 物理关联

这些就是：

## 边（edges）

因此：

```text
一个 event = 一张图（graph）
```

这是非常自然的表示方式。

---

## 5. Data 的基本结构

```python
data = Data(
    x=node_features,
    edge_index=edge_connections,
    y=label
)
```

---

## 6. 各个参数的意义

## x

节点特征（node features）

```python
x.shape = [num_nodes, num_features]
```

例如：

* 10 个 hits
* 每个 hit 有 5 个特征

则：

```text
[10, 5]
```

表示：

10 个节点，每个节点 5 个特征。

---

## edge_index

边连接关系

```python
edge_index.shape = [2, num_edges]
```

例如：

* 0 → 1
* 1 → 2
* 2 → 3

表示哪些节点之间有连接关系。

---

## y

标签（label）

例如：

* 是否是 antideuteron（分类任务）
* 粒子通過位置（回归任务）

这是模型最终要学习预测的目标。

---

## 7. “将单个 event 的 hit 信息转换成 PyG 的 Data 对象”是什么意思

它的意思是：

## 把 ROOT 文件中的 event 数据

转换成：

## GNN 可以直接训练的数据格式

也就是：

```python
torch_geometric.data.Data
```

这种标准结构。

这一步属于：

## 数据预处理（preprocessing）

而且是整个 GNN 任务中非常关键的一步。

---

## 8. 最简单的例子

原始 event：

```python
event = {
    "hit1": [x, y, z, E],
    "hit2": [x, y, z, E],
    ...
}
```

转换成：

```python
Data(
    x=tensor(...),
    edge_index=tensor(...),
    y=label
)
```

这一步就是：

## event → Data object

---

## 9. 总结

```text
PyG = PyTorch Geometric
```

它是：

## 专门用于图神经网络（GNN）的库

而：

```text
event → Data object
```

表示：

## 将物理事件转换成图结构数据

这是：

## GAPS + Machine Learning + GNN

中最核心的一步之一。
