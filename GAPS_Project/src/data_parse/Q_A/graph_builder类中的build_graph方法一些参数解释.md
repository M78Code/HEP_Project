# GNN 中节点特征与 kNN Graph 参数说明

## 1. 为什么这里只使用 5 个特征？

代码中：

```python
x = np.stack([
    positions[:, 0],   # fX
    positions[:, 1],   # fY
    positions[:, 2],   # fZ
    energies,          # energy
    times,             # time
], axis=1)
```

因此：

```text
每个节点 = 5 个特征
```

即：

```text
[x, y, z, E, t]
```

也就是：

* 空间位置（x, y, z）
* 能量沉积（Energy）
* 时间信息（Time）

---

## 为什么是这 5 个？

因为这是最基础、最核心的物理信息。

### ① 空间位置（x, y, z）

表示粒子发生 hit 的位置。

它决定：

## 飞行轨迹（Track）

通过多个 hit 的空间分布，可以重建粒子的运动路径。

---

### ② 能量沉积（Energy）

表示粒子在探测器中留下了多少能量。

它决定：

## 粒子种类识别（PID）

不同粒子的能量沉积模式通常不同。

例如：

* antiproton
* antideuteron

它们的能量分布往往有明显区别。

---

### ③ 时间信息（Time）

表示 hit 发生的时间。

它决定：

## TOF（Time Of Flight）

即飞行时间信息。

这也是 GAPS 中非常重要的物理量。

---

## 为什么不加入更多特征？

并不是不能加，而是：

## 第一版 baseline 应该尽量简单

优先使用：

* 最稳定
* 最重要
* 最容易解释

的特征。

例如未来完全可以继续加入：

* volume_id
* detector type
* beta
* momentum
* trigger info
* charge
* track quality
* angle
* reconstructed variables

使特征矩阵从：

```text
[N, 5]
```

变成：

```text
[N, 10]
[N, 20]
[N, 50]
```

都没有问题。

只是第一版建议先：

## 简单 + 稳定 + 可解释

先把系统跑通。

---

# 2. loop=False 是什么意思？

代码中：

```python
edge_index = knn_graph(
    pos_tensor,
    k=k,
    loop=False
)
```

这里的：

## loop

表示：

## 自连接（self-loop）

也就是：

```text
节点连接自己
```

例如：

```text
3 → 3
7 → 7
10 → 10
```

这种边。

---

## loop=False

表示：

## 不允许节点连接自己

即：

```text
只连接其他节点
```

例如：

```text
3 → 5
3 → 8
3 → 11
```

这是最常见的设置。

---

## loop=True

表示：

## 允许节点连接自己

即：

```text
3 → 3
```

这种边也会被加入图中。

---

## 两者的区别

### loop=False（更常用）

更符合：

## detector hit 之间的物理关系

因为自己连接自己通常没有太大的物理意义。

---

### loop=True（某些模型会需要）

有些 GNN 层，比如：

* GCN
* GraphConv

希望保留：

## 节点自身的原始信息

因此它们常常会自动加入 self-loop。

例如：

```text
GCNConv
```

很多时候内部已经自动处理。

---

## 实际建议

这里推荐：

```python
loop=False
```

因为很多 PyG 模型层本身已经会自动加 self-loop。

如果手动再加，容易重复。

---

# 3. k=8 是什么意思？

代码中：

```python
def build_graph(..., k=8)
```

表示：

## 默认每个节点连接最近的 8 个邻居

也就是：

## 8-nearest neighbors

---

## 举个例子

假设当前 event 有：

```text
100 个 hits
```

那么：

每个 hit 都会去寻找：

```text
距离它最近的 8 个 hit
```

并建立边连接。

例如：

```text
Node 15
→ Node 2
→ Node 7
→ Node 33
→ ...
共 8 个
```

---

## 为什么默认是 8？

这是：

## 一个经验值（经验超参数）

并不是固定的物理定律。

通常：

### 太小

例如：

```text
k = 2
k = 3
```

图会太稀疏：

* 信息传播不足
* 容易漏掉重要关系

---

### 太大

例如：

```text
k = 50
k = 100
```

图会太密集：

* 噪声太多
* 计算量大幅增加
* 容易过拟合

---

因此很多论文通常先尝试：

```text
k = 6
k = 8
k = 10
k = 16
```

再通过实验调优。

---

## 可以修改吗？

当然可以。

完全可以使用：

```python
k = 4
k = 12
k = 20
```

都没有问题。

这属于：

## 超参数（Hyperparameter）

需要通过实验寻找最优值。

---

## 最常见建议

第一版 baseline 推荐：

```python
k = 8
```

这是非常合理且常见的选择。

---

# 总结

```text
5个特征
=
最基础最重要的物理信息

loop=False
=
不连接自己

k=8
=
每个节点连接最近的8个邻居
（经验超参数）
```

这是 GNN 图构建中最经典、最稳定的 baseline 设置。
