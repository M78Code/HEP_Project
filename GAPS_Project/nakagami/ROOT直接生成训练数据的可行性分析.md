# ROOT 直接生成训练数据的可行性分析

本文档说明是否可以直接处理 `rootFiles` 和 `rootFiles_2` 中的 ROOT 文件，生成 CNN+DNN 或 GNN 训练数据，并区分“严格复现中上 7.1.1”和“构造本研究的新 GAPS baseline”两种目标。

## 1. 结论

技术上可以直接从 ROOT 文件生成训练数据。

当前 ROOT 中已经包含生成训练样本所需的核心信息：

```text
volumeId_
totalEnergyDeposition_
meanPosition_
firstTime_
lastTime_
primaryStoppingPosition_
primaryPdg_
primaryBeta_
randomSeed_
```

这些信息足以构造：

```text
hit-level graph data
Si(Li) voxel data
TOF feature data
label
```

但是，是否推荐直接从 ROOT 生成数据，取决于目标。

## 2. 两种不同目标

### 2.1 目标 A：严格复现中上 7.1.1

如果目标是严格复现中上修论 7.1.1 的 CNN+DNN 结果，则不建议一开始就直接从 ROOT 重新生成全部输入。

原因是：

```text
中上 7.1.1 使用的是已经构造好的 CSV 输入。
CSV 中 row[3:1443] 已经是 Si(Li) 1440维能量图。
CSV 中 row[1620:1631] 已经是 TOF 11维特征。
```

直接从 ROOT 重新生成时，TOF 11维已经可以通过对照确认，但 Si(Li) 的 1440维映射仍需进一步确认。

最关键的未确认点是：

```text
Si(Li) volumeId_ 如何映射到 10×12×12 的 channel index。
```

如果这个映射与中上 CSV 中的 `row[3:1443]` 不完全一致，那么生成的数据虽然合理，但不再是中上 7.1.1 的严格复现。

因此，对于严格复现，建议优先使用 1631列 CSV。

### 2.2 目标 B：构造本研究的新 GAPS baseline

如果目标是基于本研究使用的新 GAPS ROOT 数据，构造自己的 CNN+DNN 或 GNN baseline，则可以直接从 ROOT 生成训练数据。

这种情况下，输入表示可以由本研究自行定义，例如：

```text
Si(Li) voxel: 根据 hit 位置进行 voxelize
TOF features: 根据 volume_id、firstTime、position 构造
Graph data: 每个 hit 作为 node，空间或物理关系作为 edge
label: 根据 primaryPdg_ 构造
```

这一路线适合写作：

```text
中上（2021）のCNN+DNN混合構造を参考にし、本研究で用いる新GAPSデータに適用可能な入力表現を構成した。
```

即：

```text
参考中上（2021）的 CNN+DNN 混合结构，并根据本研究的新 GAPS 数据构造输入表示。
```

但不应写成：

```text
中上 7.1.1 を完全に再現した。
```

## 3. 当前 ROOT 文件提供的信息

通过 `uproot` 检查，ROOT 中的主要 tree 为：

```text
TreeMc
SimulationParameterTree
```

`TreeMc` 中关键 branch 包括：

```text
Mc/CEventBase/eventNumber_
Mc/CEventBase/primaryBeta_
Mc/CEventBase/primaryBetaGenerated_
Mc/CEventBase/primaryStoppingVolume_
Mc/CEventBase/primaryStoppingPosition_
Mc/CEventBase/primaryStoppingTime_
Mc/CEventBase/primaryStoppingKE_
Mc/CEventBase/totalEnergyDeposition_
Mc/CEventBase/meanPosition_
Mc/CEventBase/volumeId_
Mc/CEventBase/firstTime_
Mc/CEventBase/lastTime_
Mc/CEventBase/hitTrackIndex_
Mc/CEventBase/appliedTriggerBitMask_
Mc/CEventBase/firedTriggerBitMask_
Mc/primaryPosition_
Mc/primaryTime_
Mc/primaryPdg_
Mc/randomSeed_
Mc/tracks_
```

这些 branch 可用于生成以下训练输入：

| 训练信息 | ROOT branch |
|---|---|
| label | `Mc/primaryPdg_` |
| beta | `Mc/CEventBase/primaryBeta_` |
| hit volume id | `Mc/CEventBase/volumeId_` |
| hit energy | `Mc/CEventBase/totalEnergyDeposition_` |
| hit position | `Mc/CEventBase/meanPosition_` |
| hit time | `Mc/CEventBase/firstTime_`, `lastTime_` |
| stopping position | `Mc/CEventBase/primaryStoppingPosition_` |

## 4. TOF 11维特征已经可以从 ROOT 重建

通过 CSV 与 ROOT 对照，已经确认中上 CSV 中 TOF 11维的定义。

正确的 11维为：

```text
0  outer TOF 最早 hit 的 energy
1  inner TOF 最早 hit 的 energy
2  inner firstTime - outer firstTime
3  inner TOF 最早 hit 的 x
4  inner TOF 最早 hit 的 y
5  inner TOF 最早 hit 的 z
6  outer TOF 最早 hit 的 x
7  outer TOF 最早 hit 的 y
8  outer TOF 最早 hit 的 z
9  stopping position x
10 stopping position y
```

对应 CSV 列为：

```python
row[1620:1631]
```

ROOT 中可通过以下规则区分 TOF：

```text
volume_id 第一位 = 1 → TOF
volume_id 第二位 = 0 → outer TOF
volume_id 第二位 = 1 → inner TOF
```

因此，TOF 部分可以从 ROOT 可靠重建。

## 5. Si(Li) 10×12×12 映射尚需验证

严格复现中上 7.1.1 的最大难点是 Si(Li) 映射。

中上 CSV 中：

```python
row[3:1443]
```

对应：

```text
1440 个 Si(Li) energy 值
reshape 为 10×12×12
```

但是，从 ROOT 中直接生成同样的 1440维输入时，需要知道：

```text
volumeId_ 的哪几位对应 layer
哪几位对应 x/y module
哪几位对应 active Si(Li) channel
这些编号如何线性化为 CSV 中的 0-1439 index
```

如果映射规则不完全一致，则直接从 ROOT 生成的 voxel 与中上 CSV 不等价。

因此，在使用 ROOT 严格复现前，必须先做验证：

```text
读取一个 CSV row
通过 row[0] 找到对应 ROOT 文件
通过 row[1] 找到对应 ROOT entry
从 ROOT 生成 Si(Li) 1440维
与 CSV row[3:1443] 逐项比较
```

只有当二者一致时，才能认为 ROOT→Si(Li) voxel 的转换正确复现了中上输入。

## 6. 推荐路线

### 6.1 短期推荐：使用已有 1631列 CSV

如果目标是尽快跑中上式 CNN+DNN baseline，推荐使用已有 1631列 CSV。

读取方式：

```python
voxel = row[3:1443].reshape(10, 12, 12)
tof = row[1620:1631]
label = row[2]
```

并排除：

```text
*VolID*.csv
```

优点：

```text
不需要重新确认 Si(Li) volumeId 到 1440 channel 的映射。
与中上 CSV 输入定义最接近。
实现成本低。
```

缺点：

```text
CSV 分散在多个目录。
Dbar/Pbar 样本数量需要进一步整理。
```

### 6.2 中期路线：从 ROOT 生成新 GAPS baseline

如果目标是本研究的新 GAPS 数据，推荐从 ROOT 直接生成训练数据。

这种数据可以用于：

```text
GNN 模型
CNN+DNN baseline
Hybrid GNN baseline
```

但在论文中应明确：

```text
该 baseline 参考中上的 CNN+DNN 架构，但输入表示根据本研究的新 GAPS ROOT 数据重新构造。
```

### 6.3 长期路线：验证 ROOT→CSV 完全复现

如果需要彻底复现中上的 preprocessing，则需要补全：

```text
Si(Li) volumeId_ → 10×12×12 channel index 的映射验证
ROOT 中 Atrest/Inflight 事件分类规则
CSV 中其他 flag 列的含义
```

这条路线工作量较大，但可以得到最严格的复现。

## 7. 对论文写作的影响

如果使用 CSV 跑中上式 CNN+DNN，可以写：

```text
中上（2021）のCSV形式の入力データに含まれるSi(Li)三次元エネルギーマップおよびTOF 11次元特徴量を用いて、CNN+DNN baselineを再評価した。
```

如果使用 ROOT 重新生成新 GAPS baseline，应写：

```text
中上（2021）で提案されたCNN+DNN混合構造を参考にし、本研究で用いる新GAPSシミュレーションデータに適用可能な形で、Si(Li) voxelおよびTOF特徴量を構成した。
```

两者不能混用为同一个结论。

## 8. 最终判断

可以直接处理 `rootFiles` 和 `rootFiles_2` 生成新的训练数据。

但是：

```text
若目标是严格复现中上 7.1.1：
  优先使用已有 1631列 CSV。
  ROOT 方案还需要验证 Si(Li) 映射。

若目标是构造本研究的新 GAPS baseline：
  可以直接使用 ROOT。
  但论文中应说明这是基于新 GAPS 数据重新构造的输入表示。
```

