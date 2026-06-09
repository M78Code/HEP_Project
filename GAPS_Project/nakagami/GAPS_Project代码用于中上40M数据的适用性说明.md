# GAPS_Project CNN+DNN 代码用于中上 40M 数据的适用性说明

本文档说明当前 `GAPS_Project/src` 中的 CNN+DNN hybrid 代码是否可以用于运行中上修论 7.1.1 的 40M 数据，以及哪些部分可以复用、哪些部分需要修改。

## 1. 结论

当前 `GAPS_Project/src` 中的 CNN+DNN 模型架构可以用于跑中上 40M 数据。

但是，当前 `voxelizer.py` 的输入处理方式不能直接用于中上 40M CSV 数据。

更准确地说：

```text
cnn_dnn_hybrid.py 的模型结构可以复用。
train_hybrid.py / evaluate_hybrid.py 的训练评估框架可以复用。
HybridDatasetFast 可以复用。
voxelizer.py 中从 event/pkl 重新构造 voxel 和 TOF 特征的逻辑，不应直接用于中上 40M CSV。
```

## 2. 当前 GAPS_Project 代码的定位

当前代码位置：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src
```

主要文件：

```text
data_parse/hybrid_dataset.py
data_parse/voxelizer.py
models/cnn_dnn_hybrid.py
scripts/train_hybrid.py
scripts/evaluate_hybrid.py
```

这套代码更适合描述为：

```text
参考中上（2021）的 CNN+DNN 混合结构，为本研究的新 GAPS 数据构造 baseline。
```

不宜直接描述为：

```text
完全复现中上 7.1.1。
```

原因是新 GAPS 数据和中上旧 GAPS 数据在 detector geometry、volume_id 编码、Si(Li) 排列、TOF 特征定义上可能不同。

## 3. 可以复用的部分

### 3.1 CNN+DNN 模型结构可以复用

文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/models/cnn_dnn_hybrid.py
```

当前模型使用：

```python
CNNDNNHybrid(tof_dim=11)
```

模型输入为：

```text
voxel: (B, 1, 10, 12, 12) 或兼容的三维 Si(Li) voxel
tof:   (B, 11)
```

这一点与中上 7.1.1 的 CNN+DNN 思路一致：

```text
Si(Li) 三维能量图输入 CNN
TOF 11维物理量输入 DNN
CNN/DNN 特征合并后二分类
```

因此，该模型可用于中上 40M CSV 数据。

### 3.2 训练脚本可以复用

文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/scripts/train_hybrid.py
```

该脚本中：

```python
model = CNNDNNHybrid(tof_dim=11).to(DEVICE)
```

训练参数也基本符合中上 7.1.1 的设定方向：

```text
BATCH_SIZE = 200
LR = 4e-5
```

因此，训练框架可以复用。

需要注意的是：

```text
EPOCHS、early stopping、数据划分方式是否与中上完全一致，需要另行确认。
```

### 3.3 HybridDatasetFast 可以复用

文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/data_parse/hybrid_dataset.py
```

`HybridDatasetFast` 从 `.npz` 读取：

```python
voxels
tofs
labels
```

如果将中上 40M CSV 预处理为同样的 `.npz` key，就可以直接使用该 Dataset。

目标 `.npz` 格式应为：

```text
voxels: (N, 10, 12, 12), float32
tofs:   (N, 11), float32
labels: (N,), int64 或 int8
```

## 4. 不能直接复用的部分

### 4.1 voxelizer.py 不能直接用于中上 40M CSV

文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/data_parse/voxelizer.py
```

当前 `build_sili_voxel()` 和 `build_tof_features()` 是从 event/pkl 中的 hit 信息重新构造输入。

这适合本研究的新 GAPS 数据处理，但不适合直接复现中上 40M CSV。

中上 40M CSV 已经包含预先构造好的输入：

```text
Si(Li) 1440维能量图
TOF 11维特征
label
```

因此，对于中上 40M 数据，应直接从 CSV 固定列读取，而不是重新用 `voxelizer.py` 构造。

### 4.2 当前 voxelizer.py 的 TOF 11维定义与中上 CSV 不一致

当前 `voxelizer.py` 中的 TOF 11维定义为：

```text
[outer_e,
 inner_e,
 outer_n,
 inner_n,
 time_of_flight,
 outer_entry_x,
 outer_entry_y,
 outer_entry_z,
 inner_entry_x,
 inner_entry_y,
 inner_entry_z]
```

而通过中上 40M CSV 与 ROOT 对照，确认中上 CSV 的 11维为：

```text
[outer_first_energy,
 inner_first_energy,
 inner_first_time - outer_first_time,
 inner_first_x,
 inner_first_y,
 inner_first_z,
 outer_first_x,
 outer_first_y,
 outer_first_z,
 stopping_x,
 stopping_y]
```

因此，若目标是跑中上 40M 数据，不能使用当前 `voxelizer.py` 生成 TOF 特征。

## 5. 中上 40M CSV 的正确读取方式

通过 ROOT/CSV 对照，确认中上 40M CSV 的列关系如下。

### 5.1 基本列

```text
row[0] = ROOT 文件 random seed
row[1] = ROOT entry index
row[2] = label
row[4] = primaryBeta
```

注意：

```text
row[1] 不是 ROOT 中的 eventNumber_。
row[1] 是 ROOT entry index。
```

### 5.2 Si(Li) 输入

```python
si = row[3:1443]
voxel = si.reshape(10, 12, 12)
```

即：

```text
3-1442 共 1440 个值
```

### 5.3 TOF 11维输入

正确读取方式：

```python
tof = row[1620:1631]
```

即：

```text
1620-1630 共 11 个值
```

不要使用：

```python
row[1619:1630]
```

因为这会把一个 flag 混入 TOF 输入，并丢掉最后一个特征。

## 6. 中上 40M 数据 preprocessing 应该如何写

如果使用当前 `GAPS_Project/src` 的模型和训练框架，建议新增一个专门的 preprocessing 脚本，例如：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/scripts/preprocess_nakagami40m_csv.py
```

该脚本应做：

```text
1. 读取 /mnt/ynakagami3/.../csvFiles/CNN*.csv
2. 排除 *VolID* 文件
3. 从 row[3:1443] 读取 Si(Li) 1440维并 reshape 为 (10, 12, 12)
4. 从 row[1620:1631] 读取 TOF 11维
5. 从 row[2] 读取 label
6. 保存为 HybridDatasetFast 可读取的 npz：
   - voxels
   - tofs
   - labels
```

目标输出文件例如：

```text
train_hybrid_nakagami40M.npz
val_hybrid_nakagami40M.npz
```

## 7. 需要避免的错误

### 7.1 不要混入 VolID 文件

中上 CSV 文件中包含：

```text
CNN*_Atrest_*.csv
CNN*_Inflight_*.csv
CNN*_VolID_*.csv
```

其中 `VolID` 文件结构不同，不应作为训练输入。

应排除：

```python
if "VolID" not in filename
```

### 7.2 不要用 voxelizer.py 重新计算中上 CSV 的 TOF

如果目标是跑中上 40M 数据，应直接使用 CSV 中已经生成好的 11维 TOF 特征。

### 7.3 不要把 row[1] 当作 eventNumber_

通过 ROOT 对照确认：

```text
row[1] = ROOT entry index
```

如果需要从 CSV 回查 ROOT，应使用 entry index，而不是 `eventNumber_`。

## 8. 对当前代码的评价

当前 `GAPS_Project/src` 代码可以分为两层：

```text
模型层：基本可以用于中上 40M 数据。
数据处理层：不能直接用于中上 40M 数据。
```

因此，最准确的判断是：

```text
当前 CNN+DNN 模型架构可以作为中上 7.1.1 的 baseline 架构使用。
但当前 preprocessing 是为本研究的新 GAPS 数据构造的，不是中上 40M CSV 的严格复现处理。
```

## 9. 写论文时建议的表述

如果描述新 GAPS 数据上的 CNN+DNN baseline，建议写：

```text
本研究では、中上（2021）で提案されたCNN+DNN混合構造を参考にし、本研究で用いる新GAPSシミュレーションデータに適用可能な形で入力表現を構成した。
```

中文含义：

```text
本研究参考中上（2021）提出的 CNN+DNN 混合结构，并根据本研究所使用的新 GAPS 模拟数据构造了适用的输入表示。
```

如果之后使用中上 40M CSV 重新跑，则可以写：

```text
中上（2021）の40M CSVデータに対しては、同データ中に保存されたSi(Li)三次元エネルギーマップおよびTOF 11次元特徴量を直接用い、CNN+DNN baselineの再評価を行った。
```

## 10. 最终结论

当前代码可以用于跑中上 40M 数据，但需要新增或修改 preprocessing。

推荐方案：

```text
复用：
- cnn_dnn_hybrid.py
- train_hybrid.py
- evaluate_hybrid.py
- HybridDatasetFast

新增：
- preprocess_nakagami40m_csv.py

不要用于中上 40M CSV：
- voxelizer.py 中的 build_tof_features()
```

