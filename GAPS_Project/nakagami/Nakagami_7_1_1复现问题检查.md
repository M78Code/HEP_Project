# 中上 7.1.1 复现代码与数据检查记录

本文档记录对中上修论 7.1.1 CNN+DNN 复现实验中，40M CSV/ROOT 数据结构、TOF 11维特征、以及当前代码问题的检查结果。

## 1. 数据目录确认

数据目录：

```text
/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M
```

其中包含：

```text
csvFiles
rootFiles
rootFiles_2
gaps_run.mac
```

检查结果：

```text
rootFiles   : 247 个 ROOT 文件
rootFiles_2 : 200 个 ROOT 文件
重复文件名  : 0
```

因此，`rootFiles` 和 `rootFiles_2` 不是重复目录，而是两批不同的 ROOT 数据。

## 2. ROOT 文件结构

ROOT 文件中主要 tree 为：

```text
TreeMc
SimulationParameterTree
```

其中 `TreeMc` 包含如下关键 branch：

```text
Mc/CEventBase/eventNumber_
Mc/CEventBase/primaryBeta_
Mc/CEventBase/primaryBetaGenerated_
Mc/CEventBase/primaryStoppingVolume_
Mc/CEventBase/primaryStoppingPosition_
Mc/CEventBase/totalEnergyDeposition_
Mc/CEventBase/meanPosition_
Mc/CEventBase/volumeId_
Mc/CEventBase/firstTime_
Mc/CEventBase/lastTime_
Mc/primaryPdg_
Mc/randomSeed_
```

这些 branch 足以从 ROOT 中重建 Si(Li) 能量图和 TOF 特征。

注意：`GGeometry` 是几何对象，使用 `uproot` 直接遍历时可能因 `TGeoManager` 反序列化失败而报错。检查数据时应直接读取 `TreeMc`，不要遍历 `GGeometry`。

## 3. CSV 与 ROOT 的对应关系

以 CSV 文件第一行为例：

```text
CSV file:
CNN211209_Dbar_isot_50K_beta02to05_Atrest_000.csv

row[0] = 1639025723
row[1] = 96
row[2] = 1
row[4] = 0.378899
```

对应 ROOT 文件：

```text
dbar_isot_211209_beta0205_115M_FTFP_BERT_HP_1639025723.root
```

检查发现：

```text
row[0] 对应 ROOT 文件名中的 random seed
row[1] 对应 ROOT entry index
row[2] 对应 label
row[4] 对应 primaryBeta
```

重要结论：

```text
CSV row[1] 不是 ROOT 中的 eventNumber_。
CSV row[1] 是 ROOT entry index。
```

因此，如果用 CSV 行回查 ROOT，不能用：

```python
eventNumber_ == row[1]
```

而应使用：

```python
entry_start = int(row[1])
entry_stop = int(row[1]) + 1
```

## 4. TOF 11维特征的正确列范围

通过 ROOT entry 对照，确认中上 7.1.1 CSV 中的 11维 TOF/DNN 输入为：

```python
row[1620:1631]
```

也就是 0-indexed 的：

```text
1620-1630
```

不是：

```python
row[1619:1630]
```

也不是：

```python
row[1620:1630]
```

## 5. TOF 11维特征的物理含义

对照结果如下：

CSV：

```text
1620 7.93059
1621 8.49386
1622 11.6165
1623 270.479
1624 261.412
1625 135
1626 1130.82
1627 173.496
1628 1089.77
1629 -568.838
1630 337.748
```

ROOT 中最早 outer TOF hit：

```text
energy = 7.930594967813818
time   = 10.80201814222092
pos    = 1130.8227, 173.4957, 1089.7
```

ROOT 中最早 inner TOF hit：

```text
energy = 8.493864878114326
time   = 22.418559373278757
pos    = 270.4788, 261.4120, 135.0
```

ROOT 中 stopping position：

```text
x = -569
y = 338
z = -803
```

因此 11维特征可解释为：

| index | CSV column | 含义 |
|---:|---:|---|
| 0 | 1620 | outer TOF 最早 hit 的 energy |
| 1 | 1621 | inner TOF 最早 hit 的 energy |
| 2 | 1622 | inner firstTime - outer firstTime |
| 3 | 1623 | inner TOF 最早 hit 的 x |
| 4 | 1624 | inner TOF 最早 hit 的 y |
| 5 | 1625 | inner TOF 最早 hit 的 z |
| 6 | 1626 | outer TOF 最早 hit 的 x |
| 7 | 1627 | outer TOF 最早 hit 的 y |
| 8 | 1628 | outer TOF 最早 hit 的 z |
| 9 | 1629 | stopping position x |
| 10 | 1630 | stopping position y |

注意：CSV 的 11维 TOF 特征不包含 stopping position z。

## 6. 当前代码中的主要问题

### 6.1 `preprocess_40M.py` 中 TOF 列错位

文件：

```text
/Users/lind/Desktop/papers/DeepLearning/nakagami/data_parse/preprocess_40M.py
```

当前问题位置：

```text
第 10 行：Col 1619~1629: 11维TOF聚合特征
第 28 行：TOF_START = 1619
第 29 行：TOF_END   = 1630
```

问题：

```text
row[1619:1630] 会取到 1619-1629。
这会错误地把 row[1619] 的 flag 混入 TOF 输入，并丢掉 row[1630] 的 stopping y。
```

应修改为：

```python
TOF_START = 1620
TOF_END   = 1631
```

并把注释改为：

```text
Col 1620~1630: 11维TOF聚合特征
```

### 6.2 `preprocess_40M.py` 没有排除 VolID 文件

文件：

```text
/Users/lind/Desktop/papers/DeepLearning/nakagami/data_parse/preprocess_40M.py
```

当前问题位置：

```text
第 109 行：file_list = sorted(pathlib.Path(args.csv_dir).glob("CNN*.csv"))
```

问题：

```text
CNN*_VolID_*.csv 与 Atrest/Inflight CSV 的列结构不同。
如果 VolID 文件混入训练，会造成数据格式错误或训练数据污染。
```

应修改为：

```python
file_list = sorted(
    p for p in pathlib.Path(args.csv_dir).glob("CNN*.csv")
    if "VolID" not in p.name
)
```

### 6.3 `nakagami_model.py` 中 DNN 输入维度错误

文件：

```text
/Users/lind/Desktop/papers/DeepLearning/nakagami/models/nakagami_model.py
```

当前问题位置：

```text
第 37 行：input2 (TOF paddle): (B, 9)
第 65 行：nn.Linear(9, 256)
```

问题：

```text
中上 7.1.1 使用的是 11维 TOF/DNN 输入。
当前模型仍是 9维输入，因此不是 7.1.1 的正确复现。
```

应修改为：

```python
nn.Linear(11, 256)
```

并将注释改为：

```text
input2 (TOF features): (B, 11)
```

### 6.4 `test_preprocess_40M.sh` 测试脚本也可能复制 VolID 文件

文件：

```text
/Users/lind/Desktop/papers/DeepLearning/nakagami/bash/test_preprocess_40M.sh
```

当前问题位置：

```text
第 14 行：ls $CSV_DIR/CNN*.csv | head -10 | xargs -I{} cp {} $TEST_DIR/
```

问题：

```text
测试数据中也可能混入 VolID 文件。
```

应修改为：

```bash
find $CSV_DIR -maxdepth 1 -name "CNN*.csv" ! -name "*VolID*" | head -10 | xargs -I{} cp {} $TEST_DIR/
```

## 7. 如果使用 GAPS_Project/src 的 hybrid pipeline，还存在另一个问题

文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/data_parse/voxelizer.py
```

当前 `build_tof_features()` 的定义为：

```text
[outer_e, inner_e, outer_n, inner_n, time_of_flight,
 outer_entry_x, outer_entry_y, outer_entry_z,
 inner_entry_x, inner_entry_y, inner_entry_z]
```

这与中上 CSV 的 11维定义不同。

中上 CSV 对应的 11维为：

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

因此：

```text
如果目标是复现中上 7.1.1，应优先使用 DeepLearning/nakagami 的 CSV pipeline。
如果使用 GAPS_Project/src/scripts/train_hybrid.py，则需要重写 voxelizer.py 中的 build_tof_features()。
```

## 8. 可能导致复现结果偏差的原因

目前已经确认的高风险问题包括：

```text
1. TOF 11维输入错位一列。
2. 模型 DNN branch 仍使用 9维输入。
3. VolID CSV 文件可能混入训练。
4. GAPS_Project/src 中的 TOF 11维定义与中上 CSV 不一致。
5. CSV row[1] 被误认为 ROOT eventNumber_，实际是 ROOT entry index。
```

这些问题都可能导致复现结果明显低于中上论文结果。其中第 1、2、3 点会直接影响 CNN+DNN 复现实验。

## 9. 建议的修改优先级

优先级 1：

```text
修改 preprocess_40M.py：
- TOF_START = 1620
- TOF_END = 1631
- 排除 VolID 文件
```

优先级 2：

```text
修改 nakagami_model.py：
- DNN 输入从 9维改为 11维
```

优先级 3：

```text
修改 test_preprocess_40M.sh：
- 测试文件复制时排除 VolID
```

优先级 4：

```text
如果继续使用 GAPS_Project/src 的 hybrid pipeline：
- 重新实现 voxelizer.py 的 build_tof_features()
```

## 10. 当前结论

如果目标是复现中上修论 7.1.1，则当前 `DeepLearning/nakagami` 中的代码还不是正确复现。

最核心的问题是：

```text
TOF 输入列范围错误，且模型 DNN 输入维度错误。
```

正确的 11维输入应来自：

```python
row[1620:1631]
```

模型 DNN branch 应接收：

```text
(B, 11)
```

