# 中上 40M CSV 复现执行方案

本文档说明如何使用当前 `GAPS_Project/src` 中的 CNN+DNN 模型，基于中上 40M 相关 CSV 数据生成训练用 `.npz`，并单独训练一个不污染新 GAPS pipeline 的 CNN+DNN baseline。

## 1. 总体策略

采用：

```text
CSV 复现优先，ROOT 验证/补充后置
```

原因：

```text
中上 7.1.1 的输入已经在 CSV 中构造好。
直接使用 CSV 可以避免重新确认 Si(Li) volumeId -> 10×12×12 channel index 的映射。
ROOT 可用于验证 TOF 定义和后续补充，但不作为第一阶段 preprocessing 主线。
```

当前建议：

```text
不要修改 voxelizer.py。
不要覆盖当前新 GAPS 的 train_hybrid.py。
新增两个独立脚本：
  preprocess_nakagami40m_csv.py
  train_hybrid_nakagami40m.py
```

## 2. 使用的数据目录

目前确认可用的 1631列 Mc CSV 分散在两个目录。

Dbar 目录：

```text
/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
```

Pbar 目录：

```text
/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc
```

说明：

```text
Dbar = antideuteron = 反重陽子 = signal = label 1
Pbar = antiproton   = 反陽子   = background = label 0
```

不要使用：

```text
/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized
```

原因：

```text
csvFiles_Digitized 的非 VolID CSV 是 1452 列，
不是本方案使用的 1631列 Mc CSV 格式。
```

## 3. 关键风险：Dbar 与 Pbar 来自不同目录

当前 Dbar 和 Pbar 并不在同一个 CSV 目录中：

| 项目 | Dbar 来源 | Pbar 来源 |
|---|---|---|
| 目录 | `211209_isot_0205_renewal_looseTrigger_40M/csvFiles` | `220104_4Mevents_isot_loose/csvFiles_Mc` |
| 数据规模标识 | 40M | 4M |
| CSV 列数 | 1631 | 1631 |
| 粒子类别 | Dbar | Pbar |

这带来一个重要风险：

```text
如果两个目录的数据生成条件、探测器几何、Si(Li) channel 映射或触发条件不同，
模型可能学到的是“目录/生成条件差异”，而不是 Dbar/Pbar 的粒子差异。
```

这种情况属于隐式数据泄露，会导致结果虚高或失真。因此，在合并两个目录训练前，必须先做前置验证。

## 4. Step 0：验证两个目录的 CSV 格式一致

先确认两个目录均为 1631列 Mc CSV，并且 TOF 区域位置一致。

服务器命令：

```bash
python - <<'PY'
import csv, glob, os

dirs = [
    "/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles",
    "/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc",
]

for d in dirs:
    print("\n====", d, "====")
    files = sorted(glob.glob(os.path.join(d, "CNN*.csv")))
    files = [f for f in files if "VolID" not in os.path.basename(f)]
    print("non-VolID files:", len(files))
    for f in files[:3]:
        with open(f) as fh:
            row = next(csv.reader(fh))
        print(os.path.basename(f), "cols:", len(row), "label:", row[2])
        print("first 8:", row[:8])
        print("1620-1630:", row[1620:1631] if len(row) >= 1631 else "N/A")
PY
```

最低条件：

```text
1. 非 VolID 文件列数均为 1631。
2. row[2] 的 label 正确：Dbar=1, Pbar=0。
3. row[1620:1631] 看起来是连续的 TOF 11维物理量。
4. 不能混入 1452列 Digitized CSV。
```

## 5. Step 1：Si(Li) 映射一致性的抽样验证

仅确认列数一致还不够。还需要检查 Dbar/Pbar 两个目录中的 Si(Li) 1440维区域是否具有一致的编码风格。

建议抽样比较：

```text
row[3:1443]
```

统计：

```text
非零 channel 数
总能量
最大值
最大值所在 index
非零 index 的范围
非零 index 在 10×12×12 中的 layer 分布
```

服务器命令：

```bash
python - <<'PY'
import csv, glob, os
import numpy as np

sources = {
    "Dbar_40M": "/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles",
    "Pbar_4M": "/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc",
}

for name, d in sources.items():
    print("\n====", name, d, "====")
    files = sorted(glob.glob(os.path.join(d, "CNN*.csv")))
    files = [f for f in files if "VolID" not in os.path.basename(f)]
    files = [f for f in files if (("Dbar" in f) if name.startswith("Dbar") else ("Pbar" in f))]

    stats = []
    for f in files[:5]:
        with open(f) as fh:
            reader = csv.reader(fh)
            for _, row in zip(range(200), reader):
                if len(row) != 1631:
                    continue
                si = np.asarray(row[3:1443], dtype=np.float32)
                nz = np.flatnonzero(si)
                if len(nz) == 0:
                    continue
                layer_counts = np.bincount(nz // 144, minlength=10)
                stats.append([
                    len(nz),
                    float(si.sum()),
                    float(si.max()),
                    int(si.argmax()),
                    int(nz.min()),
                    int(nz.max()),
                    layer_counts.tolist(),
                ])

    if not stats:
        print("no valid rows")
        continue

    arr = np.array([s[:6] for s in stats], dtype=float)
    print("n sampled:", len(stats))
    print("nonzero mean/std:", arr[:,0].mean(), arr[:,0].std())
    print("energy sum mean/std:", arr[:,1].mean(), arr[:,1].std())
    print("max energy mean/std:", arr[:,2].mean(), arr[:,2].std())
    print("argmax min/max:", arr[:,3].min(), arr[:,3].max())
    print("nz index min/max:", arr[:,4].min(), arr[:,5].max())
    print("example layer_counts:", stats[0][6])
PY
```

判断原则：

```text
如果 Dbar/Pbar 的非零 index 范围、layer 分布、channel index 风格明显不同，
则不能直接混合作为严格复现数据。
```

注意：Dbar/Pbar 的物理能量分布本来可能不同，因此不能要求统计量完全一致。这里主要检查的是 channel index 编码是否显著不一致。

## 6. CSV 列定义

通过 ROOT/CSV 对照，确认 1631列 CSV 的关键列如下。

基本列：

```text
row[0] = ROOT 文件 random seed
row[1] = ROOT entry index
row[2] = label
row[4] = primaryBeta
```

注意：

```text
row[1] 不是 ROOT 的 eventNumber_。
row[1] 是 ROOT entry index。
```

Si(Li) 输入：

```python
voxel = row[3:1443].reshape(10, 12, 12)
```

TOF 11维输入：

```python
tof = row[1620:1631]
```

不要使用：

```python
row[1619:1630]
```

因为这会把一个 flag 混入 TOF 输入，并丢掉最后一个特征。

## 7. TOF 11维定义

`row[1620:1631]` 对应：

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

## 8. 新增脚本 1：preprocess_nakagami40m_csv.py

建议新建文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/scripts/preprocess_nakagami40m_csv.py
```

用途：

```text
读取中上 1631列 CSV，生成 HybridDatasetFast 可直接读取的 .npz。
```

该脚本应做：

```text
1. 支持多个 CSV 输入目录。
2. 读取 CNN*.csv。
3. 排除 *VolID* 文件。
4. 只保留 len(row) == 1631 的文件。
5. 从 row[3:1443] 读取 Si(Li) 1440维并 reshape 为 (10, 12, 12)。
6. 从 row[1620:1631] 读取 TOF 11维。
7. 从 row[2] 读取 label。
8. 按 Dbar/Pbar 文件分层划分 train/val。
9. 输出 HybridDatasetFast 需要的 key：
   - voxels
   - tofs
   - labels
```

建议添加测试参数：

```python
parser.add_argument(
    "--max_files_per_class",
    type=int,
    default=None,
    help="各粒子種ごとの最大ファイル数（テスト用）",
)
```

在分割前限制：

```python
if args.max_files_per_class is not None:
    dbar_files = dbar_files[:args.max_files_per_class]
    pbar_files = pbar_files[:args.max_files_per_class]
```

输出目录建议：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/dataset/nakagami40M
```

输出文件：

```text
train_hybrid_nakagami40M.npz
val_hybrid_nakagami40M.npz
```

重要：`.npz` key 必须是复数形式：

```python
np.savez_compressed(
    out_path,
    voxels=np.stack(voxels),
    tofs=np.stack(tofs),
    labels=np.array(labels, dtype=np.int64),
)
```

因为当前 `HybridDatasetFast` 读取的是：

```python
self.voxels = data["voxels"]
self.tofs = data["tofs"]
self.labels = data["labels"]
```

## 9. 新增脚本 2：train_hybrid_nakagami40m.py

建议新建文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/scripts/train_hybrid_nakagami40m.py
```

用途：

```text
单独训练中上 40M CSV 版本 CNN+DNN baseline，
避免污染当前新 GAPS 的 train_hybrid.py。
```

该脚本可以基本复制当前：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/src/scripts/train_hybrid.py
```

但修改数据路径和保存路径。

建议修改：

```python
DATA_DIR = PROJECT_ROOT / "dataset" / "nakagami40M"
SAVE_PATH = PROJECT_ROOT / "results" / "cnn_dnn_hybrid_nakagami40M_best.pth"
```

读取数据：

```python
train_set = HybridDatasetFast(DATA_DIR / "train_hybrid_nakagami40M.npz")
val_set = HybridDatasetFast(DATA_DIR / "val_hybrid_nakagami40M.npz")
```

模型保持：

```python
model = CNNDNNHybrid(tof_dim=11).to(DEVICE)
```

训练参数建议保持中上 7.1.1 对齐：

```python
BATCH_SIZE = 200
LR = 4e-5
```

## 10. 推荐执行顺序

### 10.1 前置验证

先执行：

```text
Step 0: CSV 格式一致性验证
Step 1: Si(Li) 映射一致性抽样验证
```

如果发现明显目录差异，暂停合并训练，重新寻找同一生成条件下的 Dbar/Pbar 数据。

### 10.2 小样本测试

不要一开始处理全部数据。先每类取少量文件测试：

```text
--max_files_per_class 2
```

检查：

```text
voxels shape: (N, 10, 12, 12)
tofs shape:   (N, 11)
labels:       同时包含 0 和 1
```

### 10.3 小样本训练

用 small npz 跑 1-2 epoch，确认：

```text
DataLoader 可以读取
模型 forward 正常
loss 正常下降
没有 KeyError
没有 shape mismatch
```

### 10.4 平衡数据训练

建议先使用相同数量的 Dbar/Pbar 文件，例如：

```text
Dbar 20 files
Pbar 20 files
```

这样可以先验证模型是否能在较平衡数据上学习。

### 10.5 全量或中等规模处理

如果一次性处理接近千万事件，内存和磁盘压力会很大。若数据量过大，应考虑 shard 保存：

```text
train_000.npz
train_001.npz
...
```

第一阶段可以先用单个 npz 验证流程。

## 11. 当前可用文件数量与风险

目前确认：

```text
Dbar 1631列 Mc CSV:
  /211209_isot_0205_renewal_looseTrigger_40M/csvFiles
  非 VolID Dbar 文件约 192 个

Pbar 1631列 Mc CSV:
  /220104_4Mevents_isot_loose/csvFiles_Mc
  非 VolID Pbar 文件约 20 个
```

如果每个 Dbar 文件约 50K event、每个 Pbar 文件约 40K event，则粗略估计：

```text
Dbar ≈ 9.6M
Pbar ≈ 0.8M
```

因此全量训练时仍可能存在类别不平衡。第一阶段建议先使用各 20 个文件。

## 12. 与 ROOT 方案的关系

ROOT 方案暂时后置。

原因：

```text
TOF 11维已经能从 ROOT 对照确认。
但 Si(Li) volumeId -> 10×12×12 channel index 的映射尚未完全验证。
```

如果之后要从 ROOT 完整生成训练数据，需要额外完成：

```text
1. 选取 CSV row。
2. 通过 row[0] 定位 ROOT 文件。
3. 通过 row[1] 定位 ROOT entry。
4. 从 ROOT 重建 Si(Li) 1440维。
5. 与 CSV row[3:1443] 逐项比较。
```

## 13. 论文中的表述建议

如果使用本方案跑中上 CSV，可以写：

```text
中上（2021）のCSV形式の入力データに含まれるSi(Li)三次元エネルギーマップおよびTOF 11次元特徴量を直接用い、CNN+DNN baselineの再評価を行った。
```

中文：

```text
直接使用中上（2021）CSV 输入数据中保存的 Si(Li) 三维能量图和 TOF 11维特征，对 CNN+DNN baseline 进行了重新评估。
```

如果使用新 GAPS 数据上的 `voxelizer.py` pipeline，应另写为：

```text
中上（2021）のCNN+DNN混合構造を参考にし、本研究で用いる新GAPSシミュレーションデータに適用可能な入力表現を構成した。
```

两者不要混为同一个实验。

## 14. 最终建议

当前最推荐执行路线：

```text
1. 先做 Step 0：CSV 格式一致性验证。
2. 再做 Step 1：Si(Li) 映射一致性抽样验证。
3. 新建 preprocess_nakagami40m_csv.py。
4. 新建 train_hybrid_nakagami40m.py。
5. 先用 Dbar/Pbar 各 2 个文件生成 small npz。
6. 训练 1-2 epoch 验证 pipeline。
7. 再用 Dbar/Pbar 各 20 个文件生成较平衡数据。
8. 记录结果，并与新 GAPS baseline 分开讨论。
```

