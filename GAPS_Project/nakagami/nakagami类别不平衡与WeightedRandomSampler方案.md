# nakagami 类别不平衡与 WeightedRandomSampler 方案

本文档记录中上 Atrest CSV 数据中 Dbar/Pbar 类别不平衡问题，并给出当前推荐的训练处理方案。

## 1. 数据规模确认

对 Atrest CSV 文件逐一统计行数后，确认：

| 类别 | 文件数 | 每文件事件数 | 总事件数 |
|---|---:|---:|---:|
| Dbar Atrest | 96 | 40,000 | 3,840,000 |
| Pbar Atrest | 10 | 2,000 | 20,000 |

因此 Atrest 数据的类别比例约为：

```text
Dbar : Pbar = 192 : 1
```

这里：

```text
Dbar = 反重陽子 = signal = label 1
Pbar = 反陽子   = background = label 0
```

## 2. 为什么会出现这种不平衡

文件名中的：

```text
Dbar 50K
Pbar 40K
```

表示生成或输入阶段的规模标识，不代表最终 CSV 中保留下来的 Atrest event 数。

Atrest CSV 是经过触发或事件选择后得到的子集，因此实际行数为：

```text
Dbar Atrest: 每文件约 40,000 rows
Pbar Atrest: 每文件约 2,000 rows
```

这说明 Pbar Atrest 在当前可用 CSV 中统计量明显不足。

## 3. 普通训练的风险

如果直接使用普通 `shuffle=True` 训练，例如：

```text
Dbar 10 files + Pbar 10 files
```

实际事件数约为：

```text
Dbar ≈ 400,000
Pbar ≈ 20,000
```

比例仍为：

```text
20 : 1
```

这种情况下模型可能学到：

```text
几乎全部预测为 Dbar
```

并仍然得到很高的 accuracy。

因此：

```text
accuracy 不能作为主要评价指标。
```

应重点关注：

```text
ROC AUC
confusion matrix
Pbar rejection
rejection power at fixed signal efficiency
```

## 4. 可选处理方案

| 方案 | 数据使用方式 | 优点 | 缺点 |
|---|---|---|---|
| A. 原比例 + 加权 loss | 使用全部数据，loss 加权 | 保留全部数据 | 可能仍受 batch 不平衡影响 |
| B. 全下采样 | Dbar 下采样到 Pbar 数量 | 类别严格平衡 | 丢弃大量 Dbar |
| C. 中等下采样 + 加权 | 例如 Dbar:Pbar = 3:1 | 折中 | 需要调参 |
| D. WeightedRandomSampler | 保留数据，每 batch 近似平衡 | 简单有效，保留全部 Pbar | Pbar 会被重复采样 |

当前推荐：

```text
方案 D：WeightedRandomSampler
```

## 5. 为什么推荐 WeightedRandomSampler

WeightedRandomSampler 的思路是：

```text
数据集保留全部样本。
训练时按类别反比设置采样概率。
每个 batch 中 Dbar/Pbar 近似平衡。
```

优点：

```text
1. 不丢弃 Pbar。
2. 不需要直接复制数据文件。
3. 每个 batch 都能看到 Pbar。
4. 比仅仅使用 shuffle=True 更适合严重类别不平衡。
```

需要注意：

```text
Pbar 样本会在一个 epoch 中被重复采样。
```

这是当前 Pbar 统计量不足时可以接受的折中。

## 6. train_hybrid.py 修改方案

目标文件：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/scripts/train_hybrid.py
```

### 6.1 修改 import

原来：

```python
from torch.utils.data import DataLoader
```

改为：

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
```

并添加：

```python
import numpy as np
```

### 6.2 创建 WeightedRandomSampler

在创建 `train_loader` 前加入：

```python
train_labels = train_set.labels
n_pbar = int((train_labels == 0).sum())
n_dbar = int((train_labels == 1).sum())

print(f'train label counts: Pbar={n_pbar:,}, Dbar={n_dbar:,}')

sample_weights = np.where(
    train_labels == 0,
    1.0 / max(n_pbar, 1),
    1.0 / max(n_dbar, 1),
)

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)
```

### 6.3 修改 train_loader

原来：

```python
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
```

改为：

```python
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
```

注意：

```text
使用 sampler 时不要再设置 shuffle=True。
```

### 6.4 val_loader 保持不变

验证集不使用 sampler：

```python
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
```

验证集应尽量反映真实数据分布。

## 7. 是否需要 pos_weight

第一版不建议同时使用：

```text
WeightedRandomSampler + pos_weight
```

原因：

```text
WeightedRandomSampler 已经在采样层面对类别不平衡进行了补偿。
再使用 pos_weight 容易过度补偿。
```

因此当前建议：

```python
criterion = nn.BCEWithLogitsLoss()
```

保持不变。

## 8. preprocessing 建议

如果采用 WeightedRandomSampler，则 preprocessing 阶段不需要强行做事件级下采样。

推荐先生成：

```text
Dbar 10 files + Pbar 10 files
```

即：

```bash
--max_files_per_class 10
```

预期事件数约为：

```text
Dbar ≈ 400,000
Pbar ≈ 20,000
```

然后在训练时通过 WeightedRandomSampler 使 batch 近似平衡。

## 9. 评价指标建议

由于原始验证集可能仍然严重不平衡，评价时不要只看：

```text
accuracy
```

应重点看：

```text
ROC AUC
confusion matrix
background rejection
rejection power at fixed signal efficiency
```

当前 `evaluate_hybrid.py` 已经包含：

```text
ROC AUC
confusion matrix
background rejection
rejection curve
rejection at fixed signal efficiency
```

这些比单纯 accuracy 更有意义。

## 10. 推荐执行顺序

当前推荐：

```text
1. 保持 preprocess_40M.py 的 Atrest-only 策略。
2. run_preprocess_40M.sh 使用 --max_files_per_class 10。
3. 生成 Dbar 10 + Pbar 10 的 Atrest 数据。
4. 修改 train_hybrid.py，加入 WeightedRandomSampler。
5. 训练 CNN+DNN。
6. 使用 evaluate_hybrid.py 评估。
7. 论文中报告 AUC、rejection、confusion matrix，而不是只报告 accuracy。
```

## 11. 论文表述建议

日语表述示例：

```text
本データセットでは反重陽子事象と反陽子事象の数に大きな不均衡が存在するため、学習時には各クラスのサンプル数の逆数に比例した重みを用いる WeightedRandomSampler を導入した。これにより、各 mini-batch において信号事象と背景事象がほぼ均衡してサンプリングされるようにした。一方、評価時にはサンプリングを行わず、検証データの分布に基づいて ROC AUC および Rejection Power を評価した。
```

中文含义：

```text
由于本数据集中反重氘核事件和反质子事件数量存在显著不平衡，训练时引入了按各类别样本数倒数加权的 WeightedRandomSampler，使每个 mini-batch 中信号和背景事件近似平衡。评价时不进行采样，而是基于验证数据本身的分布计算 ROC AUC 和 Rejection Power。
```

