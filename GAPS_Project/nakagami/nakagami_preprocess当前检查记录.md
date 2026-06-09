# nakagami preprocessing 当前检查记录

本文档记录对当前 `GAPS_Project/nakagami` 中 preprocessing 脚本的检查结果，重点说明 Atrest/Inflight 列数差异、当前只使用 Atrest 的合理性，以及后续运行参数建议。

## 1. 检查对象

当前检查的主要文件为：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
```

相关脚本：

```text
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/bash/test_preprocess_40M.sh
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/bash/run_preprocess_40M.sh
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/scripts/train_hybrid.py
/Users/lind/Desktop/ppt/HEP_Project/GAPS_Project/nakagami/scripts/evaluate_hybrid.py
```

## 2. 数据列数诊断结果

对两个 CSV 目录进行检查后，确认：

```text
Atrest   文件：1631 列
Inflight 文件：1632 列
```

具体结果：

```text
Dbar_40M:
  Atrest   97 files, 1631 columns
  Inflight 97 files, 1632 columns

Pbar_4M:
  Atrest   10 files, 1631 columns
  Inflight 10 files, 1632 columns
```

因此，如果 preprocessing 严格要求：

```python
len(row) == 1631
```

则所有 `Inflight` 文件都会被跳过。

这正是之前小样本测试中出现以下异常的原因：

```text
train: 只有 Dbar
val:   只有 Pbar
```

当时部分 split 中抽到了 `Inflight` 文件，但由于 `Inflight` 是 1632 列，被整文件跳过，导致某个类别消失。

## 3. 当前策略：只使用 Atrest

当前 `preprocess_40M.py` 已明确采用：

```text
只使用 Atrest 文件
排除 Inflight 文件
排除 VolID 文件
```

对应代码逻辑：

```python
files = sorted(
    p for p in d.glob("CNN*.csv")
    if "VolID" not in p.name
    and "Inflight" not in p.name
)
```

在该策略下：

```text
Atrest 1631列与 TOF_START=1620, TOF_END=1631 完全一致。
```

因此当前常量：

```python
TOF_START = 1620
TOF_END = 1631
N_COLS = 1631
```

对于 Atrest 数据是正确的。

## 4. 为什么暂时不使用 Inflight

`Inflight` 文件为 1632 列，说明其列结构相对于 Atrest 多出一列。

虽然可以尝试用：

```text
Inflight TOF = row[1621:1632]
```

来适配，但目前尚未通过 ROOT/CSV 对照验证 `Inflight` 的 11维 TOF 列定义。

为了避免引入未经确认的错位特征，当前更稳妥的策略是：

```text
第一阶段只使用 Atrest。
```

这样可以先验证 CNN+DNN pipeline 是否正常工作。

## 5. 当前可用 Atrest 文件数量

当前可用的 Atrest 文件数为：

```text
Dbar Atrest: 97 files
Pbar Atrest: 10 files
```

因此，如果使用：

```bash
--max_files_per_class 20
```

实际结果不是 `Dbar 20 + Pbar 20`，而是：

```text
Dbar 20 files
Pbar 10 files
```

因为 Pbar Atrest 总共只有 10 个文件。

如果希望文件数平衡，建议使用：

```bash
--max_files_per_class 10
```

这样实际使用：

```text
Dbar 10 files
Pbar 10 files
```

## 6. 当前 preprocess_40M.py 的正确点

当前脚本已经满足以下条件：

```text
1. TOF_START = 1620
2. TOF_END = 1631
3. N_COLS = 1631
4. 排除 VolID
5. 排除 Inflight
6. Pbar 限定使用 csvFiles_Mc 目录
7. 输出 npz key 为 voxels / tofs / labels
```

这些设置与当前 Atrest-only 策略一致。

## 7. 当前仍需注意的问题

### 7.1 run_preprocess_40M.sh 的参数和注释需要调整

当前如果脚本仍写：

```bash
--max_files_per_class 20
```

并注释为：

```text
Dbar 20 + Pbar 20
```

则不准确。

因为 Pbar Atrest 只有 10 个文件。

建议改为：

```bash
--max_files_per_class 10
```

并将注释改为：

```text
Dbar 10 + Pbar 10 Atrest files
```

### 7.2 小样本测试仍应先运行

修改后应先运行：

```bash
bash nakagami/bash/test_preprocess_40M.sh
```

预期结果：

```text
train_hybrid_nakagami40M.npz:
  labels 同时包含 0 和 1

val_hybrid_nakagami40M.npz:
  labels 同时包含 0 和 1

skipped = 0
```

如果 train 或 val 仍只包含单一类别，则需要进一步检查 split 逻辑。

## 8. 建议执行顺序

当前建议：

```text
1. 保持 preprocess_40M.py 的 Atrest-only 策略。
2. 运行 test_preprocess_40M.sh。
3. 确认 small npz 中 train/val 都同时包含 Dbar 和 Pbar。
4. 将 run_preprocess_40M.sh 中 max_files_per_class 改为 10。
5. 运行 run_preprocess_40M.sh 生成 Dbar 10 + Pbar 10 的平衡 Atrest 数据。
6. 再运行 train_hybrid.py。
```

## 9. 结论

当前 `preprocess_40M.py` 的方向是正确的：

```text
只使用经过确认的 1631列 Atrest CSV。
```

这比混用未经验证的 1632列 Inflight 更稳妥。

当前最重要的后续动作是：

```text
重新运行小样本测试，确认 train/val label 分布正常。
```

