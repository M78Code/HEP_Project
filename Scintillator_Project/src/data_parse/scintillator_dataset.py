import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ScintillatorDataset(Dataset):
    """
    从split后的JSON文件加载闪烁体波形数据

    输入格式：[2, 1024] (CH0 + CH1)
    标签：position (cm)
    """

    def __init__(self, json_path: str | Path, normalize: bool = True, roi: tuple | None = (450, 800), augment: bool = False):
        """
        :param json_path: train.json / val.json / test.json 的路径
        :param normalize: 是否对波形做归一化（除以通道最大绝对值）
        把train.json / val.json /test.json 里的事件数据，转换成模型可以直接训练用的NumPy数据格式
        """
        json_path = Path(json_path) # 把str路径转换成Path对象，方便后面打开文件
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = data["events"]

        """
        创建两个空列表
        self.waveforms: 保存波形数据
        self.labels: 保存标签，也就是位置position_label
        对应监督学习里的：
        X = waveforms
        y = labels
        """
        self.waveforms = []     # list of ndarray [2, 350]（ROI截取后）or [2, 1024]（roi=None时）
        self.labels = []        # 保存标签，list of float

        """
        遍历每一个event
        把CH0, CH1从Python转成NumPy数组
        dtype=np.float32是为了节省内存，适合深度学习模型训练，和PyTorch/TensorFlow常用格式一致
        
        """
        for ev in events:
            ch0 = np.array(ev['CH0'], dtype=np.float32) # (1024,) 每个通道形状是：(1024, )一个通道有1024个采样点
            ch1 = np.array(ev['CH1'], dtype=np.float32) # (1024,)
            """
            把两个通道合并成一个二维数组
            合并后的结构
            [
                [CH0的1024个点],
                [CH1的1024个点]
            ]
            """
            waveform = np.stack([ch0, ch1], axis=0) # (2, 1024) 把两个通道合并成一个二维数组

            # Step15: ROI固定窗口截取
            if roi is not None:
                waveform = waveform[:, roi[0]:roi[1]] # (2, 350)

            if normalize:
                # 两通道联合归一化：除以两通道共同的最大绝对值，保留CH0/CH1幅度比例（R₀信息）
                max_abs = np.max(np.abs(waveform))
                if max_abs > 1e-9:
                    waveform /= max_abs
            """
            把这一条 event 的双通道波形保存起来
            每个 waveform 形状是(2, 1024)
            """
            self.waveforms.append(waveform)
            self.labels.append(float(ev['position_label'])) # 把这个 event 的位置标签保存起来

        # --- 数据增强：对稀疏位置过采样（仅augment=True时生效）START ---
        if augment:
            SPARSE_THRESHOLD = 50
            TARGET_COUNT = 200
            NOISE_STD = 0.001

            labels_arr = np.array(self.labels)
            positions = np.unique(labels_arr)

            for pos in positions:
                idx = np.where(labels_arr == pos)[0]
                n = len(idx)
                if n < SPARSE_THRESHOLD:
                    n_to_add = TARGET_COUNT - n
                    for _ in range(n_to_add):
                        src = self.waveforms[np.random.choice(idx)]
                        noise = np.random.normal(0, NOISE_STD, src.shape).astype(np.float32)
                        self.waveforms.append(src + noise)
                        self.labels.append(pos)
        # --- 数据增强：对稀疏位置过采样（仅augment=True时生效）END ---

        """
        这个就是模型要预测的目标
        把所有 event 的波形列表转成一个大的 NumPy 数组
        如果有 N 条 event，最终形状是(N, 2, 1024)
        eg: val.json有3533条event，那么理论上是(3533, 2, 1024)
        把所有标签转成 NumPy 数组,形状是(N,)
        """
        self.waveforms = np.array(self.waveforms, dtype=np.float32) # (N, 2, 350) ROI模式 / (N, 2, 1024) 全段模式
        self.labels = np.array(self.labels, dtype=np.float32)       # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.waveforms[idx])   # (2, 350) ROI模式 / (2, 1024) 全段模式
        y = torch.tensor(self.labels[idx]) # scalar
        return x, y


if __name__ == "__main__":

    # import json
    # with open('../../dataset/split/val.json', 'r', encoding="utf-8") as f:
    #     data = json.load(f)
    # print(f'确认data类型，{type(data)}')
    # print(f'看有哪些key，{data.keys()}')

    split_dir = Path('../../dataset/split')
    train_dataset = ScintillatorDataset(split_dir / "train.json")
    val_dataset = ScintillatorDataset(split_dir / "val.json")
    test_dataset = ScintillatorDataset(split_dir / "test.json")

    print(f'Train: {len(train_dataset)} events')
    print(f'Val: {len(val_dataset)} events')
    print(f'Test: {len(test_dataset)} events')

    x, y = train_dataset[0]
    print(f'waveform shape: {x.shape}')
    print(f'label: {y.item()} cm')
    print(f'CH0 range: [{x[0].min():.4f}, {x[0].max():.4f}]')
    print(f'CH1 range: [{x[1].min():.4f}, {x[1].max():.4f}]')



"""
Train: 16514 events
Val: 3533 events
Test: 3555 events
waveform shape: torch.Size([2, 1024])
label: 15.0 cm
CH0 range: [-1.0000, 0.0085]
CH1 range: [-1.0000, 0.0264]



正常。

  - shape [2, 1024] 正确
  - 归一化后值域 [-1, 0.xx] 正常（脉冲是负方向的）
  - label 15.0 cm 正确

  下一步是Step 8：DataLoader + Step 9：模型。这两步通常写在一起。你来写还是我给出代码？

❯  归一化后值域 [-1, 0.xx] 正常（脉冲是负方向的）
    - label 15.0 cm 正确，这两个是什么意思，为什么是可以的呢

⏺ 1. 归一化后值域 [-1, 0.0085] 为什么正常？

  归一化方式是"除以通道最大绝对值"。

  假设CH0原始波形长这样：

  时间 →
  0.0  0.0  0.0  -0.85  -1.20  -0.90  0.01  0.0  ...
                   ↑
                脉冲（负方向）

  最大绝对值 = 1.20，除以它之后：

  0.0  0.0  0.0  -0.71  -1.00  -0.75  0.008  0.0  ...

  - 最小值变成 -1.0（脉冲峰值处）
  - 其余都在 (-1, 0) 之间
  - 正方向几乎是0（只有极小的噪声 0.0085）

  脉冲为什么是负方向？ 这是闪烁体探测器的电路特性，光电倍增管输出的是负极性脉冲，这在物理实验中很常见。

  ---
  2. label 15.0 cm 为什么正确？

  train.json 是所有15个位置混合后打乱的，第0个event恰好来自 run9_15.dat（位置=15cm），所以显示 15.0 cm。

  验证逻辑是：
  - label 字段存在 → split脚本正确写入了 position_label
  - 值是浮点数 15.0 而不是字符串 → float(ev['position_label']) 转换正确
  - Dataset能正常取出 → __getitem__ 没有bug

  也就是说，这两个输出合在一起说明：数据读进来了，格式对，归一化逻辑对，label也对，所以可以进入下一步。
"""



