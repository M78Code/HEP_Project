import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    """
    1D残差块：两层Conv + BN + ReLU, 加shortcut连接
    当输入输出通道不同时，shortcut用1x1 Conv对齐
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        # 通道数不同时，shortcut需要投影对齐
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """
    1D ResNet，用于波形位置回归
    输入：[batch, 2, 1024]
    输出：[batch, 1]
    总参数量：约 100K
    """
    def __init__(self):
        super().__init__()

        # 入口卷积：快速升维 + 下采样
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # [batch, 32, 512]
        )

        # 残差块堆叠
        self.layer1 = nn.Sequential(
            ResBlock1D(32, 32),
            ResBlock1D(32, 32),
            nn.MaxPool1d(kernel_size=2),        # [batch, 32, 256]
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(32, 64),
            ResBlock1D(64, 64),
            nn.MaxPool1d(kernel_size=2),        # [batch, 64, 128]
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(64, 128),
            ResBlock1D(128, 128),
            nn.MaxPool1d(kernel_size=2),        # [batch, 128, 64]
        )

        # 全局平均池化 + 回归头
        self.pool = nn.AdaptiveAvgPool1d(1)     # [batch, 128, 1]
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


if __name__ == '__main__':
    model = ResNet1D()
    dummy = torch.randn(64, 2, 1024)
    out = model(dummy)
    print(f'输入 shape: {dummy.shape}')
    print(f'输出 shape: {out.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量：{total:,}')


"""
输入 shape: torch.Size([64, 2, 1024])
输出 shape: torch.Size([64, 1])
总参数量：248,577
"""