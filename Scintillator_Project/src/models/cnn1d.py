import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    轻量版 1D CNN，用于波形位置回归
    输入：[batch, 2, 1024]     (CH0 + CH1, 1024采样点)
    输出：[batch, 1]           (预测位置，单位cm)
    """
    def __init__(self):
        super().__init__()

        # 卷积块：Conv1d + BatchNorm + ReLU + MaxPool
        self.conv_block = nn.Sequential(
            # Block 1: [batch, 2, 1024] —> [batch, 32, 512]
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2: [batch, 32, 512] —> [batch, 64, 256]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3: [batch, 64, 256] —> [batch, 128, 128]
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 压缩时间轴：[batch, 128, 128] —> [batch, 128, 1]
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层：[batch, 128] —> [batch, 1]
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)  # [batch, 128, 128]
        x = self.pool(x)        # [batch, 128, 1]
        x = x.squeeze(-1)       # [batch, 128]
        x = self.fc(x)          # [batch, 1]
        return x


if __name__ == "__main__":
    model = CNN1D()
    print(model)
    print()

    # 验证输入输出形状
    dummy = torch.randn(64, 2, 1024)
    out = model(dummy)
    print(f'输入 shape: {dummy.shape}')
    print(f'输出 shape: {out.shape}')     # 期望 [64, 1]

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量：{total:,}')


"""
CNN1D(
  (conv_block): Sequential(
    (0): Conv1d(2, 32, kernel_size=(7,), stride=(1,), padding=(3,))
    (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv1d(32, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv1d(64, 128, kernel_size=(3,), stride=(1,), padding=(1,))
    (9): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (pool): AdaptiveAvgPool1d(output_size=1)
  (fc): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=1, bias=True)
  )
)

输入 shape: torch.Size([64, 2, 1024])
输出 shape: torch.Size([64, 1])
总参数量：44,257

"""