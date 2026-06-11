"""
NakagamiNet: 中上修論 A.1 (6.2節) アーキテクチャの忠実な PyTorch 翻訳。

ネットワーク構造（修論 図A.1 / 表A.1):
  - Input1 (Si edep): [B, 1, 10, 12, 12]
    → BN → ReLU → Conv3D(1→512, k=3, pad=1)
    → ResBlock × 3 (preActive)
    → MaxPool3D(2)
    → ResBlock × 3
    → MaxPool3D(2)
    → ResBlock × 3
    → GlobalAvgPool3D → Linear(512→256) → BN → ReLU
  - Input2 (TOF 9次元): [B, 9]
    → Linear(9→256) → BN → ReLU
    → Linear(256→256) → BN → ReLU
    → Linear(256→128) → BN → ReLU
    → Linear(128→64) → BN → ReLU
  - Concat: [B, 256+64=320]
    → Linear(320→128) → BN → ReLU
    → Linear(128→128) → BN → ReLU
    → Linear(128→64) → BN → ReLU
    → Linear(64→1)

ResBlock (PreActivation Bottleneck):
  BN → ReLU → Conv(1×1, ch→64)
  → BN → ReLU → Dropout(0.1) → Conv(3×3×3, 64→64)
  → BN → ReLU → Conv(1×1, 64→ch)
  → Add (skip connection)

  ※ Dropout率は中上 IdentifywithNN.py の実装に合わせて 0.1 とした
  （論文 表A.1 は "DropOut率: 0" と書かれているがコード実装は 0.1）

注意: 出力はsigmoidなし（BCEWithLogitsLoss を想定）。
      推論時は外部で torch.sigmoid を適用すること。
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A.1 章で用いられる PreActivation Bottleneck ResBlock"""
    def __init__(self, channels=512, mid=64, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(channels, mid, kernel_size=1)

        self.bn2 = nn.BatchNorm3d(mid)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv3d(mid, mid, kernel_size=3, padding=1)

        self.bn3 = nn.BatchNorm3d(mid)
        self.conv3 = nn.Conv3d(mid, channels, kernel_size=1)

    def forward(self, x):
        shortcut = x
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.drop(self.relu(self.bn2(out))))
        out = self.conv3(self.relu(self.bn3(out)))
        return out + shortcut


class NakagamiNet(nn.Module):
    """
    中上 A.1 (6.2節) 構造の忠実な PyTorch 実装。

    Args:
        tof_dim : TOF入力次元 (4M dataset では 9)
        dropout : ResBlock内のDropout率 (中上コード IdentifywithNN.py に合わせて 0.1)
    """
    def __init__(self, tof_dim=9, dropout=0.1):
        super().__init__()

        # ── CNN branch ──
        self.entry = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),
        )
        # 3 ResBlock → MaxPool → 3 ResBlock → MaxPool → 3 ResBlock
        self.res1 = nn.Sequential(
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.res2 = nn.Sequential(
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
        )
        self.pool2 = nn.MaxPool3d(2)
        self.res3 = nn.Sequential(
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
            ResBlock(512, 64, dropout),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ── DNN branch (TOF 9次元入力) ──
        self.dnn = nn.Sequential(
            nn.Linear(tof_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),     nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128,  64),     nn.BatchNorm1d( 64), nn.ReLU(inplace=True),
        )

        # ── Combined head (CNN 256 + DNN 64 = 320) ──
        self.classifier = nn.Sequential(
            nn.Linear(320, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128,  64), nn.BatchNorm1d( 64), nn.ReLU(inplace=True),
            nn.Linear( 64,   1),
        )

    def forward(self, voxel, tof):
        # CNN
        x = self.entry(voxel)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        x = self.gap(x)
        cnn_feat = self.cnn_head(x)
        # DNN
        dnn_feat = self.dnn(tof)
        # Concat → classifier (logits)
        out = self.classifier(torch.cat([cnn_feat, dnn_feat], dim=1))
        return out.squeeze(1)


if __name__ == '__main__':
    model = NakagamiNet(tof_dim=9, dropout=0.1)
    voxel = torch.randn(4, 1, 10, 12, 12)
    tof = torch.randn(4, 9)
    out = model(voxel, tof)
    print(f'Output shape: {out.shape}')   # 期待: [4]
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
