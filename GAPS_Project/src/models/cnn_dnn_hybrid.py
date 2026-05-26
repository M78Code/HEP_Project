import torch
import torch.nn as nn

"""
 形状追踪（A.2）：
  - Input voxel: [B, 1, 10, 12, 12]  (layers × x × y)
  - conv_init (Conv3d, padding=1): [B, 512, 10, 12, 12]
  - res1 × 3 (shape不变): [B, 512, 10, 12, 12]
  - pool (MaxPool3d(2)): [B, 512, 5, 10, 10]
  - res2 × 3: [B, 512, 5, 10, 10]
  - gap (AdaptiveAvgPool3d(1)): [B, 512, 1, 1, 1]
  - cnn_fc (Flatten→Linear(512→128)→BN→ReLU): [B, 128]
  - DNN: [B, 11] → [B, 64]
  - Concat: [B, 192] → classifier → [B, 1] → squeeze → [B]
"""
class PreActResBlock3D(nn.Module):
    """Pre-Activation Residual Block（BN→ReLU→Conv）"""
    def __init__(self, channels=512, mid=64, dropout=0.3):
        super(PreActResBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, mid, kernel_size=1),
            nn.BatchNorm3d(mid),
            nn.ReLU(),
            nn.Conv3d(mid, mid, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid),
            nn.ReLU(),
            nn.Conv3d(mid, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class CNNDNNHybrid(nn.Module):
    def __init__(self, tof_dim=11, dropout=0.3):
        super(CNNDNNHybrid, self).__init__()

        # CNN branch: [B, 1, 10, 12, 12]  (layers × x × y)
        self.conv_init = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),    # [B,512,10,12,12]
        )
        self.res1 = nn.Sequential(
            PreActResBlock3D(512, 64, dropout=dropout),
            PreActResBlock3D(512, 64, dropout=dropout),
            PreActResBlock3D(512, 64, dropout=dropout),
        )
        self.pool = nn.MaxPool3d(2)     # [B,512,5,10,10] ← (10→5, 20→10, 20→10)
        self.res2 = nn.Sequential(
            PreActResBlock3D(512, 64, dropout),
            PreActResBlock3D(512, 64, dropout),
            PreActResBlock3D(512, 64, dropout),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)  # [B,512,1,1,1]
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # DNN branch: [B, 11]
        self.dnn = nn.Sequential(
            nn.Linear(tof_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )

        # Merge: 128+64=192
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1),
        )


    def forward(self, voxel, tof_feat):
        c = self.conv_init(voxel)
        c = self.res1(c)
        c = self.pool(c)
        c = self.res2(c)
        c = self.gap(c)
        cnn_out = self.cnn_fc(c)
        dnn_out = self.dnn(tof_feat)
        return self.classifier(
            torch.cat([cnn_out, dnn_out], dim=1)
        ).squeeze(1)


if __name__ == '__main__':

    B = 4
    voxel = torch.randn(B, 1, 10, 12, 12)   # layers × x × y
    tof = torch.randn(B, 11)
    label = torch.randint(0, 2, (B,)).float()

    model = CNNDNNHybrid(tof_dim=11, dropout=0.3)
    model.eval()

    with torch.no_grad():
        out = model(voxel, tof)

    print(f"output shape: {out.shape}")  # 期望: torch.Size([4])
    print(f"output dtype: {out.dtype}")  # 期望: float32

    loss = nn.BCEWithLogitsLoss()(out, label)
    print(f"loss: {loss.item():.4f}")  # 期望: 正常值

    total = sum(p.numel() for p in model.parameters())
    print(f"total params: {total:,}")

    # ResBlock 形状不变检查
    block = PreActResBlock3D(512, 64, 0.3)
    block.eval()
    x = torch.randn(2, 512, 10, 10, 5)
    with torch.no_grad():
        y = block(x)
    print(f"ResBlock: {x.shape} -> {y.shape}")  # 形状应不变

"""
output shape: torch.Size([4])
output dtype: torch.float32
loss: 0.7344
total params: 1,200,003
ResBlock: torch.Size([2, 512, 10, 10, 5]) -> torch.Size([2, 512, 10, 10, 5])
"""
