"""
PyTorch translation of ynakagami2/DeepLearning/IdentifywithNN.py three-input model.

Inputs:
  input1: Si(Li) voxel, [B, 1, 10, 12, 12]
  input2: TOF paddle energy deposition, [B, 172]
  input3: primary TOF physical quantities, [B, 11]
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
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


class NakagamiThreeInputNet(nn.Module):
    def __init__(self, dropout_res=0.1, dropout_dense=0.2):
        super().__init__()
        self.entry = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res))
        self.pool1 = nn.MaxPool3d(2)
        self.res2 = nn.Sequential(ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res))
        self.pool2 = nn.MaxPool3d(2)
        self.res3 = nn.Sequential(ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res), ResBlock(512, 64, dropout_res))
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # input2 in IdentifywithNN.py: TOF paddle energy deposition, dim=172
        self.tof_paddle_branch = nn.Sequential(
            nn.Linear(172, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        # input3 in IdentifywithNN.py: primary TOF physical quantities, dim=11
        self.tof_primary_branch = nn.Sequential(
            nn.Linear(11, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        self.tof_merge = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(320, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, voxel, tof_paddle, tof_primary):
        x = self.entry(voxel)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        cnn_feat = self.cnn_head(x)

        paddle_feat = self.tof_paddle_branch(tof_paddle)
        primary_feat = self.tof_primary_branch(tof_primary)
        tof_feat = self.tof_merge(torch.cat([paddle_feat, primary_feat], dim=1))

        logits = self.classifier(torch.cat([cnn_feat, tof_feat], dim=1))
        return logits.squeeze(1)


if __name__ == '__main__':
    model = NakagamiThreeInputNet()
    voxel = torch.randn(4, 1, 10, 12, 12)
    tof_paddle = torch.randn(4, 172)
    tof_primary = torch.randn(4, 11)
    out = model(voxel, tof_paddle, tof_primary)
    print(out.shape)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
