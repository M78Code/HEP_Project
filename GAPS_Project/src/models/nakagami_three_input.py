"""Nakagami-style three-input CNN+DNN model.

Inputs:
  voxel       : Si(Li) voxel energy, [B, 1, 10, 12, 12]
  tof_paddle  : TOF paddle energy deposition, [B, 172]
  tof_feature : TreeRec-based TOF feature vector, [B, 11]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int = 512, mid: int = 64, dropout: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(channels, mid, kernel_size=1)

        self.bn2 = nn.BatchNorm3d(mid)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv3d(mid, mid, kernel_size=3, padding=1)

        self.bn3 = nn.BatchNorm3d(mid)
        self.conv3 = nn.Conv3d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.drop(self.relu(self.bn2(out))))
        out = self.conv3(self.relu(self.bn3(out)))
        return out + shortcut


class NakagamiThreeInputNet(nn.Module):
    """PyTorch translation of the Nakagami three-input architecture."""

    def __init__(self, dropout_res: float = 0.1, dropout_dense: float = 0.2):
        super().__init__()
        self.entry = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.res2 = nn.Sequential(
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
        )
        self.pool2 = nn.MaxPool3d(2)
        self.res3 = nn.Sequential(
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
            ResBlock(512, 64, dropout_res),
        )
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.tof_paddle_branch = nn.Sequential(
            nn.Linear(172, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.tof_feature_branch = nn.Sequential(
            nn.Linear(11, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.tof_merge = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        voxel: torch.Tensor,
        tof_paddle: torch.Tensor,
        tof_feature: torch.Tensor,
    ) -> torch.Tensor:
        x = self.entry(voxel)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        cnn_feat = self.cnn_head(x)

        paddle_feat = self.tof_paddle_branch(tof_paddle)
        feature_feat = self.tof_feature_branch(tof_feature)
        tof_feat = self.tof_merge(torch.cat([paddle_feat, feature_feat], dim=1))

        logits = self.classifier(torch.cat([cnn_feat, tof_feat], dim=1))
        return logits.squeeze(1)
