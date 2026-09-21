import torch
import torch.nn as nn


class SharedWaveformEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
        )

    def forward(self, waveform):
        return self.layers(waveform).flatten(1)


class HybridResidualNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = SharedWaveformEncoder()
        self.physics = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4 + 64 + 1, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, waveforms, features, baseline_position):
        left = self.encoder(waveforms[:, 0:1])
        right = self.encoder(waveforms[:, 1:2])
        waveform_features = torch.cat(
            [left, right, left - right, left * right], dim=1
        )
        physics_features = self.physics(features)
        output = self.head(
            torch.cat([waveform_features, physics_features, baseline_position], dim=1)
        )
        correction = output[:, :1]
        log_variance = output[:, 1:2].clamp(-6.0, 4.0)
        return baseline_position + correction, log_variance
