import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Pre-Activation Bottleneck ResBlock.
    TF原版顺序：BN→ReLU→Conv(1x1)→BN→ReLU→Dropout(0.1)→Conv(3x3)→BN→ReLU→Conv(1x1)→Add
    """
    def __init__(self, channels=512, mid=64, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channels, momentum=0.01)
        self.conv1 = nn.Conv3d(channels, mid, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(mid, momentum=0.01)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv3d(mid, mid, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(mid, momentum=0.01)
        self.conv3 = nn.Conv3d(mid, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.drop(self.relu(self.bn2(out)))
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        return out + shortcut


class NakagamiNet(nn.Module):

    """
    Faithful PyTorch translation of Nakagami's TF CNN+DNN hybrid (A.1, 6.2節).

    input1 (Si wafer) : (B, 1, 10, 12, 12)   [TF: (B, 10, 12, 12, 1)]
    input2 (TOF paddle): (B, 9)
    output             : (B,)  sigmoid score
    """

    def __init__(self):
        super().__init__()

        # --- CNN branch ---
        self.entry = nn.Sequential(
            nn.BatchNorm3d(1, momentum=0.01),
            nn.ReLU(),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(ResBlock(), ResBlock(), ResBlock())
        self.pool1 = nn.MaxPool3d(2)
        self.res2 = nn.Sequential(ResBlock(), ResBlock(), ResBlock())
        self.pool2 = nn.MaxPool3d(2)
        self.res3 = nn.Sequential(ResBlock(), ResBlock(), ResBlock())
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
        )

        # --- DNN branch ---
        self.dnn = nn.Sequential(
            nn.Linear(9, 256), nn.BatchNorm1d(256, momentum=0.01), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256, momentum=0.01), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128, momentum=0.01), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64, momentum=0.01), nn.ReLU(),
        )

        # --- Combined head (256+64=320) ---
        self.final = nn.Sequential(
            nn.Linear(320, 128), nn.BatchNorm1d(128, momentum=0.01), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128, momentum=0.01), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64, momentum=0.01), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        x = self.entry(x1)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        x = self.cnn_head(self.gap(x))
        y = self.dnn(x2)
        return torch.sigmoid(self.final(torch.cat([x, y], dim=1))).squeeze(1)