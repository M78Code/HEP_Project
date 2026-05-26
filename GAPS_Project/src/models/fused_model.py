"""
GNN+GravNet 特征级融合模型
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool
from GAPS_Project.src.models.cnn_dnn_hybrid import PreActResBlock3D


class CNNEncoder(nn.Module):
    """复用 Nakagami A.2的CNN结构，输出128维特征（去掉分类头）"""

    def __init__(self, dropout=0.3):
        super(CNNEncoder, self).__init__()
        self.conv_init = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Conv3d(1, 512, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(*[PreActResBlock3D(512, 64, dropout) for _ in range(3)])
        self.pool = nn.MaxPool3d(2)
        self.res2 = nn.Sequential(*[PreActResBlock3D(512, 64, dropout) for _ in range(3)])
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, voxel):
        """voxel: [B, 1, 10, 20, 20] → [B, 128]"""
        x = self.conv_init(voxel)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


class FusedGravNet(nn.Module):
    """
    GravNet + CNN encoder 特征级融合
    graph_feat (45d) + cnn_feat (128d) = 173d → 拼接到 GravNet 输出后一起分类
    """

    def __init__(self, in_channels=8, hidden_dim=128, space_dimensions=4, propagate_dimensions=22, k=8, num_classes=2,
                 dropout=0.3, graph_feat_dim=45, num_blocks=6, cnn_feat_dim=128):
        super(FusedGravNet, self).__init__()
        self.num_blocks = num_blocks

        # ── GravNet blocks ──
        self.pre_linears = nn.ModuleList()
        self.gravnet_layers = nn.ModuleList()
        self.post_norms = nn.ModuleList()

        current_dim = in_channels
        for _ in range(num_blocks):
            self.pre_linears.append(nn.Sequential(nn.Linear(current_dim, hidden_dim), nn.Tanh()))
            self.gravnet_layers.append(
                GravNetConv(in_channels=hidden_dim, out_channels=hidden_dim,
                            space_dimensions=space_dimensions, propagate_dimensions=propagate_dimensions, k=k)
            )
            self.post_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        self.skip_linear = nn.Linear(in_channels, hidden_dim)

        # ── CNN encoder ──
        self.cnn_encoder = CNNEncoder(dropout=dropout)

        # ── 分类头：GravNet输出 + graph_feat + cnn_feat ──
        gravnet_out_dim = hidden_dim * num_blocks + hidden_dim
        concat_dim = gravnet_out_dim + graph_feat_dim + cnn_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, graph_feat=None, voxel=None):
        """
        Args:
            x          : 节点特征 [N_total, in_channels]
            edge_index : 边（GravNet内部自建图，占位）
            batch      : batch向量 [N_total]
            graph_feat : 图级特征 [B, 45]
            voxel      : Si(Li) voxel [B, 1, 10, 20, 20]
        """
        # GravNet 部分
        x_skip = self.skip_linear(x)
        block_outputs = []
        x_cur = x
        for pre_linear, gravnet, norm in zip(self.pre_linears, self.gravnet_layers, self.post_norms):
            x_cur = pre_linear(x_cur)
            x_cur = gravnet(x_cur, batch=batch)
            x_cur = norm(x_cur).relu()
            block_outputs.append(x_cur)

        x_cat = torch.cat(block_outputs + [x_skip], dim=1)
        x_graph = global_mean_pool(x_cat, batch) # [B, gravnet_out_dim]

        # CNN部分
        cnn_feat = self.cnn_encoder(voxel) # [B, 128]

        # 拼接全部
        parts = [x_graph]
        if graph_feat is not None:
            parts.append(graph_feat)
        parts.append(cnn_feat)

        return self.classifier(torch.cat(parts, dim=1))


