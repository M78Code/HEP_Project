# Step 3.3: DGCNN实现（Dynamic Graph CNN）

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, global_max_pool


class DGCNNClassifier(torch.nn.Module):
    """
    DGCNN（Dynamic Graph CNN）分类器
    来源：Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2018
    HEP应用参考：Ju et al., "GNNs for Particle Reconstruction in HEP detectors", 2020

    核心：EdgeConv + 动态图（每层在新特征空间重建k-NN图）
    边特征：h([x_i, x_j - x_i])，捕捉节点自身特征 + 与邻居的相对关系

    架构：
        输入 x [N, 5]
        → DynamicEdgeConv × 3层（每层动态重建图）
        → 每层输出concat → Linear
        → global_mean_pool + global_max_pool（拼接）
        → MLP → [batch, 2]

    Args：
        in_channels: 节点特征维度（默认5）
        hidden_dim : 隐层维度
        k          : 每层动态k-NN的邻居数
        num_classes: 分类数
        dropout    : dropout率
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64, k: int = 8, num_classes: int = 2,
                 dropout: float = 0.3):
        super(DGCNNClassifier, self).__init__()
        self.k = k

        # ── EdgeConv MLP工厂 ───────────────────────────────
        def edge_mlp(in_dim, out_dim):
            # EdgeConv输入是 [x_i, x_j - x_i]，所以实际输入维度 × 2
            return nn.Sequential(
                nn.Linear(in_dim * 2, out_dim * 2),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )

        # ── 3层 DynamicEdgeConv ────────────────────────────
        self.conv1 = DynamicEdgeConv(edge_mlp(in_channels, hidden_dim), k=k)
        self.conv2 = DynamicEdgeConv(edge_mlp(hidden_dim, hidden_dim), k=k)
        self.conv3 = DynamicEdgeConv(edge_mlp(hidden_dim, hidden_dim), k=k)

        # ── 各层输出concat后的压缩层 ───────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # ── 分类头（mean + max 拼接）──────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x           : 节点特征  [N_total, in_channels]
            edge_index  : 占位，DynamicEdgeConv内部动态建图，不使用 [N_total, 2]
        Returns:
            logits      : [batch_size, num_classes]
        """
        x1 = self.conv1(x, batch=batch)     # 第1层，在原始特征空间建图
        x2 = self.conv2(x1, batch=batch)    # 第2层，在x1特征空间建图
        x3 = self.conv3(x2, batch=batch)    # 第3层，在x2特征空间建图

        # concat 三层输出 → 压缩
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [N, hidden*3]
        x_fused = self.fusion(x_cat)                    # [N, hidden]

        # 节点级 → 图级（mean + max 拼接，比单用mean信息更丰富）
        x_mean = global_mean_pool(x_fused, batch) # [batch, hidden]
        x_max = global_max_pool(x_fused, batch)  # [batch, hidden]



