# Step 3.2: GravNet实现

"""
从论文09可知，GravNet最关键的特点：
不用预先建好的图——GravNet 内部自己学一个 S 维坐标空间，在里面动态建 k-NN 图，边权重用距离的高斯衰减 exp(-d²) 计算。这就是它比普通
GNN 更适合不规则探测器几何的原因。
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool


class GravNetClassifier(nn.Module):
    """
    GravNet分类器，专为不规则稀疏探测器几何设计
    来源：Qasim et al., "Learning representations of irregular
            particle-detector geometry with distance-weighted
            graph networks", 2019

    架构（4个GravNet block）：
        输入 x [N, 5]
         → Block × 4
            Linear → tanh
            GravNetConv（内部动态建图，距离加权聚合）→ 保存每block输出
         → concat所有block输出 + skip
         → global_mean_pool
         → MLP → [batch, 2]

    Args:
         in_channels            : 节点特征维度（默认5）
         hidden_dim             : 隐层维度
         space_dimensions       : GravNet内部学习的坐标空间维度S（论文=4）
         propagate_dimensions   : 沿边传播的特征维度F_LR（论文=22）
         k                      : GravNet内部k近邻数（论文=40， 我们图小用8）
         num_classes            : 分类数
         dropout                : dropout率
    """

    def __init__(self, in_channels: int = 8, hidden_dim: int = 64, space_dimensions: int = 4,
                 propagate_dimensions: int = 22, k: int = 8, num_classes: int = 2, dropout: float = 0.3, graph_feat_dim: int = 2, num_blocks: int = 4):
        super(GravNetClassifier, self).__init__()
        self.num_blocks = num_blocks

        # ── 4个GravNet block ──────────────────────────────
        self.pre_linears = nn.ModuleList()
        self.gravnet_layers = nn.ModuleList()
        self.post_norms = nn.ModuleList()

        current_dim = in_channels
        for _ in range(self.num_blocks):
            self.pre_linears.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.Tanh()
                )
            )
            self.gravnet_layers.append(
                GravNetConv(in_channels=hidden_dim, out_channels=hidden_dim, space_dimensions=space_dimensions,
                            propagate_dimensions=propagate_dimensions, k=k,
                            ),
            )
            self.post_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        # ── 分类头 ─────────────────────────────────────────
        # concat 4个block输出 + 原始特征的线性映射
        self.skip_linear = nn.Linear(in_channels, hidden_dim)
        concat_dim = hidden_dim * self.num_blocks + hidden_dim + graph_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, graph_feat=None):
        """
        Args:
            x           : 节点特征 [N_total, in_channels]
            edge_index  : 占位，GravNet内部自己建图，不使用
            batch       : batch向量 [N_total]
        Returns:
            logits      : [batch_size, num_classes]
        """
        x_skip = self.skip_linear(x)  # 原始特征映射
        block_outputs = []

        x_cur = x
        for pre_linear, gravnet, norm in zip(self.pre_linears, self.gravnet_layers, self.post_norms):
            x_cur = pre_linear(x_cur)
            x_cur = gravnet(x_cur, batch=batch)
            x_cur = norm(x_cur).relu()
            block_outputs.append(x_cur)

        # 拼接所有block输出 + skip
        x_cat = torch.cat(block_outputs + [x_skip], dim=1)

        # 节点级 → 图级
        x_graph = global_mean_pool(x_cat, batch)
        if graph_feat is not None:
            x_graph = torch.cat([x_graph, graph_feat], dim=1)
        return self.classifier(x_graph)


if __name__ == '__main__':
    n_graphs = 4
    nodes_per_graph = 12
    in_channels = 8
    graph_feat_dim = 3

    x = torch.randn(n_graphs * nodes_per_graph, in_channels)
    batch = torch.arange(n_graphs).repeat_interleave(nodes_per_graph)
    graph_feat = torch.randn(n_graphs, graph_feat_dim)

    model = GravNetClassifier(
        in_channels=in_channels,
        hidden_dim=32,
        graph_feat_dim=graph_feat_dim,
        num_blocks=2,
    )
    logits = model(x, edge_index=None, batch=batch, graph_feat=graph_feat)

    print(f"input x.shape: {x.shape}")
    print(f"output logits.shape: {logits.shape}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
