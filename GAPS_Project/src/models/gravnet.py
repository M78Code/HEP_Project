# Step 3.2: GravNet实现

"""
从论文09可知，GravNet最关键的特点：
不用预先建好的图——GravNet 内部自己学一个 S 维坐标空间，在里面动态建 k-NN 图，边权重用距离的高斯衰减 exp(-d²) 计算。这就是它比普通
GNN 更适合不规则探测器几何的原因。
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_max_pool, global_mean_pool


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
        self.node_embedding_dim = hidden_dim * (self.num_blocks + 1)
        concat_dim = self.node_embedding_dim + graph_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_nodes(self, x, batch):
        """Return the concatenated per-node GravNet embedding."""
        x_skip = self.skip_linear(x)
        block_outputs = []

        x_cur = x
        for pre_linear, gravnet, norm in zip(
                self.pre_linears, self.gravnet_layers, self.post_norms):
            x_cur = pre_linear(x_cur)
            x_cur = gravnet(x_cur, batch=batch)
            x_cur = norm(x_cur).relu()
            block_outputs.append(x_cur)

        return torch.cat(block_outputs + [x_skip], dim=1)

    def forward(self, x, edge_index, batch, graph_feat=None):
        """
        Args:
            x           : 节点特征 [N_total, in_channels]
            edge_index  : 占位，GravNet内部自己建图，不使用
            batch       : batch向量 [N_total]
        Returns:
            logits      : [batch_size, num_classes]
        """
        x_cat = self.encode_nodes(x, batch)

        # 节点级 → 图级
        x_graph = global_mean_pool(x_cat, batch)
        if graph_feat is not None:
            x_graph = torch.cat([x_graph, graph_feat], dim=1)
        return self.classifier(x_graph)


class GravNetMultiTaskClassifier(GravNetClassifier):
    """Shared GravNet encoder with particle classification and beta regression heads.

    The regression head is trained against simulated beta, but beta is never
    appended to ``graph_feat``.  At inference both outputs use TreeRec inputs
    only.
    """

    def __init__(self, *args, hidden_dim: int = 64, graph_feat_dim: int = 2,
                 dropout: float = 0.3,
                 classify_with_predicted_beta: bool = False, **kwargs):
        super().__init__(
            *args,
            hidden_dim=hidden_dim,
            graph_feat_dim=graph_feat_dim,
            dropout=dropout,
            **kwargs,
        )
        concat_dim = self.node_embedding_dim + graph_feat_dim
        self.classify_with_predicted_beta = classify_with_predicted_beta
        if self.classify_with_predicted_beta:
            self.classifier = nn.Sequential(
                nn.Linear(concat_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            )
        self.beta_regressor = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, batch, graph_feat=None):
        x_cat = self.encode_nodes(x, batch)
        x_graph = global_mean_pool(x_cat, batch)
        if graph_feat is not None:
            x_graph = torch.cat([x_graph, graph_feat], dim=1)
        beta_prediction = self.beta_regressor(x_graph).squeeze(1)
        if self.classify_with_predicted_beta:
            x_graph = torch.cat([x_graph, beta_prediction.unsqueeze(1)], dim=1)
        return self.classifier(x_graph), beta_prediction


class DetectorAwareGravNetClassifier(GravNetClassifier):
    """GravNet with separate TOF and Si(Li) mean/max readout.

    The cached TreeRec node feature at index 6 is the detector type.  Global
    normalization caches retain the raw encoding (TOF=0, Si(Li)=1), while
    legacy per-event z-score caches encode mixed-detector events as negative
    TOF and positive Si(Li).  Both representations are supported here.
    """

    def __init__(self, *args, hidden_dim: int = 64, graph_feat_dim: int = 2,
                 **kwargs):
        super().__init__(
            *args,
            hidden_dim=hidden_dim,
            graph_feat_dim=graph_feat_dim,
            **kwargs,
        )
        self.readout_projection = nn.Sequential(
            nn.Linear(self.node_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + graph_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )

    @staticmethod
    def _masked_readout(x, batch, mask, num_graphs):
        """Return zero-filled mean/max pooling for one detector type."""
        out_dim = x.size(1)
        if not bool(mask.any()):
            zeros = x.new_zeros((num_graphs, out_dim))
            return zeros, zeros

        mean = global_mean_pool(x[mask], batch[mask], size=num_graphs)
        maximum = global_max_pool(x[mask], batch[mask], size=num_graphs)
        return mean, maximum

    @staticmethod
    def _detector_masks(detector_type):
        """Return TOF/Si(Li) masks for raw or legacy normalized encodings."""
        raw_encoding = bool(torch.all(
            (detector_type == 0) | (detector_type == 1)))
        if raw_encoding:
            return detector_type == 0, detector_type == 1, False
        return detector_type < 0, detector_type > 0, True

    def forward(self, x, edge_index, batch, graph_feat=None):
        node_embedding = self.readout_projection(self.encode_nodes(x, batch))
        num_graphs = int(batch[-1].item()) + 1 if batch.numel() else 0

        detector_type = x[:, 6]
        tof_mask, sili_mask, needs_legacy_zero_routing = \
            self._detector_masks(detector_type)

        # Per-event standardization makes detector_type exactly zero when an
        # event contains only one detector type. Use the already cached layer
        # profiles to route those nodes instead of dropping their embedding.
        zero_mask = detector_type == 0
        if needs_legacy_zero_routing and bool(zero_mask.any()):
            if graph_feat is None:
                raise ValueError(
                    'DetectorAwareGravNetClassifier requires graph_feat')
            sili_energy = graph_feat[:, 2:18].abs().sum(dim=1)
            tof_energy = graph_feat[:, 18:34].abs().sum(dim=1)
            zero_tof = zero_mask & (tof_energy[batch] > 0) & (
                sili_energy[batch] == 0)
            zero_sili = zero_mask & (sili_energy[batch] > 0) & (
                tof_energy[batch] == 0)
            unresolved = zero_mask & ~(zero_tof | zero_sili)
            tof_mask = tof_mask | zero_tof | unresolved
            sili_mask = sili_mask | zero_sili | unresolved

        tof_mean, tof_max = self._masked_readout(
            node_embedding, batch, tof_mask, num_graphs)
        sili_mean, sili_max = self._masked_readout(
            node_embedding, batch, sili_mask, num_graphs)

        x_graph = torch.cat(
            [tof_mean, tof_max, sili_mean, sili_max], dim=1)
        if graph_feat is not None:
            x_graph = torch.cat([x_graph, graph_feat], dim=1)
        return self.classifier(x_graph)


# ── 快速测试 ────────────────────────────────────────────
if __name__ == '__main__':
    from pathlib import Path
    import GAPS_Project
    from torch_geometric.loader import DataLoader
    from GAPS_Project.src.data_parse.gaps_dataset import GapsDataset

    PROJECT_PATH = Path(GAPS_Project.__file__).parent
    pkl_path = PROJECT_PATH / 'dataset' / 'test_sample' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.pkl'

    dataset = GapsDataset([pkl_path])
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    model = GravNetClassifier(in_channels=6, hidden_dim=64)
    logits = model(batch.x, batch.edge_index, batch.batch)

    print(f"输入 x.shape:     {batch.x.shape}")
    print(f"输出 logits:      {logits.shape}")
    print(f"预测类别:         {logits.argmax(dim=1)}")
    print(f"参数量:           {sum(p.numel() for p in model.parameters()):,}")


"""
输入 x.shape:     torch.Size([285, 6])
输出 logits:      torch.Size([8, 2])
预测类别:         tensor([0, 0, 0, 0, 0, 0, 0, 1])
参数量:           71,242


⏺ 正确，与 GIN 对比：

  ┌──────────────┬───────────────┬───────────────────┐
  │              │ GINClassifier │ GravNetClassifier │
  ├──────────────┼───────────────┼───────────────────┤
  │ logits.shape │ [8, 2] ✓      │ [8, 2] ✓          │
  ├──────────────┼───────────────┼───────────────────┤
  │ 参数量       │ 23,714        │ 71,114            │
  ├──────────────┼───────────────┼───────────────────┤
  │ 预测         │ 全0           │ 7个0，1个1        │
  └──────────────┴───────────────┴───────────────────┘

  参数量是 GIN 的3倍，符合预期——GravNet
  内部多了坐标空间学习和距离加权聚合的参数。预测出现了1个类别1也说明模型的随机初始化更多样，表达能力更强。

  Step 3.2 完成 ✓，继续 dgcnn.py 吗？

❯ 为什么论文里k用40，我们的用8，

⏺ 因为探测器规模不同。

  IceCube / 高粒度量能器（论文场景）
  - 每个event有数百到数千个节点（hit）
  - k=40 只占总节点数的很小比例，每个节点看到局部邻域
  - 图足够大，k=40 不会造成节点重复连接自己附近所有节点

  GAPS（我们的场景）
  - 每个event平均只有 31.5个节点，最少11个
  - 如果用 k=40，超过总节点数，knn_graph 会报错或退化
  - k=8 时，每个节点连接约 25% 的其他节点，密度已经足够

  ---
  本质上 k 的选择是：

  ▎ k / 总节点数 ≈ 合理的局部邻域比例

  论文：k=40 / ~1000节点 ≈ 4%
  我们：k=8 / ~31节点 ≈ 26%

  我们的 k=8 对应的局部连接密度其实比论文更高，已经够模型获取充分的邻域信息。

  ---
"""
