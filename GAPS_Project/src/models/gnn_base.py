# Step 3.1: 基础GNN框架

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


class GINClassifier(nn.Module):
    """
    GIN（Graph Isomorphism Network）二分类器
    用于反质子 vs 反重氘核的粒子识别

    架构：
        输入　x [N, 5]
          → GINConv × 3层（逐步提取节点特征）
          → global_mean_pool （节点特征聚合为图级向量）
          → MLP → 输出 [batch, 2]

    Args:
        in_channels (int): 节点特征维度（默认5: fX、fY、fZ、energy、time）
        hidden_dim  (int): 隐层维度，GIN内部，把5维映射成64维
        num_classes (int): 分类数（默认2）
        dropout   (float): dropout率，防止过拟合，训练时随机丢弃30%神经元，这是标准做法
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.3):
        super(GINClassifier, self).__init__()

        """
        GINConv层，每层内部是一个MLP（多层感知机）
        MLP的本质上就是：输入 → 线性变换（Linear） → 激活函数（ReLU） → 再线性变换 → 输出
        """
        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        # ── 三层图卷积  ──────────────────────────────────
        self.conv1 = GINConv(mlp(in_channels, hidden_dim))
        self.conv2 = GINConv(mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(mlp(hidden_dim, hidden_dim))

        # ── 分类头 ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x           : 节点特征 [N_total, in_channels]
            edge_index  : 边索引 [2, N_total]
            batch       : batch向量 [N_total], 标记每个节点属于哪个图
        Returns:
            logits      : [batch_size, num_classes]
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # 节点级 → 图级
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x = self.classifier(x)          # [batch_size, 2]
        return x


# ── 快速测试 ────────────────────────────────────────────
if __name__ == '__main__':
    from pathlib import Path
    import GAPS_Project
    from torch_geometric.loader import DataLoader
    from GAPS_Project.src.data_parse.gaps_dataset import GapsDataset

    PROJECT_ROOT = Path(GAPS_Project.__file__).parent
    pkl_path = PROJECT_ROOT / 'dataset' / 'processed' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.pkl'

    dataset = GapsDataset([pkl_path])
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    model = GINClassifier(in_channels=5, hidden_dim=64)
    logits = model(batch.x, batch.edge_index, batch.batch)

    print(f"输入 x.shape:     {batch.x.shape}")
    print(f"输出 logits:      {logits.shape}")  # → [8, 2]
    print(f"预测类别:         {logits.argmax(dim=1)}")
    print(f"参数量:           {sum(p.numel() for p in model.parameters()):,}")


"""
输入 x.shape:     torch.Size([285, 5])    8个event合并，共285个节点，每节点5个特征
输出 logits:      torch.Size([8, 2])      8个event，每个输出2个类别分数
预测类别:         tensor([0, 0, 0, 0, 0, 0, 0, 0])  模型未训练，随机初始化偏向某一类，正常现象
参数量:           23,714   轻量级模型，适合小数据集
"""