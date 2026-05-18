import torch
import torch.nn as nn

class DNNBaseline(nn.Module):
    """
    纯全连接DNN，仅使用graph_feat（46维全局特征），不使用任何图结构。
    用于对比：图结构学习（GravNet）相比直接使用全局特征的DNN带来多少提升。

    输入: graph_feat [B, 46]
    输出: logits     [B, 2]
    """
    def __init__(self, graph_feat_dim: int = 46, hidden_dim: int = 128, dropout: float = 0.3, num_classes: int=2):
        super(DNNBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x=None, edge_index=None, batch=None, graph_feat=None):
        # x/edge_index/batch占位，保持与GravNet调用接口一致
        return self.net(graph_feat)