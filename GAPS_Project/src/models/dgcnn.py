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
    """
    def __init__(self, in_channels, out_channels):
        super(DGCNNClassifier, self).__init__()