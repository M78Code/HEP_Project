# Step 2.1: hit->图构造

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

"""
设计节点特征微量：
每个节点 = [fX, fY, fZ, energy, time], 共5维，time的NaN填0
边的构建方式：用k近邻（k-NN），基于空间距离连接最近的k个hit
"""

"""
| PDG         | 粒子               |
| ----------- | ---------------- |
| 2212        | 质子 proton        |
| -2212       | 反质子 antiproton   |
| 2112        | 中子 neutron       |
| -2112       | 反中子 antineutron  |
| 1000010020  | 氘核 deuteron      |
| -1000010020 | 反氘核 antideuteron |
| 1000010030  | 氚核 triton        |
| 1000020030  | 氦-3              |
| 1000020040  | α 粒子             |

"""

No_ANTIPROTON = -2212
No_ANTIDEUTERON = -1000010020

class GraphBuilder:
    """
    将单个event的hit信息转换成PyG的Data对象
    注：PyG（PyTorch Geometric），是基于PyTorch的图神经网络（GNN）库，专门用来处理Graph（图结构数据），而不是普通的Image（图像）、Sequence（序列）、Table（表格）

    Args:
        k         : int    k近邻边数，每个节点连接，最近的8个节点
        normalize : bool   是否对节点特征归一化（默认True）
    """
    def __init__(self, k: int = 8, normalize: bool = True):
        self.k = k
        self.normalize = normalize

    def build_from_dict(self, event: dict) -> Data:
        """
        从pickle中的event字典构建PyG图
        :param event: {'energy': array(N,), 'positions': array(N,3), 'times': array(N,), 'label': int, ...}
        """

        # ── 1. 读取数据 ──────────────────────────
        energies = event['energy']  # (N,)
        positions = event['positions']  # (N, 3)
        times = event['times']  # (N,)
        label = event['label']
        N = len(energies)

        # ── 2. 处理NaN时间（填0）────────────────────
        times = np.where(np.isnan(times), 0.0, times)

        # —— 3. 构建节点特征矩阵 [N, 5] ———————————————
        """
        特征：[fX, fY, fZ, energy, time]
        x.shape = [N, 5]，表示N个节点，每个节点5个特征，分别是[fX, fY, fZ, energy, time]（空间位置 + 能量 + 时间），这是GNN最核心输入
        """
        x = np.stack([
            positions[:, 0],    # fX
            positions[:, 1],    # fY
            positions[:, 2],    # fZ
            energies,           # energy
            times               # time
        ], axis=1).astype(np.float32)

        if self.normalize:
            x = self._normalize(x)

        #  —— 4. 构建边（k近邻边，基于空间距离）—————————————
        pos_tensor = torch.tensor(positions, dtype=torch.float32)
        edge_index = knn_graph(pos_tensor, k=self.k, loop=False)

        # ── 5. 标签（PDG→0/1分类）────────────────────
        # 反质子=-2212 → 0，反重氘核=-1000010020 → 1
        y = torch.tensor([1 if label == No_ANTIDEUTERON else 0], dtype=torch.long)

        """
        PyG标准图对象：
            包含：
            X（节点特征）
            edge_index（边连接关系）
            pos（原始空间坐标），方便可视化，距离计算，physics analysis
            y（标签）
            num_nodes（节点数量）
        """
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            pos=pos_tensor,
            y=y,
            num_nodes=N,
        )

    def _normalize(self, x):
        """各特征减均值除标准差，std=0时跳过"""
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return (x - mean) / std




# def build_graph(volume_ids, energies, positions, times, label, k=8):
#     """
#
#     :param volume_ids:  array (N,)     探测器体积ID，就是探测器编号，比如[101, 203, 305, 401]，这些hit分别来自不同的detector，
#     :param energies:    array (N,)     能量沉积（MeV），每个hit的能量沉积，比如[2.1, 5.3, 1.7, 4.8]，非常重要。因为反氘核和反质子的能量分布不同
#     :param positions:   array (N, 3)   hit位置（fX, fY, fZ），hit的三维位置，[[12.3, 5.1, -8.2], [13.0, 4.8, -7.9]]，这是构建graph最核心的信息
#     :param times:       array (N,)     hit时间（ns），含NaN，[12.5, NaN, 18.2]，NaN表示时间没有测到
#     :param label:       int            粒子标签（PDF编号），这是监督学习里的真值标签（Ground Truth）


