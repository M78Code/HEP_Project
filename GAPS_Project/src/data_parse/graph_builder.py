# Step 2.1: hit->图构造

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

"""
  节点特征（9维）：[fX, fY, fZ, energy, time, beta, dE/dx, det_type, layer_norm]
    det_type  : 0=TOF(1XX), 1=Si(Li)(2XX)
    layer_norm: (volume_id // 1000000) % 100 / 16.0
  图级属性：beta, n_hits, total_energy
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
      k         : int    k近邻边数，每个节点连接最近的k个节点
      normalize : bool   是否对节点特征归一化（默认True）
    节点特征（9维）：[fX, fY, fZ, energy, time, beta, dE/dx, det_type, layer_norm]
    图级属性：beta, n_hits, total_energy
    """
    def __init__(self, k: int = 8, normalize: bool = True):
        self.k = k
        self.normalize = normalize

    def build_from_dict(self, event: dict) -> Data:
        """
        从pickle中的event字典构建PyG图
        ::param event: {'energy': array(N,), 'positions': array(N,3), 'times': array(N,), 'volume_id': array(N,), 'label': int, 'beta': float, ...}
        """

        # ── 1. 读取数据 ──────────────────────────
        energies = event['energy']  # (N,)
        positions = event['positions']  # (N, 3)
        times = event['times']  # (N,)
        volume_ids = event.get('volume_id', np.zeros(len(energies), dtype=np.int64))
        label = event['label']
        N = len(energies)

        # ── 2. 处理NaN时间（填0）────────────────────
        times = np.where(np.isnan(times), 0.0, times)

        # ── 3. 图级标量 ────────────────────────────
        beta_val = float(event.get('beta', 0.0))
        n_hits = float(N)
        total_energy = float(energies.sum())

        # ── 4. 计算 dE/dx（每个hit能量 / k近邻平均距离）──
        # Bethe-Bloch: dE/dx ∝ 1/β²，antiD β更低 → dE/dx更大，是关键判别量
        if N > 1:
            tree = cKDTree(positions)
            k_query = min(self.k + 1, N)    # 防止N < k+1
            dists, _ = tree.query(positions, k=k_query)
            mean_dists = dists[:, 1:].mean(axis=1)  # 去掉自身（距离=0）
            dEdx = (energies / (mean_dists + 1e-6)).astype(np.float32)
        else:
            dEdx = np.zeros(N, dtype=np.float32)

        # ── 5. volume_id → det_type, layer_norm ──
        layer_idx = (volume_ids // 1000000).astype(np.int64)  # e.g. 200,201,...
        det_type = np.where(layer_idx >= 200, 1.0, 0.0).astype(np.float32)
        layer_norm = (layer_idx % 100).astype(np.float32) / 16.0

        # ── 6. 构建节点特征矩阵（9维）───────────────
        beta_col = np.full(N, beta_val, dtype=np.float32)
        x = np.stack([
            positions[:, 0],  # fX
            positions[:, 1],  # fY
            positions[:, 2],  # fZ
            energies,  # energy
            times,  # time
            beta_col,  # beta
            dEdx,  # dE/dx
            det_type,  # 探测器类型
            layer_norm,  # 层号归一化
        ], axis=1).astype(np.float32)

        if self.normalize:
            x = self._normalize(x)

        #  —— 7. 构建边（k近邻边，基于空间距离）—————————————
        pos_tensor = torch.tensor(positions, dtype=torch.float32)
        edge_index = knn_graph(pos_tensor, k=self.k, loop=False)

        # ── 8. 标签（PDG→0/1分类）────────────────────
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
            beta=torch.tensor([beta_val], dtype=torch.float32),
            n_hits=torch.tensor([n_hits], dtype=torch.float32),
            total_energy=torch.tensor([total_energy], dtype=torch.float32),
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


