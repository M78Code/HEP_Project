# Step 2.1: hit->图构造

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

"""
  节点特征（8维）：[fX, fY, fZ, energy, time, dE/dx, det_type, layer_norm]
    det_type  : 0=TOF(1XX), 1=Si(Li)(2XX)
    layer_norm: (volume_id // 1000000) % 100 / 16.0
  图级属性：beta, n_hits, total_energy
  注：beta是事件级标量，同一event内所有节点相同，已移至graph_feat
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
    节点特征（8维）：[fX, fY, fZ, energy, time, dE/dx, det_type, layer_norm]
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
            beta_col,  # beta（事件级标量，恢复训练需保持9维）
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

        mc_energy = event.get('mc_energy', np.array([], dtype=np.float32))
        mc_volume_id = event.get('mc_volume_id', np.array([], dtype=np.int64))
        sili_profile, tof_profile = self._layer_profile(mc_energy, mc_volume_id)

        # TOF特征（从hitseries推导，6维）
        tof_features = self._tof_features(energies, volume_ids, positions,
                                          np.where(np.isnan(event['times']), np.nan, event['times']))
        tof_features = np.nan_to_num(tof_features, nan=0.0, posinf=0.0, neginf=0.0)

        # MC stopping特征（从pickle新字段读取，6维）
        stopping_pos = event.get('stopping_pos', np.zeros(3, dtype=np.float32))
        stopping_vol = int(event.get('stopping_vol', 0))
        stopping_ke  = float(event.get('stopping_ke', 0.0))
        stopping_li  = stopping_vol // 1000000
        stopping_det = 1.0 if stopping_li >= 200 else 0.0
        stopping_layer_norm = float(stopping_li % 100) / 16.0
        stopping_feat = np.array([
            stopping_pos[0] / 1000.0,  # 坐标cm → ~1
            stopping_pos[1] / 1000.0,
            stopping_pos[2] / 1000.0,
            stopping_layer_norm,        # 已归一化 0~1
            stopping_det,               # 0 or 1
            stopping_ke / 1000.0,       # 能量MeV → ~1
        ], dtype=np.float32)  # (6,)
        stopping_feat = np.nan_to_num(stopping_feat, nan=0.0, posinf=0.0, neginf=0.0)

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
            sili_profile=torch.tensor(sili_profile, dtype=torch.float32),    # (16,)
            tof_profile=torch.tensor(tof_profile, dtype=torch.float32),     # (16,)
            tof_feat=torch.tensor(tof_features, dtype=torch.float32),       # (6,)
            stopping_feat=torch.tensor(stopping_feat, dtype=torch.float32), # (6,)
        )

    @staticmethod
    def _tof_features(energies: np.ndarray, volume_ids: np.ndarray,
                      positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """从hitseries计算TOF相关特征，返回6维向量"""
        layer_idx = (volume_ids // 1000000).astype(np.int64)
        is_tof = layer_idx < 200

        tof_energies  = energies[is_tof]
        tof_positions = positions[is_tof]
        tof_times     = times[is_tof]

        n_tof       = float(is_tof.sum())
        tof_total_e = float(tof_energies.sum()) if n_tof > 0 else 0.0

        valid_mask  = ~np.isnan(tof_times)
        valid_times = tof_times[valid_mask]
        valid_pos   = tof_positions[valid_mask]

        if len(valid_times) > 0:
            first_idx      = int(np.argmin(valid_times))
            entry_x        = float(valid_pos[first_idx, 0])
            entry_y        = float(valid_pos[first_idx, 1])
            entry_z        = float(valid_pos[first_idx, 2])
            tof_time_range = float(valid_times.max() - valid_times.min())
        else:
            entry_x = entry_y = entry_z = tof_time_range = 0.0

        return np.array([
            n_tof / 20.0,           # hit数，典型10~30，归一化到~1
            tof_total_e / 100.0,    # 能量MeV，归一化
            entry_x / 1000.0,       # 坐标cm → ~1
            entry_y / 1000.0,
            entry_z / 1000.0,
            tof_time_range / 50.0,  # 时间ns，归一化
        ], dtype=np.float32)  # (6,)

    @staticmethod
    def _layer_profile(mc_energy: np.ndarray, mc_volume_id: np.ndarray,
                       n_layers: int = 16):
        """
        按探测器层号聚合MC能量沉积，返回 Si(Li) 和 TOF 各16维能量剖面。
        layer_idx = volume_id // 1000000
        Si(Li): layer_idx >= 200,  layer = layer_idx % 100
        TOF   : layer_idx <  200,  layer = layer_idx % 100
        """
        sili = np.zeros(n_layers, dtype=np.float32)
        tof = np.zeros(n_layers, dtype=np.float32)
        for e, vid in zip(mc_energy, mc_volume_id):
            li = int(vid) // 1000000
            ln = li % 100
            if 0 <= ln < n_layers:
                if li >= 200:
                    sili[ln] += e
                else:
                    tof[ln] += e
        return sili, tof

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


