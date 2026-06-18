# Step 2.1: hit->图构造

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
try:
    from torch_cluster import knn_graph
except ImportError:
    from torch_geometric.nn import knn_graph

"""
  节点特征（8维）：[fX, fY, fZ, energy, time, dE/dx, det_type, layer_norm]
    det_type  : 0=TOF(1XX), 1=Si(Li)(2XX)
    layer_norm: (volume_id // 1000000) % 100 / 16.0
  图级特征（45维）：n_hits(1) + total_energy(1) + sili_profile(16) + tof_profile(16) + tof_feat(11)
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
    图级特征（45维）：n_hits(1) + total_energy(1) + sili_profile(16) + tof_profile(16) + tof_feat(11)
    """
    def __init__(self, k: int = 8, normalize: bool = True,
                 tof_paddle_index: dict[int, int] | None = None):
        self.k = k
        self.normalize = normalize
        self.tof_paddle_index = tof_paddle_index

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

        # ── 6. 构建节点特征矩阵（8维，纯Rec特征）───────────────
        x = np.stack([
            positions[:, 0],  # fX
            positions[:, 1],  # fY
            positions[:, 2],  # fZ
            energies,  # energy
            times,  # time
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

        # 使用Rec hitseries计算layer profile
        sili_profile, tof_profile = self._layer_profile(energies, volume_ids)

        # TOF特征（inner/outer分离，11维）
        tof_features = self._tof_features(energies, volume_ids, positions,
                                          np.where(np.isnan(event['times']), np.nan, event['times']))
        tof_features = np.nan_to_num(tof_features, nan=0.0, posinf=0.0, neginf=0.0)

        tof_paddle_energy = None
        if self.tof_paddle_index is not None:
            from GAPS_Project.src.data_parse.tof_paddles import build_tof_paddle_energy
            tof_paddle_energy = build_tof_paddle_energy(
                energies, volume_ids, self.tof_paddle_index)

        """
        PyG标准图对象：
            包含：
            X（节点特征）
            edge_index（边连接关系）
            pos（原始空间坐标），方便可视化，距离计算，physics analysis
            y（标签）
            num_nodes（节点数量）
        """
        # mc_beta仅用于评估时β窗口分析，不参与训练
        mc_beta = float(event.get('beta', 0.0))

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            pos=pos_tensor,
            y=y,
            num_nodes=N,
            n_hits=torch.tensor([n_hits], dtype=torch.float32),
            total_energy=torch.tensor([total_energy], dtype=torch.float32),
            sili_profile=torch.tensor(sili_profile, dtype=torch.float32),    # (16,)
            tof_profile=torch.tensor(tof_profile, dtype=torch.float32),     # (16,)
            tof_feat=torch.tensor(tof_features, dtype=torch.float32),       # (11,)
            mc_beta=torch.tensor([mc_beta], dtype=torch.float32),           # 仅元数据
        )
        if tof_paddle_energy is not None:
            data.tof_paddle_energy = torch.tensor(
                tof_paddle_energy, dtype=torch.float32)
        return data

    # TOF layer分组（基于volume_id空间分布分析）
    # 官方volume_id規則: digit2=0→outer, digit2=1→inner
    OUTER_TOF_LAYERS = {100, 101, 102, 103, 104, 105, 106}  # face 0-6
    INNER_TOF_LAYERS = {110, 111, 112, 113, 114, 115, 116}   # CUBE 6面 + corner paddles

    @staticmethod
    def _tof_features(energies: np.ndarray, volume_ids: np.ndarray,
                      positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        从hitseries计算inner/outer TOF特征，返回11维向量。
        参考先行研究(Nakagami 2021)的TOF特征构造。
        [0]  outer_energy      外层TOF总能量
        [1]  inner_energy      内层TOF总能量
        [2]  outer_n_hits      外层TOF hit数
        [3]  inner_n_hits      内层TOF hit数
        [4]  time_of_flight    飞行时间 = inner最早时间 - outer最早时间
        [5]  outer_entry_x     外层最早hit的x坐标
        [6]  outer_entry_y     外层最早hit的y坐标
        [7]  outer_entry_z     外层最早hit的z坐标
        [8]  inner_entry_x     内层最早hit的x坐标
        [9]  inner_entry_y     内层最早hit的y坐标
        [10] inner_entry_z     内层最早hit的z坐标
        """
        layer_idx = (volume_ids // 1000000).astype(np.int64)

        is_outer = np.isin(layer_idx, list(GraphBuilder.OUTER_TOF_LAYERS))
        is_inner = np.isin(layer_idx, list(GraphBuilder.INNER_TOF_LAYERS))

        # ── 能量和hit数 ──
        outer_energy = float(energies[is_outer].sum()) if is_outer.any() else 0.0
        inner_energy = float(energies[is_inner].sum()) if is_inner.any() else 0.0
        outer_n_hits = float(is_outer.sum())
        inner_n_hits = float(is_inner.sum())

        # ── outer最早hit ──
        outer_first_t = 0.0
        outer_entry = np.zeros(3, dtype=np.float32)
        if is_outer.any():
            o_times = times[is_outer]
            o_pos = positions[is_outer]
            o_valid = ~np.isnan(o_times)
            if o_valid.any():
                o_first = int(np.argmin(o_times[o_valid]))
                outer_first_t = float(o_times[o_valid][o_first])
                outer_entry = o_pos[o_valid][o_first]

        # ── inner最早hit ──
        inner_first_t = 0.0
        inner_entry = np.zeros(3, dtype=np.float32)
        if is_inner.any():
            i_times = times[is_inner]
            i_pos = positions[is_inner]
            i_valid = ~np.isnan(i_times)
            if i_valid.any():
                i_first = int(np.argmin(i_times[i_valid]))
                inner_first_t = float(i_times[i_valid][i_first])
                inner_entry = i_pos[i_valid][i_first]

        # ── 飞行时间（inner - outer） ──
        has_outer_t = is_outer.any() and (~np.isnan(times[is_outer])).any()
        has_inner_t = is_inner.any() and (~np.isnan(times[is_inner])).any()
        tof = (inner_first_t - outer_first_t) if (has_outer_t and has_inner_t) else 0.0

        return np.array([
            outer_energy / 100.0,       # 能量MeV，归一化
            inner_energy / 100.0,
            outer_n_hits / 20.0,        # hit数，归一化
            inner_n_hits / 20.0,
            tof / 50.0,                 # 飞行时间ns，归一化
            outer_entry[0] / 1000.0,    # 坐标mm → ~1
            outer_entry[1] / 1000.0,
            outer_entry[2] / 1000.0,
            inner_entry[0] / 1000.0,
            inner_entry[1] / 1000.0,
            inner_entry[2] / 1000.0,
        ], dtype=np.float32)  # (11,)

    @staticmethod
    def _layer_profile(energy: np.ndarray, volume_id: np.ndarray,
                       n_layers: int = 16):
        """
        按探测器层号聚合能量沉积，返回 Si(Li) 和 TOF 各16维能量剖面。
        layer_idx = volume_id // 1000000
        Si(Li): layer_idx >= 200,  layer = layer_idx % 100
        TOF   : layer_idx <  200,  layer = layer_idx % 100
        """
        sili = np.zeros(n_layers, dtype=np.float32)
        tof = np.zeros(n_layers, dtype=np.float32)
        for e, vid in zip(energy, volume_id):
            li = int(vid) // 1000000
            ln = li % 100
            if li >= 200:
                if 0 <= ln < n_layers:
                    sili[ln] += e
            elif 0 <= ln < n_layers:
                tof[ln] += e
            elif li == 116:
                # Corner paddles share the final bin of the 16-D TOF profile.
                tof[-1] += e
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
