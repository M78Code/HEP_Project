

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

No_ANTIPROTON = -2212
No_ANTIDEUTERON = -1000010020

class GraphBuilder:

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
        n_hits = float(N)
        total_energy = float(energies.sum())

        # Bethe-Bloch: dE/dx ∝ 1/β²，antiD β更低 → dE/dx更大，是关键判别量
        if N > 1:
            tree = cKDTree(positions)
            k_query = min(self.k + 1, N)    # 防止N < k+1
            dists, _ = tree.query(positions, k=k_query)
            mean_dists = dists[:, 1:].mean(axis=1)  # 去掉自身（距离=0）
            dEdx = (energies / (mean_dists + 1e-6)).astype(np.float32)
        else:
            dEdx = np.zeros(N, dtype=np.float32)

        layer_idx = (volume_ids // 1000000).astype(np.int64)  # e.g. 200,201,...
        det_type = np.where(layer_idx >= 200, 1.0, 0.0).astype(np.float32)
        layer_norm = (layer_idx % 100).astype(np.float32) / 16.0

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

        pos_tensor = torch.tensor(positions, dtype=torch.float32)
        edge_index = knn_graph(pos_tensor, k=self.k, loop=False)

        y = torch.tensor([1 if label == No_ANTIDEUTERON else 0], dtype=torch.long)

        sili_profile, tof_profile = self._layer_profile(energies, volume_ids)

        tof_features = self._tof_features(energies, volume_ids, positions,
                                          np.where(np.isnan(event['times']), np.nan, event['times']))
        tof_features = np.nan_to_num(tof_features, nan=0.0, posinf=0.0, neginf=0.0)

        """
        PyG标准图对象：
            包含：
            X（节点特征）
            edge_index（边连接关系）
            pos（原始空间坐标），方便可视化，距离计算，physics analysis
            y（标签）
            num_nodes（节点数量）
        """
        mc_beta = float(event.get('beta', 0.0))

        return Data(
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
            mc_beta=torch.tensor([mc_beta], dtype=torch.float32),     # 仅元数据
        )

    # TOF layer分组（基于volume_id空间分布分析）
    # 官方volume_id規則: digit2=0→outer, digit2=1→inner
    OUTER_TOF_LAYERS = {100, 101, 102, 103, 104, 105, 106}  # face 0-6
    INNER_TOF_LAYERS = {110, 111, 112, 113, 114, 115}        # CUBE 6面

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

        sili = np.zeros(n_layers, dtype=np.float32)
        tof = np.zeros(n_layers, dtype=np.float32)
        for e, vid in zip(energy, volume_id):
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





