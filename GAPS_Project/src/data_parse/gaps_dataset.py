# Step 2.2: PyG数据集类

import uproot
import pickle
from pathlib import Path
from torch_geometric.data import Dataset, Data
import GAPS_Project
from GAPS_Project.src.data_parse.graph_builder import GraphBuilder

# 项目根目录
PROJECT_ROOT = Path(GAPS_Project.__file__).parent


class GapsDataset(Dataset):
    """
    从pickle文件加载所有event，转换为PyG Data对象列表

    Args:
        pkl_files   : list[Path|str]    pickle文件路径列表（支持多文件混合）
        k           : int               k近邻边数（传给GrapBuilder）
        normalize   : bool              是否归一化节点特征
        lazy        : bool              True=按需转换（省内存），False=全部预加载（快，服务器用）
    """

    def __init__(self, pkl_files: list, k: int = 8, normalize: bool = True, lazy: bool = True, use_beta: bool = True):
        super().__init__()
        self.builder = GraphBuilder(k=k, normalize=normalize, use_beta=use_beta)
        self.lazy = lazy
        self.k = k
        self.normalize = normalize
        self._load(pkl_files)

    def _load(self, pkl_files: list):
        """加载所有pkl文件中的events"""
        raw_events = []
        for path in pkl_files:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            raw_events.extend(payload['events'])

        if self.lazy:
            # 只存原始dict，get()时再转换
            self._raw_events = raw_events
        else:
            # 全部预转换为PyG Data对象
            self._data_list = [self.builder.build_from_dict(e) for e in raw_events]

    # def _load_all(self, pkl_files: list) -> list:
    #     data_list = []
    #     for path in pkl_files:
    #         with open(path, "rb") as f:
    #             payload = pickle.load(f)
    #         for event in payload["events"]:
    #             graph = self.builder.build_from_dict(event)
    #             data_list.append(graph)
    #     return data_list

    # ── PyG Dataset必须实现的三个方法 ──────────────────
    def len(self) -> int:
        if self.lazy:
            return len(self._raw_events)
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        if self.lazy:
            return self.builder.build_from_dict(self._raw_events[idx])
        return self._data_list[idx]

    # ── 便捷属性 ───────────────────────────────────────
    @property
    def num_node_features(self) -> int:
        """
        每个节点的特征维度，5
        [fX, fY, fZ, energy, time]
        """
        return self.get(0).x.shape[1]
        # return self._data_list[0].x.shape[1] if self._data_list else 0

    @property
    def num_classes(self) -> int:
        """
        二分类任务
        """
        return 2  # 0=antiproton（反质子）, 1=antideuteron（反重氘核）


# ── 快速测试 ────────────────────────────────────────────
if __name__ == "__main__":
    pkl_path = PROJECT_ROOT / 'dataset' / 'test_sample' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.pkl'

    dataset = GapsDataset([pkl_path], lazy=False)
    print(f'总event数：{len(dataset)}')
    print(f'节点特征维度：{dataset.num_node_features}')
    print(f'类别数：{dataset.num_classes}')

    g = dataset[0]
    print(f"\n第0个图:")
    print(f"  节点数:         {g.num_nodes}")
    print(f"  x.shape:        {g.x.shape}")
    print(f"  edge_index:     {g.edge_index.shape}")
    print(f"  标签 y:         {g.y.item()}")

"""
总event数：57
节点特征维度：6
类别数：2

第0个图:
  节点数:         30
  x.shape:        torch.Size([30, 6])
  edge_index:     torch.Size([2, 240])
  标签 y:         1
"""
