"""
新GAPS 5000万データ用 GravNet 訓練スクリプト。

【中文说明】
这个脚本对应日志 `train_aohba50M_atrest_gravnet_full.log`。
它不是从 root/pkl 原始文件现场构图训练，而是读取已经由 GraphBuilder
预先转换好的 PyTorch Geometric graph cache（.pt 文件）。

数据输入有两种形式：
1. `--manifest` + `--cache-dir`
   - `split_manifest.json` 记录 train/val 中使用哪些原始 pkl 文件。
   - `--cache-dir` 指向这些 pkl 已经转换出的 graph cache。
   - 代码通过 manifest 中的 pkl 路径推导对应的 `.pt` graph cache 路径。
2. `--split-cache-dir`
   - 直接读取 `train.pt` / `val.pt`，或 `train_*.pt` / `val_*.pt` shard。

每个 graph 使用的输入信息来自 `src/data_parse/graph_builder.py`：
  节点特征 8维:
    [x, y, z, energy, time, dE/dx, det_type, layer_norm]
  图级特征 45维:
    n_hits(1) + total_energy(1)
    + tracker/Si(Li) layer energy profile(16)
    + TOF layer energy profile(16)
    + TOF feature(11)

`mc_beta` 只作为评估时的 beta 分层 metadata，不参与训练。
如果 `--model gravnet_tof`，还会额外使用 `tof_paddle_energy` 172维。

训练中每个 epoch 都保存 latest checkpoint，包含 model / optimizer /
scheduler 状态，因此可以用 `--resume-checkpoint` 从中断位置继续训练。
注意：resume 时 `--epochs` 表示“目标总 epoch 数”，不是额外追加的 epoch 数。
例如 checkpoint 已到 epoch 80，想继续训练到 120，就写 `--epochs 120`。

当前 50M full 训练对应关系：
  训练日志:
    ~/train_aohba50M_atrest_gravnet_full.log
  训练输入:
    --manifest /mnt/ynakagami3/aohba_preprocess/split/split_manifest.json
    --cache-dir /mnt/ynakagami3/aohba_preprocess/graph_cache_v2
  训练结果目录:
    results/20260625-110900_GravNet_6b_h128_aohba50M_atrest_gravnet_full
  latest checkpoint:
    results/20260625-110900_GravNet_6b_h128_aohba50M_atrest_gravnet_full/
    20260625-110900_GravNet_6b_h128_aohba50M_atrest_gravnet_full_last_checkpoint.pth
  評価:
    best checkpoint を dataset/aohba4M_atrest_tof172_balanced の
    4M test split で評価し、50M training の効果を見る。

通常の 50M full は `--model gravnet` であり、TOF172 は使っていない。
TOF172 を追加する場合だけ `--model gravnet_tof` を指定する。

使い方:
  python src/scripts/train_aohba.py \
      --manifest /mnt/ynakagami3/aohba_preprocess/split/split_manifest.json
"""
import argparse
from contextlib import nullcontext
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.losses import FocalLoss
from GAPS_Project.src.models.gravnet import (
    ClusterVertexTokenClassifier,
    GravNetAttentionClassifier,
    GravNetClusterTokenClassifier,
    DetectorAwareGravNetClassifier,
    GravNetClassifier,
    GravNetPhysicsEdgeClassifier,
    GravNetMultiTaskClassifier,
    GravNetSoftObjectClassifier,
)
from GAPS_Project.src.models.gravnet_tof import GravNetTOFClassifier
from GAPS_Project.src.models.tree_rec_features import (
    HIT_TOPOLOGY_FEATURE_DIM,
    TRACK_STAR_FEATURE_DIM,
    append_hit_topology,
    append_track_star,
    build_base_graph_feat,
    cluster_vertex_token_inputs,
    fit_mc_beta_normalizer,
    fit_short_tof_antip_profile,
    load_graph_feature_normalizer,
    normalize_base_graph_feat,
    reconstruct_tof_beta,
)

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── ハイパーパラメータ ──────────────────────────────────
EPOCHS       = 80
BATCH_SIZE   = 128
LR           = 3e-4
STEP_SIZE    = 15
GAMMA        = 0.5
FOCAL_GAMMA  = 1.5
PATIENCE     = 10
MIN_EPOCHS   = 20
IN_CHANNEL   = 8
NUM_BLOCKS   = 6
HIDDEN_DIM   = 128
DEFAULT_BETA_BINS = '0.20,0.25,0.30,0.35,0.40,0.45,0.50'
DEFAULT_BETA_BIN_WEIGHTS = '1,1,1.5,2,4,6'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'使用设备：{DEVICE}')


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def beta_bin_weights(
        beta: torch.Tensor,
        bin_edges: torch.Tensor,
        bin_weights: torch.Tensor) -> torch.Tensor:
    """Return per-graph weights according to beta bins."""
    weights = torch.ones_like(beta, dtype=torch.float32)
    for i in range(len(bin_weights)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == len(bin_weights) - 1:
            mask = (beta >= lo) & (beta <= hi)
        else:
            mask = (beta >= lo) & (beta < hi)
        weights[mask] = bin_weights[i]
    return weights


def short_tof_antip_weights(
        batch, targets: torch.Tensor, weight: float,
        reference_ns: float, scale_ns: float) -> torch.Tensor:
    """Upweight short-flight-time antiP examples using TreeRec observables."""
    if not hasattr(batch, 'tof_feat'):
        raise ValueError('--short-tof-antip-weight requires tof_feat in graph cache')
    delta_t_ns = batch.tof_feat.view(-1, 11)[:, 4].float() * 50.0
    valid = torch.isfinite(delta_t_ns) & (delta_t_ns > 0.0)
    hardness = torch.sigmoid((reference_ns - delta_t_ns) / scale_ns)
    extra = 1.0 + (weight - 1.0) * hardness
    return torch.where((targets == 0) & valid, extra, torch.ones_like(extra))


def build_graph_feat(
        batch, use_mc_beta: bool = False,
        use_tof_beta: bool = False,
        use_hit_topology: bool = False,
        use_track_star: bool = False,
        graph_feature_mean: torch.Tensor | None = None,
        graph_feature_std: torch.Tensor | None = None) -> torch.Tensor:
    """Build event-level graph features with one optional beta source."""
    if use_mc_beta and use_tof_beta:
        raise ValueError('MC beta and TOF-reconstructed beta are mutually exclusive')
    graph_feat = build_base_graph_feat(batch)
    if (graph_feature_mean is None) != (graph_feature_std is None):
        raise ValueError('graph-feature mean/std must be provided together')
    if graph_feature_mean is not None:
        graph_feat = normalize_base_graph_feat(
            graph_feat, graph_feature_mean, graph_feature_std)
    if use_mc_beta:
        if not hasattr(batch, 'mc_beta'):
            raise ValueError('--use-mc-beta requires mc_beta in graph cache')
        graph_feat = torch.cat(
            [graph_feat, batch.mc_beta.view(-1, 1).float()],
            dim=1,
        )
    elif use_tof_beta:
        graph_feat = torch.cat([
            graph_feat,
            reconstruct_tof_beta(batch.tof_feat.view(-1, 11).float()),
        ], dim=1)
    if use_hit_topology:
        graph_feat = append_hit_topology(graph_feat, batch)
    if use_track_star:
        graph_feat = append_track_star(graph_feat, batch)
    return graph_feat


def forward_model(model, model_name, batch, graph_feat):
    """Call one model variant while keeping the training loops identical."""
    if model_name == 'gravnet_tof':
        if not hasattr(batch, 'tof_paddle_energy'):
            raise ValueError('GravNetTOF requires tof_paddle_energy in graph cache')
        return model(
            batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat,
            tof_paddle_energy=batch.tof_paddle_energy.view(-1, 172))
    if model_name in ('gravnet_cluster_tokens', 'cluster_tokens_only'):
        vertex_token, prong_tokens, prong_mask = cluster_vertex_token_inputs(batch)
        return model(
            batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat,
            vertex_token=vertex_token, prong_tokens=prong_tokens,
            prong_mask=prong_mask)
    if model_name == 'gravnet_physics_edges':
        if not hasattr(batch, 'edge_attr'):
            raise ValueError(
                'gravnet_physics_edges requires cached edge_attr; run '
                'attach_treerec_physics_edge_attributes.py first')
        return model(
            batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat,
            edge_attr=batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)


def soft_object_auxiliary_loss(auxiliary, batch):
    """Training-only MC supervision for the learned track and stop queries."""
    required = ('mc_soft_stop_z', 'mc_soft_direction', 'mc_soft_truth_valid')
    missing = [name for name in required if not hasattr(batch, name)]
    if missing:
        raise ValueError(
            'gravnet_soft_objects requires cached MC auxiliary targets; '
            f'missing {missing}')
    valid = batch.mc_soft_truth_valid.view(-1).bool()
    if not valid.any():
        raise ValueError('soft-object batch has no valid MC auxiliary targets')
    stop_target = batch.mc_soft_stop_z.view(-1, 3).float()
    direction_target = batch.mc_soft_direction.view(-1, 3).float()
    stop_loss = torch.nn.functional.smooth_l1_loss(
        auxiliary['stop_prediction'][valid], stop_target[valid])
    direction_prediction = torch.nn.functional.normalize(
        auxiliary['direction_prediction'][valid], dim=1, eps=1e-6)
    direction_target = torch.nn.functional.normalize(
        direction_target[valid], dim=1, eps=1e-6)
    direction_loss = 1.0 - (direction_prediction * direction_target).sum(dim=1).mean()
    return stop_loss + direction_loss, stop_loss, direction_loss


# ── IterableDataset ────────────────────────────────────
class CachedStreamDataset(IterableDataset):
    """
    キャッシュ済み .pt ファイル（PyG Data リスト）を1ファイルずつ読み込む
    IterableDataset。GraphBuilder の実行が不要で高速。

    中文说明:
    这里读入的 `.pt` 文件已经是 graph list。也就是说，训练时不会再读取
    root，也不会重新计算 hit->graph，只是逐个 shard 加载已经缓存好的
    PyG Data 对象来节省内存。
    """

    def __init__(self, pt_files: list, shuffle_files: bool = True,
                 shuffle_events: bool = True, seed: int = 42,
                 balance_tagged_classes: bool = False,
                 beta_min: float = None, beta_max: float = None):
        self.pt_files       = list(pt_files)
        self.shuffle_files  = shuffle_files
        self.shuffle_events = shuffle_events
        self.seed           = seed
        self.balance_tagged_classes = balance_tagged_classes
        self.beta_min       = beta_min
        self.beta_max       = beta_max
        self._epoch         = 0

    def _passes_beta(self, data) -> bool:
        if self.beta_min is None and self.beta_max is None:
            return True
        if not hasattr(data, 'mc_beta'):
            return False
        beta = float(data.mc_beta.view(-1)[0])
        if self.beta_min is not None and beta < self.beta_min:
            return False
        if self.beta_max is not None and beta >= self.beta_max:
            return False
        return True

    @staticmethod
    def _load_shard(pt_path: Path):
        try:
            data_list = torch.load(pt_path, weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f'failed to load graph-cache shard: {pt_path} '
                f'({type(error).__name__}: {error})'
            ) from error
        if not isinstance(data_list, list) or not data_list:
            raise RuntimeError(
                f'invalid graph-cache shard: {pt_path} '
                f'(expected a non-empty list, got {type(data_list).__name__})'
            )
        return data_list

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1

        if self.balance_tagged_classes:
            antiD_files = [
                path for path in self.pt_files if '_antiD_' in path.name]
            antiP_files = [
                path for path in self.pt_files if '_antiP_' in path.name]
            if antiD_files and antiP_files:
                yield from self._iter_balanced(
                    antiD_files, antiP_files, epoch)
                return

        files = self.pt_files.copy()
        if self.shuffle_files:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(files)

        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id::worker.num_workers]

        for file_index, pt_path in enumerate(files):
            data_list = self._load_shard(pt_path)
            if self.shuffle_events:
                random.Random(
                    self.seed + epoch * 1_000_003 + file_index
                ).shuffle(data_list)
            for data in data_list:
                if self._passes_beta(data):
                    yield data

    def _iter_balanced(self, antiD_files, antiP_files, epoch):
        """Interleave class-pure shards so every batch is class balanced."""
        antiD_files = antiD_files.copy()
        antiP_files = antiP_files.copy()
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(antiD_files)
            random.Random(self.seed + epoch + 10_000).shuffle(antiP_files)

        worker = get_worker_info()
        if worker is not None:
            antiD_files = antiD_files[worker.id::worker.num_workers]
            antiP_files = antiP_files[worker.id::worker.num_workers]

        def class_stream(files, class_offset):
            for file_index, pt_path in enumerate(files):
                data_list = self._load_shard(pt_path)
                if self.shuffle_events:
                    random.Random(
                        self.seed
                        + epoch * 1_000_003
                        + class_offset
                        + file_index
                    ).shuffle(data_list)
                for data in data_list:
                    if self._passes_beta(data):
                        yield data

        antiD_stream = class_stream(antiD_files, 100_000)
        antiP_stream = class_stream(antiP_files, 200_000)
        for antiD_graph, antiP_graph in zip(antiD_stream, antiP_stream):
            yield antiD_graph
            yield antiP_graph

    def approx_len(self) -> int:
        """サマリーJSONからグラフ数を集計（.ptを全ロードせず高速）"""
        def count_files(files):
            total = 0
            for pt_path in files:
                summary = Path(pt_path).with_suffix('.json')
                if summary.exists():
                    with open(summary) as f:
                        total += json.load(f).get('n_graphs', 0)
            return total

        if self.balance_tagged_classes:
            antiD_files = [
                path for path in self.pt_files if '_antiD_' in path.name]
            antiP_files = [
                path for path in self.pt_files if '_antiP_' in path.name]
            if antiD_files and antiP_files:
                return 2 * min(
                    count_files(antiD_files),
                    count_files(antiP_files),
                )

        return count_files(self.pt_files)


# ── データロード ───────────────────────────────────────
def make_dataloader(dataset, batch_size: int, num_workers: int,
                    prefetch_factor: int):
    kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': DEVICE.type == 'cuda',
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def pkl_to_pt(pkl_path: str, cache_dir: Path) -> Path:
    """pkl パスをキャッシュ済み .pt パスに変換する。

    中文说明:
    manifest 里保存的是原始 pkl 路径；真正训练读取的是 cache_dir 下
    antiD/antiP 子目录中的 `.pt` graph cache。
    """
    p = Path(pkl_path)
    particle = 'antiD' if 'antiD' in p.stem else 'antiP'
    return cache_dir / particle / (p.stem + '.pt')


def make_loaders_from_manifest(manifest_path: Path, cache_dir: Path, batch_size: int,
                               max_train_files: int = None, max_val_files: int = None,
                               num_workers: int = 0, prefetch_factor: int = 2,
                               beta_min: float = None, beta_max: float = None,
                               seed: int = 42):
    with open(manifest_path) as f:
        manifest = json.load(f)

    def get_pt_files(split: str, max_files: int = None):
        antiD = manifest[split]['antiD']
        antiP = manifest[split]['antiP']
        if max_files is not None:
            n_d = max_files // 2
            n_p = max_files - n_d
            antiD, antiP = antiD[:n_d], antiP[:n_p]
        files = antiD + antiP
        pt_files = [pkl_to_pt(f, cache_dir) for f in files]
        missing = [f for f in pt_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} キャッシュファイルが見つかりません。'
                f'先に cache_graphs.py を実行してください。\n例: {missing[0]}')
        return pt_files

    train_ds = CachedStreamDataset(get_pt_files('train', max_train_files),
                                   shuffle_files=True, shuffle_events=True,
                                   seed=seed,
                                   beta_min=beta_min, beta_max=beta_max)
    val_ds   = CachedStreamDataset(get_pt_files('val',   max_val_files),
                                   shuffle_files=False, shuffle_events=False,
                                   seed=seed,
                                   beta_min=beta_min, beta_max=beta_max)

    train_loader = make_dataloader(
        train_ds, batch_size, num_workers, prefetch_factor)
    val_loader = make_dataloader(
        val_ds, batch_size, num_workers, prefetch_factor)

    return train_loader, val_loader, train_ds, val_ds


def make_loaders_from_split_cache(split_cache_dir: Path, batch_size: int,
                                  num_workers: int = 0,
                                  prefetch_factor: int = 2,
                                  beta_min: float = None,
                                  beta_max: float = None,
                                  seed: int = 42):
    """Load sharded or single-file caches created from split.pkl files.

    中文说明:
    这个模式直接读取 split-cache-dir 下的 train.pt/val.pt，或者
    train_*.pt/val_*.pt 分片文件。50M full 训练若使用预先分片的 graph
    cache，也会走这里。
    """
    def find_split_files(split: str):
        sharded = sorted(split_cache_dir.glob(f'{split}_*.pt'))
        if sharded:
            return sharded
        single = split_cache_dir / f'{split}.pt'
        if single.exists():
            return [single]
        raise FileNotFoundError(
            f'no {split}_*.pt or {split}.pt found under {split_cache_dir}')

    train_ds = CachedStreamDataset(
        find_split_files('train'),
        shuffle_files=True,
        shuffle_events=True,
        seed=seed,
        balance_tagged_classes=True,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    val_ds = CachedStreamDataset(
        find_split_files('val'), shuffle_files=False, shuffle_events=False,
        seed=seed,
        beta_min=beta_min, beta_max=beta_max)
    train_loader = make_dataloader(
        train_ds, batch_size, num_workers, prefetch_factor)
    val_loader = make_dataloader(
        val_ds, batch_size, num_workers, prefetch_factor)
    return train_loader, val_loader, train_ds, val_ds


# ── 訓練ループ ─────────────────────────────────────────
def train(manifest_path: Path, cache_dir: Path, epochs: int = EPOCHS,
          max_train_files: int = None, max_val_files: int = None,
          split_cache_dir: Path = None, batch_size: int = BATCH_SIZE,
          max_train_batches: int = None, max_val_batches: int = None,
          model_name: str = 'gravnet', result_dir: Path = None,
          dataset_tag: str = None, resume_checkpoint: Path = None,
          num_workers: int = 0, prefetch_factor: int = 2,
          non_blocking_transfer: bool = False,
          use_amp: bool = False,
          profile_batches: int = 0,
          use_mc_beta: bool = False,
          use_tof_beta: bool = False,
          use_hit_topology: bool = False,
          use_track_star: bool = False,
          soft_object_aux_weight: float = 0.05,
          multi_task_beta: bool = False,
          classify_with_predicted_beta: bool = False,
          beta_loss_weight: float = 0.1,
          beta_weighted_loss: bool = False,
          beta_bins: str = DEFAULT_BETA_BINS,
          beta_bin_weights_arg: str = DEFAULT_BETA_BIN_WEIGHTS,
          short_tof_antip_weight: float = 1.0,
          beta_min: float = None, beta_max: float = None,
          graph_feature_normalizer: Path = None,
          seed: int = 42):
    if use_mc_beta and use_tof_beta:
        raise ValueError('--use-mc-beta and --use-tof-beta cannot be used together')
    if multi_task_beta and use_mc_beta:
        raise ValueError(
            '--multi-task-beta predicts beta from TreeRec and cannot be combined '
            'with --use-mc-beta')
    if classify_with_predicted_beta and not multi_task_beta:
        raise ValueError(
            '--classify-with-predicted-beta requires --multi-task-beta')
    if multi_task_beta and model_name in ('gravnet_cluster_tokens', 'cluster_tokens_only'):
        raise ValueError('cluster-token models are classification-only in this A/B')
    if model_name == 'gravnet_soft_objects' and multi_task_beta:
        raise ValueError(
            'gravnet_soft_objects keeps its MC auxiliary losses separate from '
            '--multi-task-beta in the first controlled comparison')
    if model_name == 'gravnet_soft_objects' and soft_object_aux_weight < 0.0:
        raise ValueError('--soft-object-aux-weight must be non-negative')
    if multi_task_beta and beta_weighted_loss:
        raise ValueError(
            '--multi-task-beta is intentionally kept separate from '
            '--beta-weighted-loss for the first comparison')
    if beta_loss_weight <= 0.0:
        raise ValueError('--beta-loss-weight must be positive')
    if short_tof_antip_weight < 1.0:
        raise ValueError('--short-tof-antip-weight must be at least 1.0')
    if beta_weighted_loss and short_tof_antip_weight > 1.0:
        raise ValueError(
            '--beta-weighted-loss and --short-tof-antip-weight cannot be '
            'combined in the first controlled comparison')
    if multi_task_beta and short_tof_antip_weight > 1.0:
        raise ValueError(
            '--multi-task-beta and --short-tof-antip-weight cannot be '
            'combined in the first controlled comparison')
    if use_amp and DEVICE.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    if profile_batches < 0:
        raise ValueError('--profile-batches must be non-negative')
    if profile_batches > 0 and DEVICE.type != 'cuda':
        raise ValueError('--profile-batches requires a CUDA device')

    graph_feature_mean = None
    graph_feature_std = None
    if graph_feature_normalizer is not None:
        graph_feature_mean, graph_feature_std = load_graph_feature_normalizer(
            graph_feature_normalizer)
        graph_feature_mean = graph_feature_mean.to(DEVICE)
        graph_feature_std = graph_feature_std.to(DEVICE)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if split_cache_dir is not None:
        print(f'split cache: {split_cache_dir}')
        train_loader, val_loader, train_ds, val_ds = make_loaders_from_split_cache(
            split_cache_dir, batch_size, num_workers, prefetch_factor,
            beta_min=beta_min, beta_max=beta_max, seed=seed)
    else:
        train_loader, val_loader, train_ds, val_ds = make_loaders_from_manifest(
            manifest_path, cache_dir, batch_size,
            max_train_files=max_train_files, max_val_files=max_val_files,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            beta_min=beta_min, beta_max=beta_max, seed=seed)

    train_approx = train_ds.approx_len()
    val_approx   = val_ds.approx_len()
    train_batches = (train_approx + batch_size - 1) // batch_size
    val_batches   = (val_approx   + batch_size - 1) // batch_size
    if max_train_batches is not None:
        train_batches = min(train_batches, max_train_batches)
    if max_val_batches is not None:
        val_batches = min(val_batches, max_val_batches)
    print(f'train events (approx): {train_approx:,}  batches: {train_batches:,}')
    print(f'val   events (approx): {val_approx:,}  batches: {val_batches:,}')
    if beta_min is not None or beta_max is not None:
        lo = beta_min if beta_min is not None else '-inf'
        hi = beta_max if beta_max is not None else '+inf'
        print(f'beta filter: [{lo}, {hi})')
    print(
        f'dataloader: num_workers={num_workers}, '
        f'pin_memory={DEVICE.type == "cuda"}, '
        f'prefetch_factor={prefetch_factor if num_workers > 0 else None}')
    print(
        'host-to-device transfer: '
        f'{"non-blocking enabled" if non_blocking_transfer else "blocking"}')
    print(
        'automatic mixed precision: '
        f'{"FP16 enabled" if use_amp else "disabled"}')
    if profile_batches:
        print(f'timing profile: first {profile_batches} train batches')
    print(f'random seed: {seed}')
    print(
        'graph feature normalization: '
        f'{graph_feature_normalizer if graph_feature_normalizer is not None else "disabled"}')
    graph_feat_dim = 45 + (2 if use_tof_beta else (1 if use_mc_beta else 0))
    if use_hit_topology:
        graph_feat_dim += HIT_TOPOLOGY_FEATURE_DIM
    if use_track_star:
        graph_feat_dim += TRACK_STAR_FEATURE_DIM
    print(
        f'MC beta input: {"enabled" if use_mc_beta else "disabled"} '
        f'| TOF beta input: {"enabled" if use_tof_beta else "disabled"} '
        f'| hit topology input: {"enabled" if use_hit_topology else "disabled"} '
        f'| track/star input: {"enabled" if use_track_star else "disabled"} '
        f'(graph_feat_dim={graph_feat_dim})')
    print(
        'cluster/vertex tokens: '
        f'{"enabled" if model_name in ("gravnet_cluster_tokens", "cluster_tokens_only") else "disabled"}')
    print(
        'explicit physics edges: '
        f'{"enabled" if model_name == "gravnet_physics_edges" else "disabled"}')
    print(
        'soft object queries: '
        f'{"enabled" if model_name == "gravnet_soft_objects" else "disabled"}')
    if model_name == 'gravnet_soft_objects':
        print(
            'soft-object auxiliary targets: training-only '
            f'| weight={soft_object_aux_weight:g}')
    beta_target_mean = None
    beta_target_std = None
    if multi_task_beta:
        beta_target_mean, beta_target_std, beta_target_events = \
            fit_mc_beta_normalizer(
                train_ds.pt_files, beta_min=beta_min, beta_max=beta_max)
        print(
            'beta multi-task: enabled '
            f'| target=(beta-{beta_target_mean:.6g})/{beta_target_std:.6g} '
            f'| loss weight={beta_loss_weight:g} '
            f'| train targets={beta_target_events:,} '
            f'| classifier beta input={classify_with_predicted_beta}')
    else:
        print('beta multi-task: disabled')

    if dataset_tag is None:
        dataset_tag = split_cache_dir.name if split_cache_dir is not None else 'aohba'
    if multi_task_beta:
        if model_name != 'gravnet':
            raise ValueError('--multi-task-beta currently supports --model gravnet only')
        exp_name = f'GravNetMultiTask_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetMultiTaskClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS,
            classify_with_predicted_beta=classify_with_predicted_beta).to(DEVICE)
    elif model_name == 'gravnet_tof':
        exp_name = f'GravNetTOF_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetTOFClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'gravnet_detector':
        exp_name = f'GravNetDetector_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = DetectorAwareGravNetClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'gravnet_attention':
        exp_name = f'GravNetAttention_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetAttentionClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'gravnet_cluster_tokens':
        exp_name = f'GravNetClusterTokens_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetClusterTokenClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'gravnet_physics_edges':
        exp_name = f'GravNetPhysicsEdges_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetPhysicsEdgeClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'gravnet_soft_objects':
        exp_name = f'GravNetSoftObjects_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetSoftObjectClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    elif model_name == 'cluster_tokens_only':
        exp_name = f'ClusterTokensOnly_{dataset_tag}'
        model = ClusterVertexTokenClassifier().to(DEVICE)
    else:
        exp_name = f'GravNet_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=graph_feat_dim, num_blocks=NUM_BLOCKS).to(DEVICE)
    print(f'模型: {exp_name}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(
        f'early stopping: min_epochs={MIN_EPOCHS}, '
        f'patience={PATIENCE}, monitor=val_loss')

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    criterion_none = FocalLoss(gamma=FOCAL_GAMMA, reduction='none')
    beta_criterion = torch.nn.SmoothL1Loss() if multi_task_beta else None
    beta_bin_edges = None
    beta_weights = None
    short_tof_reference_ns = None
    short_tof_scale_ns = None
    if beta_weighted_loss:
        beta_bin_edges_list = parse_float_list(beta_bins)
        beta_weights_list = parse_float_list(beta_bin_weights_arg)
        if len(beta_bin_edges_list) != len(beta_weights_list) + 1:
            raise ValueError(
                '--beta-bins must have exactly one more value than '
                '--beta-bin-weights')
        beta_bin_edges = torch.tensor(
            beta_bin_edges_list, dtype=torch.float32, device=DEVICE)
        beta_weights = torch.tensor(
            beta_weights_list, dtype=torch.float32, device=DEVICE)
        print('beta weighted loss: enabled')
        print(f'  beta bins   : {beta_bin_edges_list}')
        print(f'  bin weights : {beta_weights_list}')
    else:
        print('beta weighted loss: disabled')
    if short_tof_antip_weight > 1.0:
        short_tof_reference_ns, short_tof_scale_ns, short_tof_events = \
            fit_short_tof_antip_profile(train_ds.pt_files)
        print('short-TOF antiP hard-negative loss: enabled')
        print(
            f'  max weight  : {short_tof_antip_weight:g} '
            f'| train antiP with valid TOF={short_tof_events:,}')
        print(
            f'  q25 dt      : {short_tof_reference_ns:.6g} ns '
            f'| sigmoid scale={short_tof_scale_ns:.6g} ns')
    else:
        print('short-TOF antiP hard-negative loss: disabled')
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    best_val_loss   = float('inf')
    patience_counter = 0
    start_epoch = 1

    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location=DEVICE, weights_only=False)
        if checkpoint['model_name'] != model_name:
            raise ValueError(
                f'checkpoint model={checkpoint["model_name"]}, '
                f'requested model={model_name}')
        checkpoint_use_mc_beta = bool(checkpoint.get('use_mc_beta', False))
        if checkpoint_use_mc_beta != use_mc_beta:
            raise ValueError(
                f'checkpoint use_mc_beta={checkpoint_use_mc_beta}, '
                f'requested use_mc_beta={use_mc_beta}')
        checkpoint_use_tof_beta = bool(checkpoint.get('use_tof_beta', False))
        if checkpoint_use_tof_beta != use_tof_beta:
            raise ValueError(
                f'checkpoint use_tof_beta={checkpoint_use_tof_beta}, '
                f'requested use_tof_beta={use_tof_beta}')
        checkpoint_use_hit_topology = bool(
            checkpoint.get('use_hit_topology', False))
        if checkpoint_use_hit_topology != use_hit_topology:
            raise ValueError(
                f'checkpoint use_hit_topology={checkpoint_use_hit_topology}, '
                f'requested use_hit_topology={use_hit_topology}')
        checkpoint_use_track_star = bool(checkpoint.get('use_track_star', False))
        if checkpoint_use_track_star != use_track_star:
            raise ValueError(
                f'checkpoint use_track_star={checkpoint_use_track_star}, '
                f'requested use_track_star={use_track_star}')
        checkpoint_use_amp = bool(checkpoint.get('use_amp', False))
        if checkpoint_use_amp != use_amp:
            raise ValueError(
                f'checkpoint use_amp={checkpoint_use_amp}, '
                f'requested use_amp={use_amp}')
        checkpoint_multi_task_beta = bool(checkpoint.get('multi_task_beta', False))
        if checkpoint_multi_task_beta != multi_task_beta:
            raise ValueError(
                f'checkpoint multi_task_beta={checkpoint_multi_task_beta}, '
                f'requested multi_task_beta={multi_task_beta}')
        checkpoint_classify_with_predicted_beta = bool(
            checkpoint.get('classify_with_predicted_beta', False))
        if checkpoint_classify_with_predicted_beta != classify_with_predicted_beta:
            raise ValueError(
                'checkpoint classify_with_predicted_beta='
                f'{checkpoint_classify_with_predicted_beta}, requested '
                f'{classify_with_predicted_beta}')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        if use_amp:
            scaler.load_state_dict(checkpoint['scaler_state'])
        best_val_loss = float(checkpoint['best_val_loss'])
        patience_counter = int(checkpoint['patience_counter'])
        start_epoch = int(checkpoint['epoch']) + 1
        log_dir = resume_checkpoint.parent
        best_model_path = Path(checkpoint['best_model_path'])
        latest_checkpoint_path = resume_checkpoint
        writer = SummaryWriter(
            log_dir=str(log_dir), purge_step=start_epoch)
        print(
            f'resumed checkpoint: {resume_checkpoint} '
            f'(next epoch={start_epoch}, best_val_loss={best_val_loss:.4f})')
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        result_root = (
            result_dir if result_dir is not None
            else PROJECT_ROOT / 'results'
        )
        log_dir = result_root / f'{timestamp}_{exp_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        best_model_path = log_dir / f'{timestamp}_{exp_name}_best.pth'
        latest_checkpoint_path = (
            log_dir / f'{timestamp}_{exp_name}_last_checkpoint.pth'
        )

    if start_epoch > epochs:
        raise ValueError(
            f'checkpoint already reached epoch {start_epoch - 1}, '
            f'but --epochs={epochs}')

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────
        model.train()
        total_loss, total_class_loss, total_beta_loss = 0.0, 0.0, 0.0
        total_soft_aux_loss = 0.0
        total_correct, total_samples = 0, 0
        profile_records = []
        profile_data_wait = []
        previous_submit_end = time.perf_counter()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{epochs} [train]',
                         total=train_batches, leave=False)
        for batch_idx, batch in enumerate(train_bar):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            profile_this_batch = profile_batches > 0 and batch_idx < profile_batches
            if profile_this_batch and batch_idx > 0:
                profile_data_wait.append(time.perf_counter() - previous_submit_end)
            if profile_this_batch:
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
            batch = batch.to(DEVICE, non_blocking=non_blocking_transfer)
            if profile_this_batch:
                h2d_end.record()
                compute_start.record()
            optimizer.zero_grad()
            # 图级特征 45维；MC beta 追加 1 维，TOF beta 追加 beta 和有效标记 2 维。
            # 节点特征 batch.x 已经在 graph cache 中，
            # 这里把 event-level summary 拼成 graph_feat 后交给 GravNet。
            with autocast_context():
                graph_feat = build_graph_feat(
                    batch,
                    use_mc_beta=use_mc_beta,
                    use_tof_beta=use_tof_beta,
                    use_hit_topology=use_hit_topology,
                    use_track_star=use_track_star,
                    graph_feature_mean=graph_feature_mean,
                    graph_feature_std=graph_feature_std,
                )
                model_output = forward_model(model, model_name, batch, graph_feat)
            if multi_task_beta:
                logits, beta_prediction = model_output
            elif model_name == 'gravnet_soft_objects':
                logits, soft_auxiliary = model_output
            else:
                logits = model_output
            targets = batch.y.view(-1)
            if beta_weighted_loss:
                if not hasattr(batch, 'mc_beta'):
                    raise ValueError(
                        '--beta-weighted-loss requires mc_beta in graph cache')
                losses = criterion_none(logits.float(), targets)
                weights = beta_bin_weights(
                    batch.mc_beta.view(-1).float(),
                    beta_bin_edges,
                    beta_weights)
                class_loss = (
                    (losses * weights).sum() / weights.sum().clamp_min(1.0))
                beta_loss = None
                loss = class_loss
            elif short_tof_antip_weight > 1.0:
                losses = criterion_none(logits.float(), targets)
                weights = short_tof_antip_weights(
                    batch, targets, short_tof_antip_weight,
                    short_tof_reference_ns, short_tof_scale_ns)
                class_loss = (
                    (losses * weights).sum() / weights.sum().clamp_min(1.0))
                beta_loss = None
                loss = class_loss
            else:
                class_loss = criterion(logits.float(), targets)
                if multi_task_beta:
                    beta_target = (
                        batch.mc_beta.view(-1).float() - beta_target_mean
                    ) / beta_target_std
                    beta_loss = beta_criterion(beta_prediction.float(), beta_target)
                    loss = class_loss + beta_loss_weight * beta_loss
                elif model_name == 'gravnet_soft_objects':
                    soft_aux_loss, _, _ = soft_object_auxiliary_loss(
                        soft_auxiliary, batch)
                    beta_loss = None
                    loss = class_loss + soft_object_aux_weight * soft_aux_loss
                else:
                    beta_loss = None
                    loss = class_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if profile_this_batch:
                compute_end.record()
                profile_records.append((h2d_start, h2d_end, compute_start, compute_end))
            previous_submit_end = time.perf_counter()

            total_loss    += loss.item() * batch.num_graphs
            total_class_loss += class_loss.item() * batch.num_graphs
            if beta_loss is not None:
                total_beta_loss += beta_loss.item() * batch.num_graphs
            if model_name == 'gravnet_soft_objects':
                total_soft_aux_loss += soft_aux_loss.item() * batch.num_graphs
            preds          = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch.num_graphs
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        if profile_records:
            torch.cuda.synchronize()
            h2d_ms = [start.elapsed_time(end) for start, end, _, _ in profile_records]
            compute_ms = [start.elapsed_time(end) for _, _, start, end in profile_records]
            avg_wait_ms = (
                1000.0 * sum(profile_data_wait) / len(profile_data_wait)
                if profile_data_wait else 0.0)
            print(
                f'profile ({len(profile_records)} train batches; first fetch excluded): '
                f'dataloader_wait={avg_wait_ms:.2f} ms/batch | '
                f'h2d={sum(h2d_ms) / len(h2d_ms):.2f} ms/batch | '
                f'gpu_compute={sum(compute_ms) / len(compute_ms):.2f} ms/batch')

        train_loss = total_loss / total_samples
        train_class_loss = total_class_loss / total_samples
        train_beta_loss = total_beta_loss / total_samples if multi_task_beta else None
        train_soft_aux_loss = (
            total_soft_aux_loss / total_samples
            if model_name == 'gravnet_soft_objects' else None)
        train_acc  = total_correct / total_samples

        # ── Validation ───────────────────────────────
        model.eval()
        val_loss, val_beta_loss, val_correct, val_samples = 0.0, 0.0, 0, 0
        with torch.no_grad(), autocast_context():
            for batch_idx, batch in enumerate(tqdm(
                    val_loader, desc=f'Epoch {epoch:3d}/{epochs} [val]  ',
                    total=val_batches, leave=False)):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break
                batch = batch.to(DEVICE, non_blocking=non_blocking_transfer)
                # 验证阶段使用同样的图级特征。
                graph_feat = build_graph_feat(
                    batch,
                    use_mc_beta=use_mc_beta,
                    use_tof_beta=use_tof_beta,
                    use_hit_topology=use_hit_topology,
                    use_track_star=use_track_star,
                    graph_feature_mean=graph_feature_mean,
                    graph_feature_std=graph_feature_std,
                )
                model_output = forward_model(model, model_name, batch, graph_feat)
                if multi_task_beta:
                    logits, beta_prediction = model_output
                elif model_name == 'gravnet_soft_objects':
                    logits, _ = model_output
                else:
                    logits = model_output
                loss = criterion(logits.float(), batch.y.view(-1))
                if multi_task_beta:
                    beta_target = (
                        batch.mc_beta.view(-1).float() - beta_target_mean
                    ) / beta_target_std
                    beta_loss = beta_criterion(beta_prediction.float(), beta_target)
                    val_beta_loss += beta_loss.item() * batch.num_graphs
                val_loss    += loss.item() * batch.num_graphs
                preds        = logits.argmax(dim=1)
                val_correct += (preds == batch.y.view(-1)).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc  = val_correct / val_samples
        scheduler.step()
        elapsed = time.time() - epoch_start

        train_loss_text = (
            f'train_loss: {train_loss:.4f} '
            f'(class={train_class_loss:.4f}, beta={train_beta_loss:.4f})'
            if multi_task_beta else f'train_loss: {train_loss:.4f}')
        if model_name == 'gravnet_soft_objects':
            train_loss_text = (
                f'train_loss: {train_loss:.4f} '
                f'(class={train_class_loss:.4f}, soft_aux={train_soft_aux_loss:.4f})')
        val_loss_text = (
            f'val_loss: {val_loss:.4f} '
            f'val_beta_loss: {val_beta_loss / val_samples:.4f}'
            if multi_task_beta else f'val_loss: {val_loss:.4f}')
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'{train_loss_text}  train_acc: {train_acc:.4f} | '
              f'{val_loss_text}  val_acc: {val_acc:.4f} | '
              f'lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_class', train_class_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        if multi_task_beta:
            writer.add_scalar('Loss/train_beta', train_beta_loss, epoch)
            writer.add_scalar('Loss/val_beta', val_beta_loss / val_samples, epoch)
        if model_name == 'gravnet_soft_objects':
            writer.add_scalar('Loss/train_soft_aux', train_soft_aux_loss, epoch)
        writer.add_scalar('Acc/train',  train_acc,  epoch)
        writer.add_scalar('Acc/val',    val_acc,    epoch)

        should_stop = False
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'  → best model saved (val_loss={best_val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and epoch >= MIN_EPOCHS:
                should_stop = True

        torch.save({
            'epoch': epoch,
            'model_name': model_name,
            'dataset_tag': dataset_tag,
            'seed': seed,
            'use_mc_beta': use_mc_beta,
            'use_tof_beta': use_tof_beta,
            'use_hit_topology': use_hit_topology,
            'use_track_star': use_track_star,
            'use_cluster_vertex_tokens': model_name in (
                'gravnet_cluster_tokens', 'cluster_tokens_only'),
            'use_physics_edges': model_name == 'gravnet_physics_edges',
            'use_soft_objects': model_name == 'gravnet_soft_objects',
            'soft_object_aux_weight': (
                soft_object_aux_weight
                if model_name == 'gravnet_soft_objects' else None),
            'use_amp': use_amp,
            'multi_task_beta': multi_task_beta,
            'classify_with_predicted_beta': classify_with_predicted_beta,
            'beta_loss_weight': beta_loss_weight if multi_task_beta else None,
            'beta_target_mean': beta_target_mean,
            'beta_target_std': beta_target_std,
            'short_tof_antip_weight': short_tof_antip_weight,
            'short_tof_reference_ns': short_tof_reference_ns,
            'short_tof_scale_ns': short_tof_scale_ns,
            'graph_feature_normalizer': (
                str(graph_feature_normalizer)
                if graph_feature_normalizer is not None else None),
            'graph_feat_dim': graph_feat_dim,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if use_amp else None,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'best_model_path': str(best_model_path),
        }, latest_checkpoint_path)
        print(f'  → latest checkpoint saved: {latest_checkpoint_path}')

        if should_stop:
            print(f'  → early stopping: val_loss未改善已达{PATIENCE}个epoch')
            break

    writer.close()
    print(f'\n训练完成，最优模型: {best_model_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path,
                    default=Path('/mnt/ynakagami3/aohba_preprocess/split/split_manifest.json'))
    ap.add_argument('--cache-dir', type=Path,
                    default=Path('/mnt/ynakagami3/aohba_preprocess/graph_cache_v2'))
    ap.add_argument('--split-cache-dir', type=Path, default=None,
                    help='train.pt/val.pt direct cache directory')
    ap.add_argument('--epochs',          type=int, default=EPOCHS)
    ap.add_argument('--batch-size',      type=int, default=BATCH_SIZE)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader worker数。0なら従来通り単一プロセス')
    ap.add_argument('--prefetch-factor', type=int, default=2,
                    help='num-workers > 0 の時に各workerが先読みするbatch数')
    ap.add_argument('--non-blocking-transfer', action='store_true',
                    help='overlap pinned-memory host-to-device copies when possible')
    ap.add_argument('--amp', action='store_true',
                    help='use CUDA FP16 autocast with GradScaler')
    ap.add_argument('--profile-batches', type=int, default=0,
                    help='time the first N train batches (CUDA only)')
    ap.add_argument('--seed', type=int, default=42,
                    help='model initialization and data shuffling seed')
    ap.add_argument('--model', choices=['gravnet', 'gravnet_tof', 'gravnet_detector', 'gravnet_attention', 'gravnet_cluster_tokens', 'cluster_tokens_only', 'gravnet_physics_edges', 'gravnet_soft_objects'],
                    default='gravnet')
    ap.add_argument('--result-dir', type=Path, default=None)
    ap.add_argument(
        '--dataset-tag',
        default=None,
        help='name used in the experiment/result directory',
    )
    ap.add_argument(
        '--resume-checkpoint',
        type=Path,
        default=None,
        help='resume model, optimizer and scheduler from a last checkpoint',
    )
    ap.add_argument('--max-train-files', type=int, default=None, help='训练文件数上限（smoke test用）')
    ap.add_argument('--max-val-files',   type=int, default=None, help='验证文件数上限（smoke test用）')
    ap.add_argument('--max-train-batches', type=int, default=None,
                    help='训练batch数上限（smoke test用）')
    ap.add_argument('--max-val-batches', type=int, default=None,
                    help='验证batch数上限（smoke test用）')
    ap.add_argument('--use-mc-beta', action='store_true',
                    help='append TreeMc primary beta to graph-level features')
    ap.add_argument('--use-tof-beta', action='store_true',
                    help='append beta reconstructed from TreeRec TOF hits and a validity mask')
    ap.add_argument('--use-hit-topology', action='store_true',
                    help='append six precomputed TreeRec hit-level topology summaries')
    ap.add_argument('--use-track-star', action='store_true',
                    help='append four precomputed TreeRec track/star geometry candidates')
    ap.add_argument('--soft-object-aux-weight', type=float, default=0.05,
                    help='non-negative training-only weight for MC stop/direction auxiliary losses')
    ap.add_argument('--multi-task-beta', action='store_true',
                    help='jointly train a beta-regression head using mc_beta as a target, not an input')
    ap.add_argument('--classify-with-predicted-beta', action='store_true',
                    help='append the model-predicted beta to the classification head; requires --multi-task-beta')
    ap.add_argument('--beta-loss-weight', type=float, default=0.1,
                    help='weight for the standardized beta regression SmoothL1 loss')
    ap.add_argument('--beta-weighted-loss', action='store_true',
                    help='train loss に beta-bin ごとの重みを掛ける')
    ap.add_argument('--beta-bins', default=DEFAULT_BETA_BINS,
                    help='comma-separated beta bin edges')
    ap.add_argument('--beta-bin-weights', default=DEFAULT_BETA_BIN_WEIGHTS,
                    help='comma-separated train loss weights for beta bins')
    ap.add_argument(
        '--short-tof-antip-weight', type=float, default=1.0,
        help=(
            'maximum smooth loss multiplier for short-flight-time antiP; '
            '1.0 disables the train-only TreeRec hard-negative weighting'))
    ap.add_argument('--beta-min', type=float, default=None,
                    help='keep graphs with mc_beta >= beta_min')
    ap.add_argument('--beta-max', type=float, default=None,
                    help='keep graphs with mc_beta < beta_max')
    ap.add_argument(
        '--graph-feature-normalizer', type=Path, default=None,
        help='JSON fitted on this dataset\'s train split: log1p first 38 graph features, then z-score all 45',
    )
    args = ap.parse_args()
    train(args.manifest, args.cache_dir, epochs=args.epochs,
          max_train_files=args.max_train_files,
          max_val_files=args.max_val_files,
          split_cache_dir=args.split_cache_dir,
          batch_size=args.batch_size,
          max_train_batches=args.max_train_batches,
          max_val_batches=args.max_val_batches,
          model_name=args.model,
          result_dir=args.result_dir,
          dataset_tag=args.dataset_tag,
          resume_checkpoint=args.resume_checkpoint,
          num_workers=args.num_workers,
          prefetch_factor=args.prefetch_factor,
          non_blocking_transfer=args.non_blocking_transfer,
          use_amp=args.amp,
          profile_batches=args.profile_batches,
          use_mc_beta=args.use_mc_beta,
          use_tof_beta=args.use_tof_beta,
          use_hit_topology=args.use_hit_topology,
          use_track_star=args.use_track_star,
          soft_object_aux_weight=args.soft_object_aux_weight,
          multi_task_beta=args.multi_task_beta,
          classify_with_predicted_beta=args.classify_with_predicted_beta,
          beta_loss_weight=args.beta_loss_weight,
          beta_weighted_loss=args.beta_weighted_loss,
          beta_bins=args.beta_bins,
          beta_bin_weights_arg=args.beta_bin_weights,
          short_tof_antip_weight=args.short_tof_antip_weight,
          beta_min=args.beta_min,
          beta_max=args.beta_max,
          graph_feature_normalizer=args.graph_feature_normalizer,
          seed=args.seed)
