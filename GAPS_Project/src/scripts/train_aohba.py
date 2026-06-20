"""
新GAPS 5000万データ用GravNet訓練スクリプト。
split_manifest.json からpklファイルリストを読み込み、
IterableDataset で1ファイルずつストリーミングしてメモリを節約する。

使い方:
  python src/scripts/train_aohba.py \
      --manifest /mnt/ynakagami3/aohba_preprocess/split/split_manifest.json
"""
import argparse
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.losses import FocalLoss
from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.gravnet_tof import GravNetTOFClassifier

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

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'使用设备：{DEVICE}')


# ── IterableDataset ────────────────────────────────────
class CachedStreamDataset(IterableDataset):
    """
    キャッシュ済み .pt ファイル（PyG Data リスト）を1ファイルずつ読み込む
    IterableDataset。GraphBuilder の実行が不要で高速。
    """

    def __init__(self, pt_files: list, shuffle_files: bool = True,
                 shuffle_events: bool = True, seed: int = 42,
                 balance_tagged_classes: bool = False):
        self.pt_files       = list(pt_files)
        self.shuffle_files  = shuffle_files
        self.shuffle_events = shuffle_events
        self.seed           = seed
        self.balance_tagged_classes = balance_tagged_classes
        self._epoch         = 0

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

        for file_index, pt_path in enumerate(files):
            data_list = torch.load(pt_path, weights_only=False)
            if self.shuffle_events:
                random.Random(
                    self.seed + epoch * 1_000_003 + file_index
                ).shuffle(data_list)
            for data in data_list:
                yield data

    def _iter_balanced(self, antiD_files, antiP_files, epoch):
        """Interleave class-pure shards so every batch is class balanced."""
        antiD_files = antiD_files.copy()
        antiP_files = antiP_files.copy()
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(antiD_files)
            random.Random(self.seed + epoch + 10_000).shuffle(antiP_files)

        def class_stream(files, class_offset):
            for file_index, pt_path in enumerate(files):
                data_list = torch.load(pt_path, weights_only=False)
                if self.shuffle_events:
                    random.Random(
                        self.seed
                        + epoch * 1_000_003
                        + class_offset
                        + file_index
                    ).shuffle(data_list)
                yield from data_list

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
def pkl_to_pt(pkl_path: str, cache_dir: Path) -> Path:
    """pkl パスをキャッシュ済み .pt パスに変換する"""
    p = Path(pkl_path)
    particle = 'antiD' if 'antiD' in p.stem else 'antiP'
    return cache_dir / particle / (p.stem + '.pt')


def make_loaders_from_manifest(manifest_path: Path, cache_dir: Path, batch_size: int,
                               max_train_files: int = None, max_val_files: int = None):
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
                                   shuffle_files=True, shuffle_events=True)
    val_ds   = CachedStreamDataset(get_pt_files('val',   max_val_files),
                                   shuffle_files=False, shuffle_events=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, train_ds, val_ds


def make_loaders_from_split_cache(split_cache_dir: Path, batch_size: int):
    """Load sharded or single-file caches created from split.pkl files."""
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
        balance_tagged_classes=True,
    )
    val_ds = CachedStreamDataset(
        find_split_files('val'), shuffle_files=False, shuffle_events=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, train_ds, val_ds


# ── 訓練ループ ─────────────────────────────────────────
def train(manifest_path: Path, cache_dir: Path, epochs: int = EPOCHS,
          max_train_files: int = None, max_val_files: int = None,
          split_cache_dir: Path = None, batch_size: int = BATCH_SIZE,
          max_train_batches: int = None, max_val_batches: int = None,
          model_name: str = 'gravnet', result_dir: Path = None,
          dataset_tag: str = None, resume_checkpoint: Path = None):
    if split_cache_dir is not None:
        print(f'split cache: {split_cache_dir}')
        train_loader, val_loader, train_ds, val_ds = make_loaders_from_split_cache(
            split_cache_dir, batch_size)
    else:
        train_loader, val_loader, train_ds, val_ds = make_loaders_from_manifest(
            manifest_path, cache_dir, batch_size,
            max_train_files=max_train_files, max_val_files=max_val_files)

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

    if dataset_tag is None:
        dataset_tag = split_cache_dir.name if split_cache_dir is not None else 'aohba'
    if model_name == 'gravnet_tof':
        exp_name = f'GravNetTOF_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetTOFClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=45, num_blocks=NUM_BLOCKS).to(DEVICE)
    else:
        exp_name = f'GravNet_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{dataset_tag}'
        model = GravNetClassifier(
            in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
            graph_feat_dim=45, num_blocks=NUM_BLOCKS).to(DEVICE)
    print(f'模型: {exp_name}')
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(
        f'early stopping: min_epochs={MIN_EPOCHS}, '
        f'patience={PATIENCE}, monitor=val_loss')

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

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
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
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
        total_loss, total_correct, total_samples = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{epochs} [train]',
                         total=train_batches, leave=False)
        for batch_idx, batch in enumerate(train_bar):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)
            if model_name == 'gravnet_tof':
                if not hasattr(batch, 'tof_paddle_energy'):
                    raise ValueError(
                        'GravNetTOF requires tof_paddle_energy in graph cache')
                logits = model(
                    batch.x, batch.edge_index, batch.batch,
                    graph_feat=graph_feat,
                    tof_paddle_energy=batch.tof_paddle_energy.view(-1, 172),
                )
            else:
                logits = model(
                    batch.x, batch.edge_index, batch.batch,
                    graph_feat=graph_feat)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * batch.num_graphs
            preds          = logits.argmax(dim=1)
            total_correct += (preds == batch.y.view(-1)).sum().item()
            total_samples += batch.num_graphs
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # ── Validation ───────────────────────────────
        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(
                    val_loader, desc=f'Epoch {epoch:3d}/{epochs} [val]  ',
                    total=val_batches, leave=False)):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)
                if model_name == 'gravnet_tof':
                    logits = model(
                        batch.x, batch.edge_index, batch.batch,
                        graph_feat=graph_feat,
                        tof_paddle_energy=batch.tof_paddle_energy.view(-1, 172),
                    )
                else:
                    logits = model(
                        batch.x, batch.edge_index, batch.batch,
                        graph_feat=graph_feat)
                loss   = criterion(logits, batch.y.view(-1))
                val_loss    += loss.item() * batch.num_graphs
                preds        = logits.argmax(dim=1)
                val_correct += (preds == batch.y.view(-1)).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc  = val_correct / val_samples
        scheduler.step()
        elapsed = time.time() - epoch_start

        print(f'Epoch {epoch:3d}/{epochs} | '
              f'train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | '
              f'val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | '
              f'lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
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
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
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
    ap.add_argument('--model', choices=['gravnet', 'gravnet_tof'],
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
          resume_checkpoint=args.resume_checkpoint)
