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
from GAPS_Project.src.data_parse.graph_builder import GraphBuilder
from GAPS_Project.src.losses import FocalLoss
from GAPS_Project.src.models.gravnet import GravNetClassifier

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── ハイパーパラメータ ──────────────────────────────────
EPOCHS       = 80
BATCH_SIZE   = 128
LR           = 3e-4
STEP_SIZE    = 15
GAMMA        = 0.5
FOCAL_GAMMA  = 1.5
PATIENCE     = 10
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
class PklStreamDataset(IterableDataset):
    """
    pkl ファイルリストを1ファイルずつ読み込み、
    event をストリーミングする IterableDataset。
    全イベントをメモリに展開しないため大規模データに対応。
    """

    def __init__(self, pkl_files: list, builder: GraphBuilder,
                 shuffle_files: bool = True, shuffle_events: bool = True, seed: int = 42):
        self.pkl_files     = list(pkl_files)
        self.builder       = builder
        self.shuffle_files  = shuffle_files
        self.shuffle_events = shuffle_events
        self.seed          = seed
        self._epoch        = 0

    def __iter__(self):
        files = self.pkl_files.copy()
        if self.shuffle_files:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(files)
        self._epoch += 1

        for pkl_path in files:
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
            events = payload['events']
            if self.shuffle_events:
                random.Random(self.seed + self._epoch).shuffle(events)
            for event in events:
                yield self.builder.build_from_dict(event)

    def approx_len(self) -> int:
        """summary.json から event 数を集計（進捗表示用）"""
        total = 0
        for pkl_path in self.pkl_files:
            summary = Path(pkl_path).with_suffix('').with_name(
                Path(pkl_path).stem + '_summary.json')
            if summary.exists():
                with open(summary) as f:
                    total += json.load(f).get('total_events', 0)
        return total


# ── データロード ───────────────────────────────────────
def make_loaders_from_manifest(manifest_path: Path, builder: GraphBuilder, batch_size: int,
                               max_train_files: int = None, max_val_files: int = None):
    with open(manifest_path) as f:
        manifest = json.load(f)

    def get_files(split: str, max_files: int = None):
        antiD_files = manifest[split]['antiD']
        antiP_files = manifest[split]['antiP']
        if max_files is not None:
            n_d = max_files // 2
            n_p = max_files - n_d
            return antiD_files[:n_d] + antiP_files[:n_p]
        return antiD_files + antiP_files

    train_ds = PklStreamDataset(get_files('train', max_train_files), builder, shuffle_files=True,  shuffle_events=True)
    val_ds   = PklStreamDataset(get_files('val',   max_val_files),   builder, shuffle_files=False, shuffle_events=False)
    test_ds  = PklStreamDataset(get_files('test'),                   builder, shuffle_files=False, shuffle_events=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds, val_ds


# ── 訓練ループ ─────────────────────────────────────────
def train(manifest_path: Path, epochs: int = EPOCHS,
          max_train_files: int = None, max_val_files: int = None):
    builder = GraphBuilder(k=8, normalize=True)
    train_loader, val_loader, _, train_ds, val_ds = make_loaders_from_manifest(
        manifest_path, builder, BATCH_SIZE,
        max_train_files=max_train_files, max_val_files=max_val_files)

    train_approx = train_ds.approx_len()
    val_approx   = val_ds.approx_len()
    train_batches = (train_approx + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches   = (val_approx   + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'train events (approx): {train_approx:,}  batches: {train_batches:,}')
    print(f'val   events (approx): {val_approx:,}  batches: {val_batches:,}')

    exp_name = f'GravNet_{NUM_BLOCKS}b_h{HIDDEN_DIM}_aohba'
    model = GravNetClassifier(
        in_channels=IN_CHANNEL, hidden_dim=HIDDEN_DIM,
        graph_feat_dim=45, num_blocks=NUM_BLOCKS).to(DEVICE)
    print(f'モデル: {exp_name}')
    print(f'パラメータ数: {sum(p.numel() for p in model.parameters()):,}')

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = PROJECT_ROOT / 'results' / f'{timestamp}_{exp_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    best_model_path = log_dir / f'{timestamp}_{exp_name}_best.pth'

    best_val_loss   = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{epochs} [train]',
                         total=train_batches, leave=False)
        for batch in train_bar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
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
            for batch in tqdm(val_loader, desc=f'Epoch {epoch:3d}/{epochs} [val]  ',
                              total=val_batches, leave=False):
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
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

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'  → best model saved (val_loss={best_val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and epoch > PATIENCE:
                print(f'  → early stopping: val_loss未改善已达{PATIENCE}个epoch')
                break

    writer.close()
    print(f'\n训练完成，最优模型: {best_model_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path,
                    default=Path('/mnt/ynakagami3/aohba_preprocess/split/split_manifest.json'))
    ap.add_argument('--epochs',          type=int, default=EPOCHS)
    ap.add_argument('--max-train-files', type=int, default=None, help='训练文件数上限（smoke test用）')
    ap.add_argument('--max-val-files',   type=int, default=None, help='验证文件数上限（smoke test用）')
    args = ap.parse_args()
    train(args.manifest, epochs=args.epochs,
          max_train_files=args.max_train_files, max_val_files=args.max_val_files)
