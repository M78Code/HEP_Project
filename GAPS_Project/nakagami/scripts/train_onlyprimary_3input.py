"""Train Nakagami ynakagami2 three-input reproduction model."""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

NAKAGAMI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NAKAGAMI_ROOT))

from data_parse.three_input_dataset import ThreeInputDataset
from models.nakagami_three_input import NakagamiThreeInputNet


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for voxel, tof_paddle, tof_primary, label in tqdm(loader, desc='val', leave=False, dynamic_ncols=True):
            voxel = voxel.to(device)
            tof_paddle = tof_paddle.to(device)
            tof_primary = tof_primary.to(device)
            label_f = label.float().to(device)
            label_i = label.to(device)
            out = model(voxel, tof_paddle, tof_primary)
            loss = criterion(out, label_f)
            pred = (out > 0).long()
            total_loss += loss.item() * label.size(0)
            total_correct += (pred == label_i).sum().item()
            total += label.size(0)
    return total_loss / total, total_correct / total


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f'device: {device}')
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

    data_dir = Path(args.data_dir)
    train_set = ThreeInputDataset(data_dir / 'train_onlyprimary_4M', normalize=args.normalize, max_events=args.max_train_events)
    val_set = ThreeInputDataset(data_dir / 'val_onlyprimary_4M', normalize=args.normalize, max_events=args.max_val_events)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = NakagamiThreeInputNet(dropout_res=0.1, dropout_dense=0.2).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        clean_state = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state)
        print(f'resumed model weights from: {args.resume}')

    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f'model: NakagamiThreeInputNet')
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, patience={args.patience}, shuffle={args.shuffle}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'nakagami_onlyprimary_3input_best.pth'

    best_val_acc = -1.0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{args.epochs} train', leave=False, dynamic_ncols=True)
        for voxel, tof_paddle, tof_primary, label in bar:
            voxel = voxel.to(device)
            tof_paddle = tof_paddle.to(device)
            tof_primary = tof_primary.to(device)
            label_f = label.float().to(device)
            label_i = label.to(device)

            optimizer.zero_grad()
            out = model(voxel, tof_paddle, tof_primary)
            loss = criterion(out, label_f)
            loss.backward()
            optimizer.step()

            pred = (out > 0).long()
            bs = label.size(0)
            train_loss += loss.item() * bs
            train_correct += (pred == label_i).sum().item()
            train_total += bs
            bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= train_total
        train_acc = train_correct / train_total
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            print(f'Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.0f}s')

            # Keras original monitors val_accuracy.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, save_path)
                print(f'  -> best saved: {save_path} (val_acc={val_acc:.4f})')
            else:
                patience += 1
                if patience >= args.patience:
                    print(f'  -> early stopping at epoch {epoch}')
                    break
        else:
            elapsed = time.time() - t0
            print(f'Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val skipped | {elapsed:.0f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/mnt/ynakagami3/nakagami_data/data_4M_onlyprimary')
    parser.add_argument('--save-dir', default=str(NAKAGAMI_ROOT / 'results' / 'onlyprimary_3input'))
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--normalize', action='store_true', help='not strict reproduction; use only for stability tests')
    parser.add_argument('--shuffle', action='store_true', help='shuffle training samples in DataLoader; off by default because CSV files are already shuffled')
    parser.add_argument('--data-parallel', action='store_true', help='use torch.nn.DataParallel when multiple GPUs are available')
    parser.add_argument('--max-train-events', type=int, default=None, help='use only the first N training events for quick tests')
    parser.add_argument('--max-val-events', type=int, default=None, help='use only the first N validation events for quick tests')
    parser.add_argument('--eval-every', type=int, default=1, help='run validation every N epochs')
    parser.add_argument('--resume', default=None, help='load model weights from a previous best .pth file')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
