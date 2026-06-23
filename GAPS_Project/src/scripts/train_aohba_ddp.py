"""Two-GPU DDP training for mixed AOHBA graph-cache shards.

Launch from the GAPS_Project directory:

  torchrun --standalone --nproc_per_node=2 \
      src/scripts/train_aohba_ddp.py \
      --split-cache-dir dataset/aohba4M_atrest_tof172_mixed \
      --model gravnet \
      --dataset-tag aohba4M_atrest_mixed_ddp \
      --epochs 80 \
      --batch-size 128

``--batch-size`` is the local batch size per GPU. With two processes,
``--batch-size 128`` gives a global batch size of 256.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

EPOCHS = 80
BATCH_SIZE = 128
LR = 3e-4
STEP_SIZE = 15
GAMMA = 0.5
FOCAL_GAMMA = 1.5
PATIENCE = 10
MIN_EPOCHS = 20
IN_CHANNEL = 8
NUM_BLOCKS = 6
HIDDEN_DIM = 128


def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "launch this script with torchrun --nproc_per_node=2"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return (
        local_rank,
        dist.get_rank(),
        dist.get_world_size(),
        torch.device(f"cuda:{local_rank}"),
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, *args, **kwargs):
    if is_main(rank):
        print(*args, **kwargs)


class RankShardDataset(IterableDataset):
    """Stream only the mixed shards assigned to one DDP rank."""

    def __init__(
        self,
        pt_files: list[Path],
        rank: int,
        world_size: int,
        shuffle_files: bool,
        shuffle_events: bool,
        seed: int = 42,
    ):
        if len(pt_files) % world_size:
            raise ValueError(
                f"{len(pt_files)} shards cannot be divided equally across "
                f"{world_size} ranks"
            )
        self.all_files = list(pt_files)
        self.rank = rank
        self.world_size = world_size
        self.shuffle_files = shuffle_files
        self.shuffle_events = shuffle_events
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def load_shard(path: Path):
        try:
            graphs = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to load graph-cache shard {path}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not isinstance(graphs, list) or not graphs:
            raise RuntimeError(
                f"invalid graph-cache shard {path}: "
                f"expected non-empty list, got {type(graphs).__name__}"
            )
        return graphs

    def files_for_epoch(self, epoch: int) -> list[Path]:
        files = self.all_files.copy()
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(files)
        return files[self.rank :: self.world_size]

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        for file_index, path in enumerate(self.files_for_epoch(epoch)):
            graphs = self.load_shard(path)
            if self.shuffle_events:
                random.Random(
                    self.seed
                    + epoch * 1_000_003
                    + self.rank * 100_003
                    + file_index
                ).shuffle(graphs)
            yield from graphs

    @staticmethod
    def count_file(path: Path) -> int:
        summary_path = path.with_suffix(".json")
        if not summary_path.exists():
            raise FileNotFoundError(
                f"missing shard summary: {summary_path}"
            )
        with open(summary_path, encoding="utf-8") as file:
            return int(json.load(file)["n_graphs"])

    def local_len(self) -> int:
        return sum(
            self.count_file(path)
            for path in self.all_files[self.rank :: self.world_size]
        )


def find_split_files(cache_dir: Path, split: str) -> list[Path]:
    files = sorted(cache_dir.glob(f"{split}_*.pt"))
    if not files:
        raise FileNotFoundError(
            f"no {split}_*.pt files under {cache_dir}"
        )
    class_pure = [
        path for path in files
        if "_antiD_" in path.name or "_antiP_" in path.name
    ]
    if class_pure:
        raise ValueError(
            "DDP script requires mixed shards. Repack the class-pure "
            f"cache first; example file: {class_pure[0]}"
        )
    return files


def make_loaders(
    cache_dir: Path,
    batch_size: int,
    rank: int,
    world_size: int,
):
    train_files = find_split_files(cache_dir, "train")
    val_files = find_split_files(cache_dir, "val")

    train_ds = RankShardDataset(
        train_files,
        rank=rank,
        world_size=world_size,
        shuffle_files=True,
        shuffle_events=True,
    )
    val_ds = RankShardDataset(
        val_files,
        rank=rank,
        world_size=world_size,
        shuffle_files=False,
        shuffle_events=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=0
    )
    return train_loader, val_loader, train_ds, val_ds


def build_graph_features(batch):
    return torch.cat(
        [
            batch.n_hits.view(-1, 1),
            batch.total_energy.view(-1, 1),
            batch.sili_profile.view(-1, 16),
            batch.tof_profile.view(-1, 16),
            batch.tof_feat.view(-1, 11),
        ],
        dim=1,
    )


def forward_model(model, model_name: str, batch):
    graph_feat = build_graph_features(batch)
    if model_name == "gravnet_tof":
        if not hasattr(batch, "tof_paddle_energy"):
            raise ValueError(
                "GravNetTOF requires tof_paddle_energy in graph cache"
            )
        return model(
            batch.x,
            batch.edge_index,
            batch.batch,
            graph_feat=graph_feat,
            tof_paddle_energy=batch.tof_paddle_energy.view(-1, 172),
        )
    return model(
        batch.x,
        batch.edge_index,
        batch.batch,
        graph_feat=graph_feat,
    )


def reduce_metrics(
    loss_sum: float,
    correct: int,
    samples: int,
    device: torch.device,
):
    values = torch.tensor(
        [loss_sum, float(correct), float(samples)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_samples = int(values[2].item())
    return (
        values[0].item() / global_samples,
        values[1].item() / global_samples,
        global_samples,
    )


def checkpoint_state(
    epoch,
    model_name,
    dataset_tag,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    patience_counter,
    best_model_path,
):
    return {
        "epoch": epoch,
        "model_name": model_name,
        "dataset_tag": dataset_tag,
        "world_size": dist.get_world_size(),
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "best_model_path": str(best_model_path),
    }


def train(args):
    local_rank, rank, world_size, device = setup_distributed()
    try:
        if world_size != 2:
            raise ValueError(
                f"this experiment expects 2 GPUs, got world_size={world_size}"
            )

        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)

        train_loader, val_loader, train_ds, val_ds = make_loaders(
            args.split_cache_dir,
            args.batch_size,
            rank,
            world_size,
        )
        local_train = train_ds.local_len()
        local_val = val_ds.local_len()
        local_train_batches = (
            local_train + args.batch_size - 1
        ) // args.batch_size
        local_val_batches = (
            local_val + args.batch_size - 1
        ) // args.batch_size
        if args.max_train_batches is not None:
            local_train_batches = min(
                local_train_batches, args.max_train_batches
            )
        if args.max_val_batches is not None:
            local_val_batches = min(
                local_val_batches, args.max_val_batches
            )

        rank_print(rank, f"DDP backend: NCCL, world size: {world_size}")
        rank_print(
            rank,
            f"local batch size: {args.batch_size}, "
            f"global batch size: {args.batch_size * world_size}",
        )
        rank_print(
            rank,
            f"train events: {local_train * world_size:,} "
            f"({local_train:,}/rank), "
            f"steps/epoch: {local_train_batches:,}",
        )
        rank_print(
            rank,
            f"val events: {local_val * world_size:,} "
            f"({local_val:,}/rank), "
            f"steps/epoch: {local_val_batches:,}",
        )

        if args.model == "gravnet_tof":
            experiment = (
                f"GravNetTOF_{NUM_BLOCKS}b_h{HIDDEN_DIM}_"
                f"{args.dataset_tag}"
            )
            base_model = GravNetTOFClassifier(
                in_channels=IN_CHANNEL,
                hidden_dim=HIDDEN_DIM,
                graph_feat_dim=45,
                num_blocks=NUM_BLOCKS,
            ).to(device)
        else:
            experiment = (
                f"GravNet_{NUM_BLOCKS}b_h{HIDDEN_DIM}_{args.dataset_tag}"
            )
            base_model = GravNetClassifier(
                in_channels=IN_CHANNEL,
                hidden_dim=HIDDEN_DIM,
                graph_feat_dim=45,
                num_blocks=NUM_BLOCKS,
            ).to(device)

        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        criterion = FocalLoss(gamma=FOCAL_GAMMA)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(
            optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )

        rank_print(rank, f"model: {experiment}")
        rank_print(
            rank,
            f"parameters: "
            f"{sum(p.numel() for p in model.parameters()):,}",
        )

        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 1
        writer = None

        if args.resume_checkpoint is not None:
            checkpoint = torch.load(
                args.resume_checkpoint,
                map_location=device,
                weights_only=False,
            )
            if checkpoint["model_name"] != args.model:
                raise ValueError(
                    f"checkpoint model={checkpoint['model_name']}, "
                    f"requested model={args.model}"
                )
            model.module.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            best_val_loss = float(checkpoint["best_val_loss"])
            patience_counter = int(checkpoint["patience_counter"])
            start_epoch = int(checkpoint["epoch"]) + 1
            log_dir = args.resume_checkpoint.parent
            best_model_path = Path(checkpoint["best_model_path"])
            latest_checkpoint_path = args.resume_checkpoint
            if is_main(rank):
                writer = SummaryWriter(
                    log_dir=str(log_dir), purge_step=start_epoch
                )
                print(
                    f"resumed checkpoint: {args.resume_checkpoint} "
                    f"(next epoch={start_epoch})"
                )
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_root = args.result_dir or (PROJECT_ROOT / "results")
            log_dir = result_root / f"{timestamp}_{experiment}"
            best_model_path = log_dir / (
                f"{timestamp}_{experiment}_best.pth"
            )
            latest_checkpoint_path = log_dir / (
                f"{timestamp}_{experiment}_last_checkpoint.pth"
            )
            if is_main(rank):
                log_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(log_dir=str(log_dir))
        dist.barrier()

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            loss_sum = 0.0
            correct = 0
            samples = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch:3d}/{args.epochs} [train]",
                total=local_train_batches,
                leave=False,
                disable=not is_main(rank),
            )
            for batch_index, batch in enumerate(progress):
                if (
                    args.max_train_batches is not None
                    and batch_index >= args.max_train_batches
                ):
                    break
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = forward_model(model, args.model, batch)
                loss = criterion(logits, batch.y.view(-1))
                loss.backward()
                optimizer.step()

                count = batch.num_graphs
                loss_sum += loss.item() * count
                correct += (
                    logits.argmax(dim=1) == batch.y.view(-1)
                ).sum().item()
                samples += count
                if is_main(rank):
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss, train_acc, global_train_samples = reduce_metrics(
                loss_sum, correct, samples, device
            )

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_samples = 0
            with torch.no_grad():
                progress = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch:3d}/{args.epochs} [val]",
                    total=local_val_batches,
                    leave=False,
                    disable=not is_main(rank),
                )
                for batch_index, batch in enumerate(progress):
                    if (
                        args.max_val_batches is not None
                        and batch_index >= args.max_val_batches
                    ):
                        break
                    batch = batch.to(device)
                    logits = forward_model(model, args.model, batch)
                    loss = criterion(logits, batch.y.view(-1))
                    count = batch.num_graphs
                    val_loss_sum += loss.item() * count
                    val_correct += (
                        logits.argmax(dim=1) == batch.y.view(-1)
                    ).sum().item()
                    val_samples += count

            val_loss, val_acc, global_val_samples = reduce_metrics(
                val_loss_sum, val_correct, val_samples, device
            )
            scheduler.step()
            elapsed = time.time() - epoch_start

            should_stop = False
            if is_main(rank):
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.0f}s | "
                    f"train throughput="
                    f"{global_train_samples / elapsed:.0f} events/s "
                    f"(global train={global_train_samples:,}, "
                    f"val={global_val_samples:,})"
                )
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Acc/train", train_acc, epoch)
                writer.add_scalar("Acc/val", val_acc, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(
                        model.module.state_dict(), best_model_path
                    )
                    print(
                        f"  -> best model saved "
                        f"(val_loss={best_val_loss:.4f})"
                    )
                else:
                    patience_counter += 1
                    should_stop = (
                        patience_counter >= PATIENCE
                        and epoch >= MIN_EPOCHS
                    )

                torch.save(
                    checkpoint_state(
                        epoch,
                        args.model,
                        args.dataset_tag,
                        model,
                        optimizer,
                        scheduler,
                        best_val_loss,
                        patience_counter,
                        best_model_path,
                    ),
                    latest_checkpoint_path,
                )
                print(
                    f"  -> latest checkpoint saved: "
                    f"{latest_checkpoint_path}"
                )

            control = torch.tensor(
                [
                    best_val_loss if is_main(rank) else 0.0,
                    float(patience_counter) if is_main(rank) else 0.0,
                    float(should_stop) if is_main(rank) else 0.0,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(control, src=0)
            best_val_loss = control[0].item()
            patience_counter = int(control[1].item())
            should_stop = bool(control[2].item())

            if should_stop:
                rank_print(
                    rank,
                    f"  -> early stopping after {PATIENCE} "
                    "epochs without improvement",
                )
                break

        if is_main(rank):
            writer.close()
            print(f"\ntraining complete, best model: {best_model_path}")
        dist.barrier()
    finally:
        cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-cache-dir", type=Path, required=True
    )
    parser.add_argument(
        "--model",
        choices=["gravnet", "gravnet_tof"],
        default="gravnet",
    )
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="local batch size per GPU",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint", type=Path, default=None
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
