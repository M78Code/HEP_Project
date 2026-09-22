"""Render a documented hybrid training curve from a saved history.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = json.loads(args.history.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("history is empty")
    epoch = np.asarray([row["epoch"] for row in rows])
    train_loss = np.asarray([row["train_loss"] for row in rows])
    val_rmse = np.asarray([row["validation_rmse_cm"] for row in rows])
    best = int(np.argmin(val_rmse))
    output = args.output or args.history.with_name("learning_curve.png")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(epoch, train_loss, color="#1b9e77")
    axes[0].set_xlabel("エポック")
    axes[0].set_ylabel("学習損失")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epoch, val_rmse, color="#7570b3")
    axes[1].scatter([epoch[best]], [val_rmse[best]], color="#d95f02", zorder=3, label="最良チェックポイント")
    axes[1].set_xlabel("エポック")
    axes[1].set_ylabel("検証RMSE [cm]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("ハイブリッド波形残差学習の収束")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"figure : {output}")


if __name__ == "__main__":
    main()
