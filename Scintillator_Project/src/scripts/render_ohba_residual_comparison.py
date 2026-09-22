"""Render a residual histogram from saved traditional and hybrid predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Scintillator_Project.src.scripts.train_ohba_quality_selected_hybrid import plot_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="All-usable same-position held-out test")
    args = parser.parse_args()
    with np.load(args.predictions) as data:
        plot_result(
            args.output,
            data["labels"],
            data["traditional_baseline"],
            data["hybrid_prediction"],
            title=args.title,
        )
    print(f"figure : {args.output}")


if __name__ == "__main__":
    main()
