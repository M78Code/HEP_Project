#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}

FULL_EVAL=${FULL_EVAL:-"$PROJECT/results/aohba_stopping_summary_only_ablation_200k/strict_track/20260912-103024_GravNet_6b_h128_aohba_stopping_ablation_strict_track_200k_global_log_seed20260825/evaluation_test"}

NO_ENERGY_EVAL=${NO_ENERGY_EVAL:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/no_energy/20260913-223454_GravNet_6b_h128_aohba_strict_track_200k_no_energy_seed20260825/evaluation_test"}

ENERGY_ONLY_EVAL=${ENERGY_ONLY_EVAL:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/energy_only/20260914-085035_GravNet_6b_h128_aohba_strict_track_200k_energy_only_seed20260825/evaluation_test"}

OUT=${OUT:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/comparison_threeway_high_efficiency"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

for directory in \
    "$FULL_EVAL" \
    "$NO_ENERGY_EVAL" \
    "$ENERGY_ONLY_EVAL"
do
    test -f "$directory/labels.npy" || {
        echo "ERROR: missing $directory/labels.npy" >&2
        exit 1
    }
    test -f "$directory/scores.npy" || {
        echo "ERROR: missing $directory/scores.npy" >&2
        exit 1
    }
done

mkdir -p "$OUT"
export MPLBACKEND=Agg

python -u - \
    "$FULL_EVAL" \
    "$NO_ENERGY_EVAL" \
    "$ENERGY_ONLY_EVAL" \
    "$OUT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


full_dir = Path(sys.argv[1])
no_energy_dir = Path(sys.argv[2])
energy_only_dir = Path(sys.argv[3])
out_dir = Path(sys.argv[4])

out_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("Full TreeRec input", full_dir, "#2166ac"),
    ("No energy information", no_energy_dir, "#d95f02"),
    ("Hit energy only", energy_only_dir, "#1b9e77"),
]

targets = [0.90, 0.95, 0.98, 0.99, 1.00]


def load_result(directory):
    labels = np.asarray(np.load(directory / "labels.npy")).reshape(-1)
    scores = np.asarray(np.load(directory / "scores.npy"))

    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]
    scores = scores.reshape(-1)

    if labels.size != scores.size:
        raise RuntimeError(
            f"{directory}: labels/scores size mismatch "
            f"({labels.size} != {scores.size})"
        )

    if not np.all(np.isin(labels, [0, 1])):
        raise RuntimeError(f"{directory}: labels are not binary")

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{directory}: scores contain non-finite values")

    return labels.astype(np.int64), scores.astype(np.float64)


def operating_point(labels, scores, target):
    fpr, tpr, thresholds = roc_curve(
        labels, scores, drop_intermediate=False
    )

    valid = np.flatnonzero(tpr >= target)
    if valid.size == 0:
        raise RuntimeError(f"No ROC point reaches target {target}")

    minimum_fpr = np.min(fpr[valid])
    candidates = valid[np.isclose(fpr[valid], minimum_fpr)]
    index = int(candidates[0])

    n_background = int(np.count_nonzero(labels == 0))
    passed = int(round(float(fpr[index]) * n_background))

    if fpr[index] == 0:
        rejection = math.inf

        # One-sided 95% upper limit on the false-positive probability
        # for zero observed background events.
        upper_fpr_95 = 1.0 - 0.05 ** (1.0 / n_background)
        rejection_lower_95 = 1.0 / upper_fpr_95
        plot_rejection = float(n_background)
    else:
        rejection = 1.0 / float(fpr[index])
        rejection_lower_95 = None
        plot_rejection = rejection

    return {
        "target_signal_efficiency": float(target),
        "actual_signal_efficiency": float(tpr[index]),
        "antiP_passed": passed,
        "background_events": n_background,
        "fpr": float(fpr[index]),
        "rejection": None if math.isinf(rejection) else float(rejection),
        "zero_observed_background": bool(fpr[index] == 0),
        "rejection_lower_95": rejection_lower_95,
        "plot_rejection": float(plot_rejection),
        "threshold": float(thresholds[index]),
    }


loaded = []
reference_labels = None

for name, directory, color in models:
    labels, scores = load_result(directory)

    if reference_labels is None:
        reference_labels = labels
    elif not np.array_equal(reference_labels, labels):
        raise RuntimeError(
            f"{name}: label ordering differs from the full-input evaluation"
        )

    auc = float(roc_auc_score(labels, scores))
    points = [
        operating_point(labels, scores, target)
        for target in targets
    ]

    loaded.append({
        "name": name,
        "directory": str(directory),
        "color": color,
        "labels": labels,
        "scores": scores,
        "auc": auc,
        "points": points,
    })


fig = plt.figure(figsize=(18.0, 7.5), dpi=220)
grid = fig.add_gridspec(
    1, 2,
    width_ratios=[3.15, 2.85],
    wspace=0.10,
)

ax = fig.add_subplot(grid[0, 0])
table_ax = fig.add_subplot(grid[0, 1])
table_ax.axis("off")

for target in targets[:-1]:
    ax.axvline(
        target,
        color="0.55",
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
        zorder=0,
    )

for result in loaded:
    labels = result["labels"]
    scores = result["scores"]
    n_background = int(np.count_nonzero(labels == 0))

    fpr, tpr, _ = roc_curve(
        labels, scores, drop_intermediate=False
    )

    plotted_rejection = 1.0 / np.maximum(
        fpr,
        1.0 / n_background,
    )

    ax.step(
        tpr,
        plotted_rejection,
        where="post",
        linewidth=2.5,
        color=result["color"],
        label=f"{result['name']}  AUC={result['auc']:.6f}",
    )

    for point in result["points"]:
        ax.plot(
            point["actual_signal_efficiency"],
            point["plot_rejection"],
            marker="o",
            markersize=5.5,
            color=result["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )

ax.set_xlim(0.90, 1.0005)
ax.set_ylim(1.0, 2.0e4)
ax.set_yscale("log")
ax.set_xticks(targets)
ax.set_xticklabels(
    ["0.90", "0.95", "0.98", "0.99", "1.00"]
)
ax.set_xlabel("Signal efficiency (antiD)", fontsize=13)
ax.set_ylabel(
    "Background rejection (1 / antiP efficiency)",
    fontsize=13,
)
ax.grid(
    True,
    which="both",
    linestyle=":",
    linewidth=0.8,
    alpha=0.55,
)
ax.legend(
    loc="lower left",
    fontsize=10,
    framealpha=0.94,
)


def rejection_cell(point):
    if point["zero_observed_background"]:
        lower = point["rejection_lower_95"]
        return f">{lower:,.0f}*\n(0)"
    return (
        f"{point['rejection']:,.1f}\n"
        f"({point['antiP_passed']:,})"
    )


columns = [
    "Input",
    "AUC",
    "0.90",
    "0.95",
    "0.98",
    "0.99",
    "1.00",
]

cell_rows = []
for result in loaded:
    row = [
        result["name"],
        f"{result['auc']:.6f}",
    ]
    row.extend(rejection_cell(point) for point in result["points"])
    cell_rows.append(row)

table_ax.text(
    0.5,
    0.96,
    "High-efficiency operating points",
    ha="center",
    va="top",
    fontsize=18,
    fontweight="bold",
)

table_ax.text(
    0.5,
    0.895,
    "Each cell: rejection  (antiP passed)",
    ha="center",
    va="top",
    fontsize=11,
    color="0.30",
)

table = table_ax.table(
    cellText=cell_rows,
    colLabels=columns,
    cellLoc="center",
    colLoc="center",
    bbox=[0.00, 0.38, 1.00, 0.46],
    colWidths=[
        0.225,
        0.105,
        0.134,
        0.134,
        0.134,
        0.134,
        0.134,
    ],
)

table.auto_set_font_size(False)
table.set_fontsize(9.2)
table.scale(1.0, 1.75)

for column in range(len(columns)):
    header = table[(0, column)]
    header.set_facecolor("#1f6fae")
    header.set_text_props(color="white", fontweight="bold")

for row_index, result in enumerate(loaded, start=1):
    table[(row_index, 0)].set_text_props(
        color=result["color"],
        fontweight="bold",
    )

table_ax.text(
    0.01,
    0.29,
    "Test sample: 10,000 antiD and 10,000 antiP for each input.",
    ha="left",
    va="top",
    fontsize=10.5,
)

table_ax.text(
    0.01,
    0.225,
    "Zero false-positive points are plotted at the finite-sample cap",
    ha="left",
    va="top",
    fontsize=10.5,
)

table_ax.text(
    0.01,
    0.18,
    "(N background = 10,000). They do not imply infinite rejection.",
    ha="left",
    va="top",
    fontsize=10.5,
)

table_ax.text(
    0.01,
    0.11,
    "* Zero antiP passed. Value is the one-sided 95% confidence",
    ha="left",
    va="top",
    fontsize=9.8,
    color="0.35",
)

table_ax.text(
    0.01,
    0.065,
    "lower bound on background rejection.",
    ha="left",
    va="top",
    fontsize=9.8,
    color="0.35",
)

fig.suptitle(
    "Strict-track-stop TreeRec 200K: input-feature ablation",
    fontsize=20,
    fontweight="bold",
    y=0.975,
)

fig.subplots_adjust(
    left=0.07,
    right=0.985,
    bottom=0.12,
    top=0.88,
)

figure_path = out_dir / "rejection_threeway_high_efficiency.png"
fig.savefig(
    figure_path,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)


json_rows = []
csv_rows = []

for result in loaded:
    json_row = {
        "input": result["name"],
        "evaluation_directory": result["directory"],
        "events": int(result["labels"].size),
        "antiP": int(np.count_nonzero(result["labels"] == 0)),
        "antiD": int(np.count_nonzero(result["labels"] == 1)),
        "auc": result["auc"],
        "operating_points": result["points"],
    }
    json_rows.append(json_row)

    for point in result["points"]:
        csv_rows.append({
            "input": result["name"],
            "auc": f"{result['auc']:.9f}",
            "target_signal_efficiency":
                point["target_signal_efficiency"],
            "actual_signal_efficiency":
                point["actual_signal_efficiency"],
            "antiP_passed": point["antiP_passed"],
            "background_events": point["background_events"],
            "fpr": point["fpr"],
            "rejection":
                "inf"
                if point["rejection"] is None
                else point["rejection"],
            "rejection_lower_95":
                ""
                if point["rejection_lower_95"] is None
                else point["rejection_lower_95"],
        })

(out_dir / "metrics_threeway.json").write_text(
    json.dumps(json_rows, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with (out_dir / "operating_points.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "input",
            "auc",
            "target_signal_efficiency",
            "actual_signal_efficiency",
            "antiP_passed",
            "background_events",
            "fpr",
            "rejection",
            "rejection_lower_95",
        ],
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print("===== three-way comparison =====")
for result in loaded:
    print()
    print(f"{result['name']}: AUC={result['auc']:.6f}")
    for point in result["points"]:
        if point["zero_observed_background"]:
            value = (
                "0 antiP passed; "
                f"95% lower bound={point['rejection_lower_95']:.1f}"
            )
        else:
            value = (
                f"rejection={point['rejection']:.3f}; "
                f"antiP passed={point['antiP_passed']}"
            )
        print(
            f"  signal efficiency "
            f"{point['target_signal_efficiency']:.2f}: {value}"
        )

print()
print("figure :", figure_path)
print("table  :", out_dir / "operating_points.csv")
print("metrics:", out_dir / "metrics_threeway.json")
print("THREE-WAY HIGH-EFFICIENCY COMPARISON: COMPLETE")
PY

echo
echo "===== generated files ====="
ls -lh "$OUT"
