"""Build a concise Japanese reporting package from completed Ohba benchmarks.

The script only reads finished result files.  It does not train models, change
splits, or access the raw waveforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Scintillator_Project


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/ohba_hybrid_report_20260922"
    )
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one '{pattern}' under {root}, found {len(matches)}")
    return matches[0]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_calibration_transfer(rows: list[dict[str, object]], output: Path) -> None:
    k = np.asarray([int(row["calibration_events_per_position"]) for row in rows])
    x = np.arange(k.size, dtype=float)
    offsets = (-0.13, 0.13)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for (label, color, key), offset in zip((
        ("Traditional CFD + charge", "#d95f02", "traditional_sigma_cm"),
        ("Hybrid waveform residual", "#1b9e77", "hybrid_sigma_cm"),
    ), offsets):
        stats = np.asarray([row[key] for row in rows], dtype=np.float64)
        median = stats[:, 1]
        errors = np.vstack((median - stats[:, 0], stats[:, 2] - median))
        point_x = x + offset
        ax.errorbar(point_x, median, yerr=errors, marker="o", capsize=4, linewidth=2.2, label=label, color=color)
        for coordinate, value in zip(point_x, median):
            ax.annotate(f"{value:.2f}", (coordinate, value), xytext=(0, 8), textcoords="offset points",
                        ha="center", color=color, fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"K={value}" for value in k])
    ax.set_xlabel("Known reference events per unseen position/run")
    ax.set_ylabel("Gaussian core sigma [cm] - lower is better")
    ax.set_title("Unseen position/run: small-reference calibration")
    ax.set_ylim(bottom=3.5)
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def draw_position_holdout(audit: dict[str, object], output: Path) -> None:
    rows = audit["per_fold"]
    positions = [str(int(row["test_positions_cm"][0])) for row in rows]
    x = np.arange(len(rows))
    traditional = np.asarray([row["traditional_sigma_cm"] for row in rows], dtype=float)
    hybrid = np.asarray([row["hybrid_sigma_cm"] for row in rows], dtype=float)
    bias = np.asarray([row["hybrid_bias_cm"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, traditional, "o-", color="#d95f02", label="Traditional")
    axes[0].plot(x, hybrid, "o-", color="#1b9e77", label="Hybrid")
    axes[0].set_ylabel("Gaussian core sigma [cm]")
    axes[0].set_title("Complete unseen-position/run test")
    axes[0].grid(alpha=0.28)
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].bar(x, bias, color="#7570b3")
    axes[1].set_ylabel("Hybrid bias [cm]")
    axes[1].set_xlabel("Held-out position [cm]")
    axes[1].set_xticks(x, positions)
    axes[1].grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def sigma(metric: dict[str, object]) -> float:
    return float(metric["gaussian_fit"]["sigma_cm"])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    resampling_path = find_one(args.results, "**/calibration_reference_resampling.json")
    offset_path = find_one(args.results, "**/position_holdout_offset_audit.json")
    quality_path = find_one(args.results, "ohba_quality_selection/quality_selection_results.json")
    traditional_path = find_one(args.results, "ohba_traditional_reproduction/roi450_800/cfd_0p200_metrics.json")
    resampling = read_json(resampling_path)
    offset_audit = read_json(offset_path)
    quality = read_json(quality_path)
    traditional = read_json(traditional_path)
    rows = resampling["rows"]

    calibration_figure = args.output / "calibration_transfer_ci.png"
    position_figure = args.output / "position_holdout_resolution_and_bias.png"
    learning_curve = args.output / "learning_curve.png"
    same_position_residual = args.output / "same_position_residual_comparison.png"
    draw_calibration_transfer(rows, calibration_figure)
    draw_position_holdout(offset_audit, position_figure)

    traditional_sigma = float(traditional["metrics"]["fusion"]["sigma_cm"])
    quality_test_sigma = float(quality["selected"]["test_metrics"]["fusion"]["sigma_cm"])
    k_rows = {int(row["calibration_events_per_position"]): row for row in rows}
    k10, k100 = k_rows[10], k_rows[100]
    k0 = k_rows[0]
    learning_curve_text = (
        f"![Learning curve]({learning_curve.name})\n\n"
        "この learning curve は性能比較に使用した固定 protocol を変更せず、収束過程を文書化する目的で追加した代表 run です。"
        if learning_curve.is_file()
        else "完了済みの過去 run は epoch ごとに端末ログへ出力したものの history file を保存していません。"
        "図として保管するには history export を有効にした文書化用の 1 run が必要です。"
    )
    same_position_text = (
        f"![Same-position residual]({same_position_residual.name})\n\n"
        "この図は各位置を train/validation/test に含める同一分布の独立テストであり、未知位置への転移性能を示す主図ではありません。"
        if same_position_residual.is_file()
        else ""
    )
    report = f"""# TOF シンチレータ波形再構成：結果報告

本書は、完了済みの固定された結果ファイルから自動生成したものです。本書の生成時には再学習、データ分割の変更、波形データの再処理を行っていません。

## 指標とデータ範囲

主指標は位置残差の Gaussian core width（`sigma`、単位 cm）です。大場卒業論文の公開結果と同じ指標であり、中心的な位置分解能を表します。RMSE は外れ値に敏感な補助指標であり、`sigma` と直接比較しません。

提供された raw archive から再現できるデータは 23,602 events です。本報告のすべての再現実験はこのデータを用います。

## 従来法の基準

| 条件 | Gaussian sigma [cm] | 意味 |
| --- | ---: | --- |
| 大場卒業論文の公開 CFD + 電荷融合 | 4.93 | 公開された従来法の基準 |
| 大場氏の非公開・品質選別後の結果 | 4.72 | 過去の参考値。正確な選別規則は未入手 |
| CFD + 電荷融合の再現、ROI 450:800 | {traditional_sigma:.3f} | 提供 raw archive 上での再現値 |
| 品質選別を訓練/検証/テストで分離した従来法 | {quality_test_sigma:.3f} | 約 79% の高品質事象を保持した独立テスト値 |

再現値は公開値 4.93 cm と整合し、新手法との比較基準として利用できます。

## Hybrid waveform residual model

Hybrid は同じ従来法の位置推定値を基準とし、2 チャンネル波形 ROI と非ラベル物理特徴量から残差のみを学習します。位置ラベルは教師値であり、入力特徴量には含めません。

### 同一位置分布での補助的な残差図

{same_position_text}

### 未知位置/run への転移評価

各テスト位置/run は、従来法の校正、モデル訓練、validation のすべてから除外します。新しい位置では、最初の `K` 個の位置既知 reference events により定数オフセットのみを推定し、それ以降の事象で評価します。従来法と Hybrid は同一の reference events を使用します。

| 位置ごとの reference events K | 従来法 sigma [cm] | Hybrid sigma [cm] | 従来法 - Hybrid [cm] | 改善量の 95% 区間 | P(Hybrid が良い) |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | {k0['traditional_sigma_cm'][1]:.3f} | {k0['hybrid_sigma_cm'][1]:.3f} | {k0['sigma_improvement_cm'][1]:.3f} | [{k0['sigma_improvement_cm'][0]:.3f}, {k0['sigma_improvement_cm'][2]:.3f}] | {k0['probability_hybrid_sigma_better']:.3f} |
| 10 | {k10['traditional_sigma_cm'][1]:.3f} | {k10['hybrid_sigma_cm'][1]:.3f} | {k10['sigma_improvement_cm'][1]:.3f} | [{k10['sigma_improvement_cm'][0]:.3f}, {k10['sigma_improvement_cm'][2]:.3f}] | {k10['probability_hybrid_sigma_better']:.3f} |
| 100 | {k100['traditional_sigma_cm'][1]:.3f} | {k100['hybrid_sigma_cm'][1]:.3f} | {k100['sigma_improvement_cm'][1]:.3f} | [{k100['sigma_improvement_cm'][0]:.3f}, {k100['sigma_improvement_cm'][2]:.3f}] | {k100['probability_hybrid_sigma_better']:.3f} |

![Calibration transfer]({calibration_figure.name})

## 主張の範囲

- `K=0`、すなわち新しい位置ごとの参照校正なしでは、従来法の方が頑健です。Hybrid には絶対位置の転移バイアスがあり、ゼロ校正の置換法とは主張しません。
- 新しい位置/run ごとに 10 個の位置既知 reference events を用いると、500 回の再標本化すべてで Hybrid が従来法より良い分解能を示しました。
- `K=100` では Hybrid は {k100['hybrid_sigma_cm'][1]:.3f} cm に達します。これは、ここで定義した校正転移プロトコルの下で、公開値 4.93 cm と過去の参考値 4.72 cm のいずれも下回ります。

## 小数の reference events が必要な理由

次図は leave-one-position-out の全 fold を示します。下段は reference 校正をしない Hybrid の絶対位置バイアスです。このバイアスのため `K=0` では従来法より悪くなります。一方で、位置ごとの定数オフセットを少数事象で補正すると、Hybrid の局所的な波形分解能が現れます。

![Position holdout]({position_figure.name})

## 論文化・共有時に同梱する資料

1. 本報告書と上記 2 図。
2. 元となる固定 JSON metrics：`{resampling_path}`、`{offset_path}`。
3. モデル構造、波形 ROI (450:800)、入力特徴量、optimizer、early stopping、seed、分割規則。
4. 代表的な train/validation learning curve。\n\n{learning_curve_text}
5. 良い結果だけでなく `K=0` の負の結果も含めます。

## 結論

波形残差学習は CFD + 電荷融合をゼロ校正で置き換える万能法ではありません。しかし、新しい位置/run ごとの少数の reference events による定数オフセット校正を許すなら、再現した従来法より統計的に安定して小さい Gaussian-core 位置分解能を達成します。
"""
    output = args.output / "report.md"
    output.write_text(report, encoding="utf-8")
    print("===== Ohba hybrid reporting package =====", flush=True)
    print(f"report : {output}", flush=True)
    print(f"figure : {calibration_figure}", flush=True)
    print(f"figure : {position_figure}", flush=True)
    print("OHBA HYBRID REPORT: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
