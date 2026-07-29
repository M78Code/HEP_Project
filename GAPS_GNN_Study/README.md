# GAPS GNN Study

GAPS実験における反陽子・反重陽子識別のための機械学習コードを整理したプロジェクトである。  
本リポジトリは、修士研究で用いたデータ変換、モデル学習、評価、可視化の手順を、後から確認・再実行しやすい形でまとめることを目的とする。

## 目的

- 修士論文で用いた2系列のデータと解析手順を再現できる形で整理する。
- 中上データ4Mに対するCNN+DNNとGNNの比較を再現できるようにする。
- 大場データから作成したTreeRec由来入力について、50M訓練および50Mから抽出した4M訓練の評価を再現できるようにする。
- Rejection Curve、分類スコア分布、学習曲線、β領域ごとの性能評価を再生成できるようにする。
- AI分野を専門としない後輩でも、データの出所、入力特徴量、実行コマンド、結果の対応関係を追えるようにする。

## 基本方針

このプロジェクトでは、巨大なデータファイルや学習済み重みを直接管理しない。  
データセット、学習済みモデル、評価結果は外部ストレージまたは既存の `GAPS_Project/results` に保存し、本リポジトリではそれらの場所と生成手順を記録する。

管理対象とするもの:

- データ変換スクリプト
- モデル定義
- 学習スクリプト
- 評価スクリプト
- 図作成スクリプト
- 実験設定ファイル
- データソース、入力定義、結果対応表などの文書

管理対象にしないもの:

- 大容量の `dataset/`
- `.pt`, `.pth` などの学習済み重み
- `.npy` 形式のスコア・ラベル配列
- 長大な学習ログ

最初に確認する文書:

```text
docs/data_sources.md        # データセットの由来と注意点
docs/input_definitions.md   # CNN+DNN, GNN, TreeRec入力の定義
docs/code_verification.md   # 入力特徴量と評価処理をコードから確認した記録
docs/code_map.md            # 整理版コードと元コードの対応表
docs/result_manifest.md     # 評価結果と図の保存場所
```

## ディレクトリ構成

```text
GAPS_GNN_Study/
  README.md
  requirements.txt
  configs/      # 実験ごとの設定ファイル
  docs/         # データソース、入力定義、結果対応表
  scripts/      # 実行用シェルスクリプト
  src/
    data/       # データ変換
    models/     # モデル定義
    train/      # 学習
    eval/       # 評価
    plot/       # 図作成
  results/      # 小さな集計結果や図のみを置く想定
```

## 主要なコード入口

中上データ4Mの再実行では、以下を主に用いる。

- `src/data/export_nakagami_csv_to_voxel.py`: 中上1457列CSVから学習用 `.npy` 配列を作成
- `src/train/train_nakagami_cnndnn.py`: CNN+DNNの学習
- `src/eval/evaluate_nakagami_cnndnn.py`: CNN+DNNの評価
- `src/train/train_nakagami_gravnet.py`: Sparse Voxel GNN / GravNet / DGCNNの学習
- `src/eval/evaluate_nakagami_gravnet.py`: Sparse Voxel GNN / GravNet / DGCNNの評価

大場データから作成したTreeRec由来入力の再実行では、以下を主に用いる。

- `src/train/train_treerec_gravnet.py`: TreeRec由来入力のGravNet学習
- `src/eval/evaluate_treerec_gravnet.py`: TreeRec由来入力のGravNet評価
- `src/data/build_treerec_graph.py`: TreeRecヒット情報からグラフ入力を作成

各コードと元の `GAPS_Project` 内ファイルの対応は `docs/code_map.md` に示す。


## 初回セットアップ

`GAPS_GNN_Study` をPythonパッケージとして読み込めるように、gp1などの実行環境では最初に親ディレクトリで editable install を行う。

```bash
cd ~/HEP_Project
conda activate naka
pip install -e .
```

インストール後、以下で確認できる。

```bash
cd ~/HEP_Project/GAPS_GNN_Study
python - <<'PY'
import GAPS_GNN_Study
print(GAPS_GNN_Study.__file__)
PY
```

`__init__.py` の場所が表示されれば、直接 `python -u src/...` を実行できる。なお、`scripts/*.sh` では親ディレクトリを `PYTHONPATH` に追加する処理も入れているため、editable install を忘れた場合でも実行しやすい。

## 中上データ4Mの基本実行順序

中上データ4Mの再現では、通常は以下の順に実行する。  
各スクリプトはプロジェクトルートへ移動し、`GAPS_GNN_Study` をパッケージとして読み込めるように、プロジェクトの親ディレクトリを `PYTHONPATH` に追加してから処理を行う。

```bash
# 1. CSVからnpy形式の入力データを作成
./scripts/01_export_nakagami_4M.sh

# 2. CNN+DNNを学習
./scripts/02_train_nakagami_cnndnn_4M.sh

# 3. CNN+DNNを評価
./scripts/03_eval_nakagami_cnndnn_4M.sh

# 4. GravNetを学習
./scripts/04_train_nakagami_gravnet_4M.sh

# 5. GravNetを評価
./scripts/05_eval_nakagami_gravnet_4M.sh

# 6. CNN+DNNとGravNetのRejection Curveを比較
./scripts/06_plot_nakagami_cnndnn_vs_gravnet.sh
```

GPU番号やデータ保存先を変える場合は、環境変数で上書きできる。

```bash
GPU=1 DATA_DIR=/mnt/aohba/nakagami_atrest_voxel_gnn_4M \
  ./scripts/04_train_nakagami_gravnet_4M.sh
```

## 主なデータ系列と補足評価

本プロジェクトで主に整理するデータ系列は、以下の2つである。

### 1. 中上データ4Mに基づく比較

先行研究で用いられた固定格子入力に近いデータを用い、CNN+DNNとGNNを比較する系列である。  
CNN+DNNでは固定格子化されたSi(Li)エネルギー分布とTOF関連特徴量を入力する。GNNでは同じSi(Li)エネルギー分布から非ゼロボクセルを抽出し、各ボクセルをノードとするグラフ表現に変換する。

整理版コードでは、Nakagami 1457列CSVから作成するGNN入力は `voxels.npy`、`tof_primary.npy`、`labels.npy`、`betas.npy` のみを用いる。旧スクリプトで互換性のために作成していた0埋めの `tof_paddles.npy` は、整理版では作成・使用しない。

### 2. 大場データに基づくTreeRec由来入力の評価

大場データから作成したTreeRec由来のヒット情報を用いた評価系列である。  
こちらでは、ヒット位置、エネルギー損失、時刻、検出器情報などを用いてGNN入力を構成し、50M訓練と、50Mから抽出した4M訓練の結果を比較する。また、β領域ごとの性能変化も確認する。

### 補足評価: β情報を用いた評価

真の入射βを入力に加えた場合の性能変化を確認する補足評価である。  
この評価は、測定時に直接使える入力条件とは異なるため、モデル比較の主結果ではなく、β情報が識別性能に与える影響を調べるための参考結果として扱う。

### 参考確認（主データ系列ではない）

中上修論 Fig.6.2 と一致するデータ・コード・曲線の確認結果は、再現性確認のために記録する。  
ただし、論文で主に比較した中上データ4Mとは別条件であるため、主要な再実行対象には含めない。

## 重要な注意

「同じデータセット」と「同じ入力条件」は同じ意味ではない。  
同じ事象集合を使っていても、CNN+DNNは固定格子表現、GNNは非ゼロボクセルに基づくグラフ表現を用いるため、モデルに与えられる入力表現は異なる。

実験結果を比較するときは、少なくとも以下を明記する。

- 元データの場所
- train / validation / test の事象数
- 入力に用いたファイルまたは特徴量
- βを入力に含めたかどうか
- モデル名と主要な設定
- 評価に用いたチェックポイント
- 評価結果の保存場所

## 実行環境

主な実験は以下の環境で実行した。

- OS: Ubuntu 20.04
- GPU: NVIDIA GeForce RTX 2080 Ti
- Python: 3.11
- PyTorch: 2.5.1
- PyTorch Geometric: 2.5.3
- CUDA: 12.1

詳細な環境情報は `docs/` 以下に整理する。

## 今後の整理方針

まずは修士研究で最終的に用いた実験を再現できる最小構成を整える。  
その後、不要になった試行錯誤用スクリプトやログを分離し、再現に必要なコマンド、設定、結果の対応関係を明確にする。
