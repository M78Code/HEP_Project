from pathlib import Path
import os
import json
import pickle
import uproot
import numpy as np
import GAPS_Project


# 项目根目录（GAPS_Project/），所有路径基于此
PROJECT_ROOT = Path(GAPS_Project.__file__).parent

f = uproot.open(PROJECT_ROOT / 'dataset' / 'tar_root' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.root')
"""

"""


def get_root_key():
    """
    === 文件结构 ===
    ['GGeometry;1', 'TreeMc;1', 'TreeRec;1', 'SimulationParameterTree;1']

    'GGeometry;1'                ← 探测器几何结构，uproot无法读取，也不需要（是GAPS探测器的3D几何描述，uproot不支持这种复杂类型，但我们完全不需要它）
    'TreeMc;1'                   ← Monte Carlo模拟数据（真实标签）
    'TreeRec;1'                  ← 重建数据（实际测量）
    'SimulationParameterTree;1'  ← 模拟参数
    :return:
    """
    print('=== 文件结构 ===')
    print(f.keys())
    # for key in f.keys():
    #     obj = f[key]
    #     print(f'\n[{key}] type: {type(obj).__name__}')
    #     if hasattr(obj, 'keys'):
    #         print('  branches:', obj.keys())


def look_TreeMc_TreeRec_SimulationParameterTree():
    """
    查看 TreeMc，TreeRec，SimulationParameterTree
    === TreeMc branches ===
    ['Mc', 'Mc/CEventBase', 'Mc/CEventBase/TObject', 'Mc/CEventBase/TObject/fUniqueID', 'Mc/CEventBase/TObject/fBits', 'Mc/CEventBase/runNumber_', 'Mc/CEventBase/subRunNumber_', 'Mc/CEventBase/eventNumber_', 'Mc/CEventBase/eventTime_', 'Mc/CEventBase/eventId_', 'Mc/CEventBase/primaryBetaGenerated_', 'Mc/CEventBase/primaryMomentumDirectionGenerated_', 'Mc/CEventBase/primaryKineticEnergyGenerated_', 'Mc/primaryPosition_', 'Mc/primaryTime_', 'Mc/primaryPdg_', 'Mc/primaryStoppingKineticEnergy_', 'Mc/randomSeed_', 'Mc/primaryStoppingVolume_', 'Mc/primaryStoppingPosition_', 'Mc/primaryStoppingTime_', 'Mc/hitTrackIndex_', 'Mc/tracks_', 'Mc/totalEnergyDeposition_', 'Mc/meanPosition_', 'Mc/volumeId_', 'Mc/time_']

    === TreeRec branches ===
    ['Rec', 'Rec/CEventBase',
    'Rec/CEventBase/TObject', 'Rec/CEventBase/TObject/fUniqueID', 'Rec/CEventBase/TObject/fBits', 'Rec/CEventBase/runNumber_', 'Rec/CEventBase/subRunNumber_', 'Rec/CEventBase/eventNumber_', 'Rec/CEventBase/eventTime_', 'Rec/CEventBase/eventId_', 'Rec/CEventBase/primaryBetaGenerated_', 'Rec/CEventBase/primaryMomentumDirectionGenerated_', 'Rec/CEventBase/primaryKineticEnergyGenerated_', 'Rec/trigger_sources', 'Rec/trigger_vids', 'Rec/activeReco_', 'Rec/event_quality', 'Rec/primaryStoppingPosition_', 'Rec/primaryStoppingPosition_/primaryStoppingPosition_.first', 'Rec/primaryStoppingPosition_/primaryStoppingPosition_.second', 'Rec/primaryStoppingVolume_', 'Rec/primaryStoppingVolume_/primaryStoppingVolume_.first', 'Rec/primaryStoppingVolume_/primaryStoppingVolume_.second', 'Rec/primaryStoppingTime_', 'Rec/primaryStoppingTime_/primaryStoppingTime_.first', 'Rec/primaryStoppingTime_/primaryStoppingTime_.second', 'Rec/primaryBeta_', 'Rec/primaryBeta_/primaryBeta_.first', 'Rec/primaryBeta_/primaryBeta_.second', 'Rec/primaryBetaError_', 'Rec/primaryBetaError_/primaryBetaError_.first', 'Rec/primaryBetaError_/primaryBetaError_.second', 'Rec/primaryMomentumDirection_', 'Rec/primaryMomentumDirection_/primaryMomentumDirection_.first', 'Rec/primaryMomentumDirection_/primaryMomentumDirection_.second', 'Rec/primaryEnergyDepositions_', 'Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.first', 'Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.second', 'Rec/HitTrackIndex', 'Rec/HitTrackIndex/HitTrackIndex.first', 'Rec/HitTrackIndex/HitTrackIndex.second', 'Rec/Chi2', 'Rec/Chi2/Chi2.first', 'Rec/Chi2/Chi2.second', 'Rec/Ndof', 'Rec/Ndof/Ndof.first', 'Rec/Ndof/Ndof.second', 'Rec/ParCov', 'Rec/ParCov/ParCov.first', 'Rec/ParCov/ParCov.second', 'Rec/FitStatus', 'Rec/FitStatus/FitStatus.first', 'Rec/FitStatus/FitStatus.second', 'Rec/SdFitPar', 'Rec/SdFitPar/SdFitPar.first', 'Rec/SdFitPar/SdFitPar.second', 'Rec/SdFitErr', 'Rec/SdFitErr/SdFitErr.first', 'Rec/SdFitErr/SdFitErr.second', 'Rec/SdFitChi2', 'Rec/SdFitChi2/SdFitChi2.first', 'Rec/SdFitChi2/SdFitChi2.second', 'Rec/SdFitNdof', 'Rec/SdFitNdof/SdFitNdof.first', 'Rec/SdFitNdof/SdFitNdof.second', 'Rec/hitseries_', 'Rec/hitseries_/hitseries_.fUniqueID', 'Rec/hitseries_/hitseries_.fBits', 'Rec/hitseries_/hitseries_.volume_id_', 'Rec/hitseries_/hitseries_.energydep_', 'Rec/hitseries_/hitseries_.hit_position_', 'Rec/hitseries_/hitseries_.hit_time_', 'Rec/hitseries_/hitseries_.index_', 'Rec/Tracks', 'Rec/Tracks/Tracks.first', 'Rec/Tracks/Tracks.second', 'Rec/registeredRecos_', 'Rec/PacketType']

    === SimulationParameterTree branches ===
    ['SimulationParameter', 'SimulationParameter/TObject', 'SimulationParameter/TObject/fUniqueID', 'SimulationParameter/TObject/fBits', 'SimulationParameter/runId_', 'SimulationParameter/subRunId_', 'SimulationParameter/productionHostName_', 'SimulationParameter/G4version_', 'SimulationParameter/physicsList_', 'SimulationParameter/CraneVersion_', 'SimulationParameter/randomSeed_', 'SimulationParameter/triggerLevel_', 'SimulationParameter/UseTrigger', 'SimulationParameter/UseDigitizer', 'SimulationParameter/HPrimaryBetaGenerated', 'SimulationParameter/HPrimaryCosZenBetaGenerated', 'SimulationParameter/HPrimaryCosZenLogEGenerated', 'SimulationParameter/HPrimaryBetaTriggered', 'SimulationParameter/HPrimaryCosZenBetaTriggered', 'SimulationParameter/HPrimaryCosZenLogETriggered', 'SimulationParameter/UniformBeta', 'SimulationParameter/PrimaryBetaLow', 'SimulationParameter/PrimaryBetaHigh', 'SimulationParameter/PrimaryIsotropic', 'SimulationParameter/PrimaryIsotropicUpDownAll', 'SimulationParameter/PrimaryLimit', 'SimulationParameter/PrimaryFile', 'SimulationParameter/PrimaryFileName', 'SimulationParameter/UniformLogE', 'SimulationParameter/PrimaryLogELow', 'SimulationParameter/PrimaryLogEHigh', 'SimulationParameter/PrimaryPdg', 'SimulationParameter/nEventsGeneratedPerFile_', 'SimulationParameter/VolumeIdAliveMap', 'SimulationParameter/VolumeIdAliveMap/VolumeIdAliveMap.first', 'SimulationParameter/VolumeIdAliveMap/VolumeIdAliveMap.second']

    ⏺ 结构非常清晰，对ML来说关键信息如下：

      ---
      TreeMc（模拟真值）— 提供标签
      - Mc/primaryPdg_ — 粒子种类标签（反质子 vs 反重氘核的PDG编号）
      - Mc/CEventBase/primaryBetaGenerated_ — 真实速度β
      - Mc/totalEnergyDeposition_ — 总能量沉积
      - Mc/volumeId_ — 命中的探测器体积ID

      TreeRec（重建数据）— 提供输入特征
      - Rec/hitseries_/hitseries_.volume_id_ — 命中的探测器ID
      - Rec/hitseries_/hitseries_.energydep_ — 各hit的能量沉积
      - Rec/hitseries_/hitseries_.hit_position_ — hit的空间位置
      - Rec/hitseries_/hitseries_.hit_time_ — hit的时间
      - Rec/primaryBeta_ — 重建速度β
      - Rec/event_quality — 事件质量标志

      ---
      结构非常符合GNN的设计：
      每个event = 一张图，节点 = 各个detector hit（volume_id + energy + position + time），边 = hit之间的连接关系。
    :return:
    """

    # 查看 TreeMc
    print('=== TreeMc branches ===')
    tree_mc = f['TreeMc;1']
    print(tree_mc.keys())

    # 查看 TreeRec
    print('\n=== TreeRec branches ===')
    tree_rec = f['TreeRec;1']
    print(tree_rec.keys())

    # 查看 SimulationParameterTree
    print('\n=== SimulationParameterTree branches ===')
    tree_sim = f['SimulationParameterTree;1']
    print(tree_sim.keys())


def check_event():
    tree_mc = f['TreeMc;1']
    tree_rec = f['TreeRec;1']

    # 1. 总event数
    print(f'\n总event数：{tree_mc.num_entries}')

    # 2. 粒子标签
    pdg = tree_mc['Mc/primaryPdg_'].array()
    print(f'\nprimaryPdg_: {pdg[:10]}')
    print(f'唯一值: {np.unique(pdg)}')

    # 3. 速度β
    beta = np.array(tree_mc['Mc/CEventBase/primaryBetaGenerated_'].array())
    print(f'\nbeta范围: {beta.min():.4f} ~ {beta.max():.4f}')

    # 4. 每个event的hit数
    hit_energy = tree_rec['Rec/hitseries_/hitseries_.energydep_'].array()
    hit_counts = [len(x) for x in hit_energy[:20]]
    print(f'\n前20个event的hit数: {hit_counts}')
    print(f'平均hit数: {np.mean(hit_counts):.1f}')


def check_node_feature():
    """
    ============================= test session starts ==============================
    collecting ... collected 2 items

    root_file_reader.py::test_event PASSED                                   [ 50%]
    总event数：57

    primaryPdg_: [-1000010020, -1000010020, -1000010020, ..., -1000010020, -1000010020]
    唯一值: [-1000010020]

    beta范围: 0.1257 ~ 0.5915

    前20个event的hit数: [30, 15, 22, 54, 59, 42, 34, 29, 35, 36, 12, 45, 25, 36, 11, 34, 38, 19, 19, 35]
    平均hit数: 31.5

    root_file_reader.py::test_node_feature PASSED                            [100%]=== Event 0 ===
    hit数: 30
    volume_id: [200200101, 201150200, 202150004, ..., 100450000, 115002000, 105552000]
    energy: [3.08, 1.07, 14.4, 1.01, 1.99, 1.45, ..., 1.72, 2.11, 6.16, 1.64, 2.56, 2.14]
    position: [{fX: 39.8, fY: -63, fZ: -104}, {...}, ..., {fX: -335, fY: -1.06e+03, fZ: -497}]
    time: [nan, nan, nan, nan, nan, nan, nan, ..., 34.2, 31.7, 37.4, 36.9, 34.5, 35.5]

    数据情况清楚了，分析如下：
    节点特征可用字段：
    - position：三维坐标 (fX, fY, fZ)，范围约 -1000 ~ +100
    - energy：能量沉积，1~14 MeV量级
    - time：前几个hit是NaN，后面才有值（31~37 ns）

    需要注意的问题：
    1. time有大量NaN — Si(Li)半导体探测器没有时间测量，只有TOF闪烁体计数器有。需要决定是填0还是加一个mask标志位
    2. volume_id是整数编码（如200200101）— 编码规则不明，等大場さん说明后再决定是否用作特征

    ---
    下一步设计节点特征向量：
    每个节点 = [fX, fY, fZ, energy, time]，共5维，time的NaN填0。
    边的构建方式：用k近邻（k-NN），基于空间距离连接最近的k个hit。
    :return:
    """
    tree_rec = f['TreeRec;1']
    # 读取第一个event的各字段
    volume_id = tree_rec['Rec/hitseries_/hitseries_.volume_id_'].array()
    energy = tree_rec['Rec/hitseries_/hitseries_.energydep_'].array()
    position = tree_rec['Rec/hitseries_/hitseries_.hit_position_'].array()
    time = tree_rec['Rec/hitseries_/hitseries_.hit_time_'].array()

    print('\n\n=== Event 0 ===')
    print(f'hit数: {len(volume_id[0])}')
    print(f'volume_id: {volume_id[0]}')
    print(f'energy: {energy[0]}')
    print(f'position: {position[0]}')
    print(f'time: {time[0]}')


# Step 2.0: ROOT文件 → pickle + summary.json
def convert_root_to_pickle(root_path: Path, output_dir: Path):
    """
    将ROOT文件转换为pickle（训练用）+ summary.json（可视化查看）
    :param root_path:    ROOT文件路径
    :param output_dir:   输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = root_path.stem
    pkl_path = output_dir / f"{base_name}.pkl"
    summary_path = output_dir / f"{base_name}_summary.json"

    events = []

    with uproot.open(root_path) as file:
        tree_rec = file["TreeRec;1"]
        tree_mc = file["TreeMc;1"]

        all_energies = tree_rec["Rec/hitseries_/hitseries_.energydep_"].array()
        all_pos = tree_rec["Rec/hitseries_/hitseries_.hit_position_"].array()
        all_times = tree_rec["Rec/hitseries_/hitseries_.hit_time_"].array()
        all_labels = tree_mc["Mc/primaryPdg_"].array()
        all_betas = tree_mc["Mc/CEventBase/primaryBetaGenerated_"].array()

        for idx in range(len(all_labels)):
            energies = np.array(all_energies[idx], dtype=np.float32)
            pos_raw = all_pos[idx]
            times = np.array(all_times[idx], dtype=np.float32)
            label = int(all_labels[idx])
            beta = float(all_betas[idx])

            positions = np.stack([
                np.array(pos_raw["fX"], dtype=np.float32),
                np.array(pos_raw["fY"], dtype=np.float32),
                np.array(pos_raw["fZ"], dtype=np.float32),
            ], axis=1)  # (N, 3)

            events.append({
                "energy": energies,
                "positions": positions,
                "times": times,
                "label": label,
                "beta": beta,
                "n_hits": len(energies),
            })

    # ── 保存pickle ─────────────────────────────────────
    with open(pkl_path, "wb") as out:
        pickle.dump({"events": events, "source_file": root_path.name}, out)
    print(f"pickle saved → {pkl_path}")

    # ── 生成summary.json ───────────────────────────────
    n_hits_list = [e["n_hits"] for e in events]
    betas = [e["beta"] for e in events]
    label_counts = {}
    for e in events:
        key = str(e["label"])
        label_counts[key] = label_counts.get(key, 0) + 1

    sample = events[0]
    summary = {
        "source_file": root_path.name,
        "total_events": len(events),
        "label_counts": label_counts,
        "hit_stats": {
            "min": int(min(n_hits_list)),
            "max": int(max(n_hits_list)),
            "mean": round(float(np.mean(n_hits_list)), 2),
        },
        "beta_stats": {
            "min": round(float(min(betas)), 4),
            "max": round(float(max(betas)), 4),
            "mean": round(float(np.mean(betas)), 4),
        },
        "feature_dims": 5,
        "sample_event_0": {
            "n_hits": sample["n_hits"],
            "label": sample["label"],
            "beta": round(sample["beta"], 4),
            "energy_first5": [round(float(v), 3) for v in sample["energy"][:5]],
            "positions_first3": sample["positions"][:3].tolist(),
            "times_first5": [
                None if np.isnan(v) else round(float(v), 3)
                for v in sample["times"][:5]
            ],
        },
    }

    with open(summary_path, "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
    print(f"summary saved → {summary_path}")


if __name__ == '__main__':
    # get_root_key()
    # look_TreeMc_TreeRec_SimulationParameterTree()
    # check_event()
    # check_node_feature()
    convert_root_to_pickle(
        root_path = PROJECT_ROOT / 'dataset' / 'tar_root' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.root',
        output_dir = PROJECT_ROOT / 'dataset' / 'processed'
    )

"""
============================= test session starts ==============================
collecting ... collected 1 item

root_file_reader.py::test_event PASSED                                   [100%]
总event数：57

primaryPdg_: [-1000010020, -1000010020, -1000010020, ..., -1000010020, -1000010020]
唯一值: [-1000010020]

beta范围: 0.1257 ~ 0.5915

前20个event的hit数: [30, 15, 22, 54, 59, 42, 34, 29, 35, 36, 12, 45, 25, 36, 11, 34, 38, 19, 19, 35]
平均hit数: 31.5

 数据情况汇总：

  ┌─────────────────┬───────────────────────────────┐
  │      项目       │              值               │
  ├─────────────────┼───────────────────────────────┤
  │ event数         │ 57（测试数据）                │
  ├─────────────────┼───────────────────────────────┤
  │ 粒子种类        │ 仅反重氘核（PDG=-1000010020） │
  ├─────────────────┼───────────────────────────────┤
  │ β范围           │ 0.13 ~ 0.59                   │
  ├─────────────────┼───────────────────────────────┤
  │ 平均hit数/event │ 31.5个                        │
  ├─────────────────┼───────────────────────────────┤
  │ hit数范围       │ 11 ~ 59个                     │
  └─────────────────┴───────────────────────────────┘

  ---
  对GNN设计的启示：
  - 每个event = 平均31.5个节点的图
  - 节点数11~59可变 → GNN天然适合处理可变大小的图

  ---
  现在学习用数据还没到，但可以用这57个event提前做准备：
  1. 设计图构建代码（节点特征、边的连接方式）
  2. 写好ROOT→PyG Data对象的转换流程

  等大場さん的数据一到就能直接开始训练。要继续推进吗？

❯ 项目这几个值说明什么，是什么意思，先解释下

⏺ event数：57
  一个event = 一次粒子打入探测器的完整记录。57个就是57次反重氘核打入GAPS探测器的模拟实验，量很少，是测试用的。

  ---
  粒子种类：仅反重氘核（PDG=-1000010020）
  PDG是粒子物理界的粒子编号标准。这个文件只有反重氘核，没有反质子。真正训练时需要两种粒子的数据才能做分类。

  ---
  β范围：0.13 ~ 0.59
  β = v/c，粒子速度与光速之比。0.13表示速度是光速的13%，0.59是59%。这个范围内的粒子是GAPS实验关注的低能宇宙射线。β也是识别粒子种类的重
  要特征之一。

  ---
  平均hit数/event：31.5个，范围11~59
  一个粒子穿过探测器时，会在不同位置的探测单元上留下信号，每个信号叫一个hit。平均一次事件会触发31.5个探测单元，最少11个，最多59个。

  这就是GNN的输入——每个hit =
  图中的一个节点，节点特征包括位置、能量、时间等。hit数量不固定（11~59），正好是GNN擅长处理的可变大小图结构。

❯ 为什么通过PDG=-1000010020得出是反重氘核，而不是反质子

⏺ PDG编号是有规律的，不是随意分配的。

  ---
  普通粒子的PDG编号规则：
  - 质子 = 2212
  - 中子 = 2112
  - 氘核（1个质子+1个中子）= 1000010020

  编号格式：10LZZZAAAI
  - ZZZ = 质子数（原子序数）
  - AAA = 质量数（质子+中子）
  - 氘核：Z=1（1个质子），A=2（质量数2）→ 1000010020

  ---
  反粒子的规则：取负号
  - 反质子 = -2212
  - 反氘核 = -1000010020

  ---
  所以 -1000010020 直接对应：
  - 负号 → 反粒子
  - 1000010020 → 氘核（Z=1, A=2）

  → 反氘核，不是反质子（反质子是-2212，数字完全不同）
  
  ❯ 从哪里可以查到这种命名规则

⏺ 官方来源：PDG（Particle Data Group）

  网址：pdg.lbl.gov

  ---
  具体查询方式：

  ① 粒子编号表
  PDG每年发布《Review of Particle Physics》，里面有完整的粒子编号列表，搜索：
  PDG Monte Carlo numbering scheme

  ② 直接查单个粒子
  在 pdg.lbl.gov 搜索粒子名称，页面上会显示对应的PDG ID。

  ③ Python直接查
  import particle
  p = particle.Particle.from_pdgid(-1000010020)
  print(p.name)  # 输出粒子名称

  安装：pip install particle

  ---
  这个编号规则在高能物理界是通用标准，GEANT4、ROOT、各种模拟软件都遵循同一套规则。
"""
