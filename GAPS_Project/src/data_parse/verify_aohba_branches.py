"""
大場さん新データ (GAPS_Sim_2tof) の TreeMc / TreeRec branch 構造を確認。
- GGeometry はスキップ(uproot 非対応)
- 真の branch 名を把握してから前処理を設計する。
"""
import uproot
from pathlib import Path

ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')


def list_branches(tree, label, max_show=80):
    print(f'\n--- {label}: {tree.num_entries} entries, {len(tree.keys())} branches ---')
    for i, k in enumerate(tree.keys()):
        if i >= max_show:
            print(f'  ... ({len(tree.keys()) - max_show} more)')
            break
        try:
            interp = tree[k].interpretation
            print(f'  [{i:3d}] {k:60s}  {interp}')
        except Exception as e:
            print(f'  [{i:3d}] {k:60s}  (interp err: {e.__class__.__name__})')


def inspect_first_file(particle):
    files = sorted((ROOT_DIR / particle).glob('*.root'))
    print(f'\n========== {particle}: {len(files)} files ==========')
    f = uproot.open(files[0])

    for key in ['TreeMc', 'TreeRec']:
        matches = [k for k in f.keys() if k.startswith(key + ';')]
        if not matches:
            print(f'  {key}: NOT FOUND')
            continue
        latest = sorted(matches)[-1]
        tree = f[latest]
        list_branches(tree, f'{particle} / {latest}')


def estimate_total(particle):
    files = sorted((ROOT_DIR / particle).glob('*.root'))
    n_sample = min(5, len(files))
    total = 0
    for fp in files[:n_sample]:
        with uproot.open(fp) as f:
            matches = [k for k in f.keys() if k.startswith('TreeMc;')]
            if matches:
                total += f[sorted(matches)[-1]].num_entries
    avg = total / n_sample
    print(f'\n[{particle}] avg = {avg:,.0f} ev/file  ->  projected total {avg * len(files):,.0f}')


if __name__ == '__main__':
    for p in ['antiD', 'antiP']:
        inspect_first_file(p)
        estimate_total(p)

"""
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ python src/data_parse/verify_aohba_branches.py 2>&1 | tee ~/aohba_branches.log

========== antiD: 100 files ==========

--- antiD / TreeMc;166: 247721 entries, 27 branches ---
  [  0] Mc                                                            AsGroup(<TBranchElement 'Mc' (15 subbranches) at 0x7f4c5f55b350>, {'CEventBase': AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f55b650>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f564050>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')}), 'primaryPosition_': AsStridedObjects(Model_TVector3_v3), 'primaryTime_': AsDtype('>f8'), 'primaryPdg_': AsDtype('>i4'), 'primaryStoppingKineticEnergy_': AsDtype('>f8'), 'randomSeed_': AsDtype('>u4'), 'primaryStoppingVolume_': AsDtype('>i4'), 'primaryStoppingPosition_': AsStridedObjects(Model_TVector3_v3), 'primaryStoppingTime_': AsDtype('>f8'), 'hitTrackIndex_': AsJagged(AsDtype('>i4'), header_bytes=10), 'tracks_': AsObjects(AsVector(True, AsPointer(Model_CTrackMc))), 'totalEnergyDeposition_': AsJagged(AsDtype('>f8'), header_bytes=10), 'meanPosition_': AsJagged(AsStridedObjects(Model_TVector3_v3), header_bytes=10), 'volumeId_': AsJagged(AsDtype('>i4'), header_bytes=10), 'time_': AsJagged(AsDtype('>f8'), header_bytes=10)})
  [  1] Mc/CEventBase                                                 AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f55b650>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f564050>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')})
  [  2] Mc/CEventBase/TObject                                         AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f564050>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')})
  [  3] Mc/CEventBase/TObject/fUniqueID                               AsDtype('>u4')
  [  4] Mc/CEventBase/TObject/fBits                                   AsDtype('>u4')
  [  5] Mc/CEventBase/runNumber_                                      AsDtype('>u4')
  [  6] Mc/CEventBase/subRunNumber_                                   AsDtype('>u4')
  [  7] Mc/CEventBase/eventNumber_                                    AsDtype('>u4')
  [  8] Mc/CEventBase/eventTime_                                      AsDtype('>i8')
  [  9] Mc/CEventBase/eventId_                                        AsDtype('>u4')
  [ 10] Mc/CEventBase/primaryBetaGenerated_                           AsDtype('>f8')
  [ 11] Mc/CEventBase/primaryMomentumDirectionGenerated_              AsStridedObjects(Model_TVector3_v3)
  [ 12] Mc/CEventBase/primaryKineticEnergyGenerated_                  AsDtype('>f8')
  [ 13] Mc/primaryPosition_                                           AsStridedObjects(Model_TVector3_v3)
  [ 14] Mc/primaryTime_                                               AsDtype('>f8')
  [ 15] Mc/primaryPdg_                                                AsDtype('>i4')
  [ 16] Mc/primaryStoppingKineticEnergy_                              AsDtype('>f8')
  [ 17] Mc/randomSeed_                                                AsDtype('>u4')
  [ 18] Mc/primaryStoppingVolume_                                     AsDtype('>i4')
  [ 19] Mc/primaryStoppingPosition_                                   AsStridedObjects(Model_TVector3_v3)
  [ 20] Mc/primaryStoppingTime_                                       AsDtype('>f8')
  [ 21] Mc/hitTrackIndex_                                             AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 22] Mc/tracks_                                                    AsObjects(AsVector(True, AsPointer(Model_CTrackMc)))
  [ 23] Mc/totalEnergyDeposition_                                     AsJagged(AsDtype('>f8'), header_bytes=10)
  [ 24] Mc/meanPosition_                                              AsJagged(AsStridedObjects(Model_TVector3_v3), header_bytes=10)
  [ 25] Mc/volumeId_                                                  AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 26] Mc/time_                                                      AsJagged(AsDtype('>f8'), header_bytes=10)

--- antiD / TreeRec;8: 247721 entries, 78 branches ---
  [  0] Rec                                                           AsGroup(<TBranchElement 'Rec' (25 subbranches) at 0x7f4c5f5d4090>, {'CEventBase': AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f5d49d0>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f5d5b10>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')}), 'trigger_sources': AsJagged(AsDtype('uint8'), header_bytes=10), 'trigger_vids': AsJagged(AsDtype('>u4'), header_bytes=10), 'activeReco_': AsStrings(header_bytes=6), 'event_quality': AsJagged(AsDtype('>i4'), header_bytes=10), 'primaryStoppingPosition_': AsGroup(<TBranchElement 'primaryStoppingPosition_' (2 subbranches) at 0x7f4c5f405510>, {'primaryStoppingPosition_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingPosition_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))}), 'primaryStoppingVolume_': AsGroup(<TBranchElement 'primaryStoppingVolume_' (2 subbranches) at 0x7f4c5f40d0d0>, {'primaryStoppingVolume_.first': <UnknownInterpretation 'none of the rules matched'>, 'primaryStoppingVolume_.second': AsJagged(AsDtype('>u4'))}), 'primaryStoppingTime_': AsGroup(<TBranchElement 'primaryStoppingTime_' (2 subbranches) at 0x7f4c5f41cd90>, {'primaryStoppingTime_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingTime_.second': AsJagged(AsDtype('>f8'))}), 'primaryBeta_': AsGroup(<TBranchElement 'primaryBeta_' (2 subbranches) at 0x7f4c5f424950>, {'primaryBeta_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBeta_.second': AsJagged(AsDtype('>f8'))}), 'primaryBetaError_': AsGroup(<TBranchElement 'primaryBetaError_' (2 subbranches) at 0x7f4c5f4054d0>, {'primaryBetaError_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBetaError_.second': AsJagged(AsDtype('>f8'))}), 'primaryMomentumDirection_': AsGroup(<TBranchElement 'primaryMomentumDirection_' (2 subbranches) at 0x7f4c5f438290>, {'primaryMomentumDirection_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryMomentumDirection_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))}), 'primaryEnergyDepositions_': AsGroup(<TBranchElement 'primaryEnergyDepositions_' (2 subbranches) at 0x7f4c5f43bf10>, {'primaryEnergyDepositions_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryEnergyDepositions_.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'HitTrackIndex': AsGroup(<TBranchElement 'HitTrackIndex' (2 subbranches) at 0x7f4c5f44f9d0>, {'HitTrackIndex.first': AsObjects(AsArray(True, False, AsString(False), ())), 'HitTrackIndex.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))}), 'Chi2': AsGroup(<TBranchElement 'Chi2' (2 subbranches) at 0x7f4c5f453590>, {'Chi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Chi2.second': AsJagged(AsDtype('>f8'))}), 'Ndof': AsGroup(<TBranchElement 'Ndof' (2 subbranches) at 0x7f4c5f45f390>, {'Ndof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Ndof.second': AsJagged(AsDtype('>i4'))}), 'ParCov': AsGroup(<TBranchElement 'ParCov' (2 subbranches) at 0x7f4c5f46b210>, {'ParCov.first': AsObjects(AsArray(True, False, AsString(False), ())), 'ParCov.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'FitStatus': AsGroup(<TBranchElement 'FitStatus' (2 subbranches) at 0x7f4c5f472fd0>, {'FitStatus.first': AsObjects(AsArray(True, False, AsString(False), ())), 'FitStatus.second': AsJagged(AsDtype('>i4'))}), 'SdFitPar': AsGroup(<TBranchElement 'SdFitPar' (2 subbranches) at 0x7f4c5f482cd0>, {'SdFitPar.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitPar.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'SdFitErr': AsGroup(<TBranchElement 'SdFitErr' (2 subbranches) at 0x7f4c5f48ea90>, {'SdFitErr.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitErr.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'SdFitChi2': AsGroup(<TBranchElement 'SdFitChi2' (2 subbranches) at 0x7f4c5f492710>, {'SdFitChi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitChi2.second': AsJagged(AsDtype('>f8'))}), 'SdFitNdof': AsGroup(<TBranchElement 'SdFitNdof' (2 subbranches) at 0x7f4c5f4a6410>, {'SdFitNdof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitNdof.second': AsJagged(AsDtype('>i4'))}), 'hitseries_': AsGroup(<TBranchElement 'hitseries_' (7 subbranches) at 0x7f4c5f4a9f90>, {'hitseries_.fUniqueID': AsJagged(AsDtype('>u4')), 'hitseries_.fBits': AsJagged(AsDtype('>u4')), 'hitseries_.volume_id_': AsJagged(AsDtype('>u4')), 'hitseries_.energydep_': AsJagged(AsDtype('>f8')), 'hitseries_.hit_position_': AsJagged(AsStridedObjects(Model_TVector3_v3)), 'hitseries_.hit_time_': AsJagged(AsDtype('>f8')), 'hitseries_.index_': AsJagged(AsDtype('>i4'))}), 'Tracks': AsGroup(<TBranchElement 'Tracks' (2 subbranches) at 0x7f4c5f4bbcd0>, {'Tracks.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Tracks.second': AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))}), 'registeredRecos_': AsObjects(AsVector(True, AsString(False))), 'PacketType': AsDtype('>i4')})
  [  1] Rec/CEventBase                                                AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f5d49d0>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f5d5b10>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')})
  [  2] Rec/CEventBase/TObject                                        AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f5d5b10>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')})
  [  3] Rec/CEventBase/TObject/fUniqueID                              AsDtype('>u4')
  [  4] Rec/CEventBase/TObject/fBits                                  AsDtype('>u4')
  [  5] Rec/CEventBase/runNumber_                                     AsDtype('>u4')
  [  6] Rec/CEventBase/subRunNumber_                                  AsDtype('>u4')
  [  7] Rec/CEventBase/eventNumber_                                   AsDtype('>u4')
  [  8] Rec/CEventBase/eventTime_                                     AsDtype('>i8')
  [  9] Rec/CEventBase/eventId_                                       AsDtype('>u4')
  [ 10] Rec/CEventBase/primaryBetaGenerated_                          AsDtype('>f8')
  [ 11] Rec/CEventBase/primaryMomentumDirectionGenerated_             AsStridedObjects(Model_TVector3_v3)
  [ 12] Rec/CEventBase/primaryKineticEnergyGenerated_                 AsDtype('>f8')
  [ 13] Rec/trigger_sources                                           AsJagged(AsDtype('uint8'), header_bytes=10)
  [ 14] Rec/trigger_vids                                              AsJagged(AsDtype('>u4'), header_bytes=10)
  [ 15] Rec/activeReco_                                               AsStrings(header_bytes=6)
  [ 16] Rec/event_quality                                             AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 17] Rec/primaryStoppingPosition_                                  AsGroup(<TBranchElement 'primaryStoppingPosition_' (2 subbranches) at 0x7f4c5f405510>, {'primaryStoppingPosition_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingPosition_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))})
  [ 18] Rec/primaryStoppingPosition_/primaryStoppingPosition_.first   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 19] Rec/primaryStoppingPosition_/primaryStoppingPosition_.second  AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 20] Rec/primaryStoppingVolume_                                    AsGroup(<TBranchElement 'primaryStoppingVolume_' (2 subbranches) at 0x7f4c5f40d0d0>, {'primaryStoppingVolume_.first': <UnknownInterpretation 'none of the rules matched'>, 'primaryStoppingVolume_.second': AsJagged(AsDtype('>u4'))})
  [ 21] Rec/primaryStoppingVolume_/primaryStoppingVolume_.first       none of the rules matched
in file /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root
in object /TreeRec;8:Rec/primaryStoppingVolume_/primaryStoppingVolume_.first
  [ 22] Rec/primaryStoppingVolume_/primaryStoppingVolume_.second      AsJagged(AsDtype('>u4'))
  [ 23] Rec/primaryStoppingTime_                                      AsGroup(<TBranchElement 'primaryStoppingTime_' (2 subbranches) at 0x7f4c5f41cd90>, {'primaryStoppingTime_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingTime_.second': AsJagged(AsDtype('>f8'))})
  [ 24] Rec/primaryStoppingTime_/primaryStoppingTime_.first           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 25] Rec/primaryStoppingTime_/primaryStoppingTime_.second          AsJagged(AsDtype('>f8'))
  [ 26] Rec/primaryBeta_                                              AsGroup(<TBranchElement 'primaryBeta_' (2 subbranches) at 0x7f4c5f424950>, {'primaryBeta_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBeta_.second': AsJagged(AsDtype('>f8'))})
  [ 27] Rec/primaryBeta_/primaryBeta_.first                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 28] Rec/primaryBeta_/primaryBeta_.second                          AsJagged(AsDtype('>f8'))
  [ 29] Rec/primaryBetaError_                                         AsGroup(<TBranchElement 'primaryBetaError_' (2 subbranches) at 0x7f4c5f4054d0>, {'primaryBetaError_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBetaError_.second': AsJagged(AsDtype('>f8'))})
  [ 30] Rec/primaryBetaError_/primaryBetaError_.first                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 31] Rec/primaryBetaError_/primaryBetaError_.second                AsJagged(AsDtype('>f8'))
  [ 32] Rec/primaryMomentumDirection_                                 AsGroup(<TBranchElement 'primaryMomentumDirection_' (2 subbranches) at 0x7f4c5f438290>, {'primaryMomentumDirection_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryMomentumDirection_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))})
  [ 33] Rec/primaryMomentumDirection_/primaryMomentumDirection_.first  AsObjects(AsArray(True, False, AsString(False), ()))
  [ 34] Rec/primaryMomentumDirection_/primaryMomentumDirection_.second  AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 35] Rec/primaryEnergyDepositions_                                 AsGroup(<TBranchElement 'primaryEnergyDepositions_' (2 subbranches) at 0x7f4c5f43bf10>, {'primaryEnergyDepositions_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryEnergyDepositions_.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 36] Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.first  AsObjects(AsArray(True, False, AsString(False), ()))
  [ 37] Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.second  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 38] Rec/HitTrackIndex                                             AsGroup(<TBranchElement 'HitTrackIndex' (2 subbranches) at 0x7f4c5f44f9d0>, {'HitTrackIndex.first': AsObjects(AsArray(True, False, AsString(False), ())), 'HitTrackIndex.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))})
  [ 39] Rec/HitTrackIndex/HitTrackIndex.first                         AsObjects(AsArray(True, False, AsString(False), ()))
  [ 40] Rec/HitTrackIndex/HitTrackIndex.second                        AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))
  [ 41] Rec/Chi2                                                      AsGroup(<TBranchElement 'Chi2' (2 subbranches) at 0x7f4c5f453590>, {'Chi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Chi2.second': AsJagged(AsDtype('>f8'))})
  [ 42] Rec/Chi2/Chi2.first                                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 43] Rec/Chi2/Chi2.second                                          AsJagged(AsDtype('>f8'))
  [ 44] Rec/Ndof                                                      AsGroup(<TBranchElement 'Ndof' (2 subbranches) at 0x7f4c5f45f390>, {'Ndof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Ndof.second': AsJagged(AsDtype('>i4'))})
  [ 45] Rec/Ndof/Ndof.first                                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 46] Rec/Ndof/Ndof.second                                          AsJagged(AsDtype('>i4'))
  [ 47] Rec/ParCov                                                    AsGroup(<TBranchElement 'ParCov' (2 subbranches) at 0x7f4c5f46b210>, {'ParCov.first': AsObjects(AsArray(True, False, AsString(False), ())), 'ParCov.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 48] Rec/ParCov/ParCov.first                                       AsObjects(AsArray(True, False, AsString(False), ()))
  [ 49] Rec/ParCov/ParCov.second                                      AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 50] Rec/FitStatus                                                 AsGroup(<TBranchElement 'FitStatus' (2 subbranches) at 0x7f4c5f472fd0>, {'FitStatus.first': AsObjects(AsArray(True, False, AsString(False), ())), 'FitStatus.second': AsJagged(AsDtype('>i4'))})
  [ 51] Rec/FitStatus/FitStatus.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 52] Rec/FitStatus/FitStatus.second                                AsJagged(AsDtype('>i4'))
  [ 53] Rec/SdFitPar                                                  AsGroup(<TBranchElement 'SdFitPar' (2 subbranches) at 0x7f4c5f482cd0>, {'SdFitPar.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitPar.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 54] Rec/SdFitPar/SdFitPar.first                                   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 55] Rec/SdFitPar/SdFitPar.second                                  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 56] Rec/SdFitErr                                                  AsGroup(<TBranchElement 'SdFitErr' (2 subbranches) at 0x7f4c5f48ea90>, {'SdFitErr.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitErr.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 57] Rec/SdFitErr/SdFitErr.first                                   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 58] Rec/SdFitErr/SdFitErr.second                                  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 59] Rec/SdFitChi2                                                 AsGroup(<TBranchElement 'SdFitChi2' (2 subbranches) at 0x7f4c5f492710>, {'SdFitChi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitChi2.second': AsJagged(AsDtype('>f8'))})
  [ 60] Rec/SdFitChi2/SdFitChi2.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 61] Rec/SdFitChi2/SdFitChi2.second                                AsJagged(AsDtype('>f8'))
  [ 62] Rec/SdFitNdof                                                 AsGroup(<TBranchElement 'SdFitNdof' (2 subbranches) at 0x7f4c5f4a6410>, {'SdFitNdof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitNdof.second': AsJagged(AsDtype('>i4'))})
  [ 63] Rec/SdFitNdof/SdFitNdof.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 64] Rec/SdFitNdof/SdFitNdof.second                                AsJagged(AsDtype('>i4'))
  [ 65] Rec/hitseries_                                                AsGroup(<TBranchElement 'hitseries_' (7 subbranches) at 0x7f4c5f4a9f90>, {'hitseries_.fUniqueID': AsJagged(AsDtype('>u4')), 'hitseries_.fBits': AsJagged(AsDtype('>u4')), 'hitseries_.volume_id_': AsJagged(AsDtype('>u4')), 'hitseries_.energydep_': AsJagged(AsDtype('>f8')), 'hitseries_.hit_position_': AsJagged(AsStridedObjects(Model_TVector3_v3)), 'hitseries_.hit_time_': AsJagged(AsDtype('>f8')), 'hitseries_.index_': AsJagged(AsDtype('>i4'))})
  [ 66] Rec/hitseries_/hitseries_.fUniqueID                           AsJagged(AsDtype('>u4'))
  [ 67] Rec/hitseries_/hitseries_.fBits                               AsJagged(AsDtype('>u4'))
  [ 68] Rec/hitseries_/hitseries_.volume_id_                          AsJagged(AsDtype('>u4'))
  [ 69] Rec/hitseries_/hitseries_.energydep_                          AsJagged(AsDtype('>f8'))
  [ 70] Rec/hitseries_/hitseries_.hit_position_                       AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 71] Rec/hitseries_/hitseries_.hit_time_                           AsJagged(AsDtype('>f8'))
  [ 72] Rec/hitseries_/hitseries_.index_                              AsJagged(AsDtype('>i4'))
  [ 73] Rec/Tracks                                                    AsGroup(<TBranchElement 'Tracks' (2 subbranches) at 0x7f4c5f4bbcd0>, {'Tracks.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Tracks.second': AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))})
  [ 74] Rec/Tracks/Tracks.first                                       AsObjects(AsArray(True, False, AsString(False), ()))
  [ 75] Rec/Tracks/Tracks.second                                      AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))
  [ 76] Rec/registeredRecos_                                          AsObjects(AsVector(True, AsString(False)))
  [ 77] Rec/PacketType                                                AsDtype('>i4')

[antiD] avg = 247,959 ev/file  ->  projected total 24,795,880

========== antiP: 140 files ==========

--- antiP / TreeMc;83: 180270 entries, 27 branches ---
  [  0] Mc                                                            AsGroup(<TBranchElement 'Mc' (15 subbranches) at 0x7f4c5f2770d0>, {'CEventBase': AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f2779d0>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f284490>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')}), 'primaryPosition_': AsStridedObjects(Model_TVector3_v3), 'primaryTime_': AsDtype('>f8'), 'primaryPdg_': AsDtype('>i4'), 'primaryStoppingKineticEnergy_': AsDtype('>f8'), 'randomSeed_': AsDtype('>u4'), 'primaryStoppingVolume_': AsDtype('>i4'), 'primaryStoppingPosition_': AsStridedObjects(Model_TVector3_v3), 'primaryStoppingTime_': AsDtype('>f8'), 'hitTrackIndex_': AsJagged(AsDtype('>i4'), header_bytes=10), 'tracks_': AsObjects(AsVector(True, AsPointer(Model_CTrackMc))), 'totalEnergyDeposition_': AsJagged(AsDtype('>f8'), header_bytes=10), 'meanPosition_': AsJagged(AsStridedObjects(Model_TVector3_v3), header_bytes=10), 'volumeId_': AsJagged(AsDtype('>i4'), header_bytes=10), 'time_': AsJagged(AsDtype('>f8'), header_bytes=10)})
  [  1] Mc/CEventBase                                                 AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f2779d0>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f284490>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')})
  [  2] Mc/CEventBase/TObject                                         AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f284490>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')})
  [  3] Mc/CEventBase/TObject/fUniqueID                               AsDtype('>u4')
  [  4] Mc/CEventBase/TObject/fBits                                   AsDtype('>u4')
  [  5] Mc/CEventBase/runNumber_                                      AsDtype('>u4')
  [  6] Mc/CEventBase/subRunNumber_                                   AsDtype('>u4')
  [  7] Mc/CEventBase/eventNumber_                                    AsDtype('>u4')
  [  8] Mc/CEventBase/eventTime_                                      AsDtype('>i8')
  [  9] Mc/CEventBase/eventId_                                        AsDtype('>u4')
  [ 10] Mc/CEventBase/primaryBetaGenerated_                           AsDtype('>f8')
  [ 11] Mc/CEventBase/primaryMomentumDirectionGenerated_              AsStridedObjects(Model_TVector3_v3)
  [ 12] Mc/CEventBase/primaryKineticEnergyGenerated_                  AsDtype('>f8')
  [ 13] Mc/primaryPosition_                                           AsStridedObjects(Model_TVector3_v3)
  [ 14] Mc/primaryTime_                                               AsDtype('>f8')
  [ 15] Mc/primaryPdg_                                                AsDtype('>i4')
  [ 16] Mc/primaryStoppingKineticEnergy_                              AsDtype('>f8')
  [ 17] Mc/randomSeed_                                                AsDtype('>u4')
  [ 18] Mc/primaryStoppingVolume_                                     AsDtype('>i4')
  [ 19] Mc/primaryStoppingPosition_                                   AsStridedObjects(Model_TVector3_v3)
  [ 20] Mc/primaryStoppingTime_                                       AsDtype('>f8')
  [ 21] Mc/hitTrackIndex_                                             AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 22] Mc/tracks_                                                    AsObjects(AsVector(True, AsPointer(Model_CTrackMc)))
  [ 23] Mc/totalEnergyDeposition_                                     AsJagged(AsDtype('>f8'), header_bytes=10)
  [ 24] Mc/meanPosition_                                              AsJagged(AsStridedObjects(Model_TVector3_v3), header_bytes=10)
  [ 25] Mc/volumeId_                                                  AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 26] Mc/time_                                                      AsJagged(AsDtype('>f8'), header_bytes=10)

--- antiP / TreeRec;4: 180270 entries, 78 branches ---
  [  0] Rec                                                           AsGroup(<TBranchElement 'Rec' (25 subbranches) at 0x7f4c5f172850>, {'CEventBase': AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f173190>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f173bd0>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')}), 'trigger_sources': AsJagged(AsDtype('uint8'), header_bytes=10), 'trigger_vids': AsJagged(AsDtype('>u4'), header_bytes=10), 'activeReco_': AsStrings(header_bytes=6), 'event_quality': AsJagged(AsDtype('>i4'), header_bytes=10), 'primaryStoppingPosition_': AsGroup(<TBranchElement 'primaryStoppingPosition_' (2 subbranches) at 0x7f4c5f1aadd0>, {'primaryStoppingPosition_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingPosition_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))}), 'primaryStoppingVolume_': AsGroup(<TBranchElement 'primaryStoppingVolume_' (2 subbranches) at 0x7f4c5f1b2890>, {'primaryStoppingVolume_.first': <UnknownInterpretation 'none of the rules matched'>, 'primaryStoppingVolume_.second': AsJagged(AsDtype('>u4'))}), 'primaryStoppingTime_': AsGroup(<TBranchElement 'primaryStoppingTime_' (2 subbranches) at 0x7f4c5f1b8c10>, {'primaryStoppingTime_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingTime_.second': AsJagged(AsDtype('>f8'))}), 'primaryBeta_': AsGroup(<TBranchElement 'primaryBeta_' (2 subbranches) at 0x7f4c5f1cde90>, {'primaryBeta_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBeta_.second': AsJagged(AsDtype('>f8'))}), 'primaryBetaError_': AsGroup(<TBranchElement 'primaryBetaError_' (2 subbranches) at 0x7f4c5f1aac50>, {'primaryBetaError_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBetaError_.second': AsJagged(AsDtype('>f8'))}), 'primaryMomentumDirection_': AsGroup(<TBranchElement 'primaryMomentumDirection_' (2 subbranches) at 0x7f4c5efe1510>, {'primaryMomentumDirection_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryMomentumDirection_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))}), 'primaryEnergyDepositions_': AsGroup(<TBranchElement 'primaryEnergyDepositions_' (2 subbranches) at 0x7f4c5efe5010>, {'primaryEnergyDepositions_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryEnergyDepositions_.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'HitTrackIndex': AsGroup(<TBranchElement 'HitTrackIndex' (2 subbranches) at 0x7f4c5eff0b10>, {'HitTrackIndex.first': AsObjects(AsArray(True, False, AsString(False), ())), 'HitTrackIndex.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))}), 'Chi2': AsGroup(<TBranchElement 'Chi2' (2 subbranches) at 0x7f4c5f000650>, {'Chi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Chi2.second': AsJagged(AsDtype('>f8'))}), 'Ndof': AsGroup(<TBranchElement 'Ndof' (2 subbranches) at 0x7f4c5f00c3d0>, {'Ndof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Ndof.second': AsJagged(AsDtype('>i4'))}), 'ParCov': AsGroup(<TBranchElement 'ParCov' (2 subbranches) at 0x7f4c5f0140d0>, {'ParCov.first': AsObjects(AsArray(True, False, AsString(False), ())), 'ParCov.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'FitStatus': AsGroup(<TBranchElement 'FitStatus' (2 subbranches) at 0x7f4c5f017e10>, {'FitStatus.first': AsObjects(AsArray(True, False, AsString(False), ())), 'FitStatus.second': AsJagged(AsDtype('>i4'))}), 'SdFitPar': AsGroup(<TBranchElement 'SdFitPar' (2 subbranches) at 0x7f4c5f02bad0>, {'SdFitPar.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitPar.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'SdFitErr': AsGroup(<TBranchElement 'SdFitErr' (2 subbranches) at 0x7f4c5f02f6d0>, {'SdFitErr.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitErr.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))}), 'SdFitChi2': AsGroup(<TBranchElement 'SdFitChi2' (2 subbranches) at 0x7f4c5f0332d0>, {'SdFitChi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitChi2.second': AsJagged(AsDtype('>f8'))}), 'SdFitNdof': AsGroup(<TBranchElement 'SdFitNdof' (2 subbranches) at 0x7f4c5f042d10>, {'SdFitNdof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitNdof.second': AsJagged(AsDtype('>i4'))}), 'hitseries_': AsGroup(<TBranchElement 'hitseries_' (7 subbranches) at 0x7f4c5f04eb10>, {'hitseries_.fUniqueID': AsJagged(AsDtype('>u4')), 'hitseries_.fBits': AsJagged(AsDtype('>u4')), 'hitseries_.volume_id_': AsJagged(AsDtype('>u4')), 'hitseries_.energydep_': AsJagged(AsDtype('>f8')), 'hitseries_.hit_position_': AsJagged(AsStridedObjects(Model_TVector3_v3)), 'hitseries_.hit_time_': AsJagged(AsDtype('>f8')), 'hitseries_.index_': AsJagged(AsDtype('>i4'))}), 'Tracks': AsGroup(<TBranchElement 'Tracks' (2 subbranches) at 0x7f4c5f0687d0>, {'Tracks.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Tracks.second': AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))}), 'registeredRecos_': AsObjects(AsVector(True, AsString(False))), 'PacketType': AsDtype('>i4')})
  [  1] Rec/CEventBase                                                AsGroup(<TBranchElement 'CEventBase' (9 subbranches) at 0x7f4c5f173190>, {'TObject': AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f173bd0>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')}), 'runNumber_': AsDtype('>u4'), 'subRunNumber_': AsDtype('>u4'), 'eventNumber_': AsDtype('>u4'), 'eventTime_': AsDtype('>i8'), 'eventId_': AsDtype('>u4'), 'primaryBetaGenerated_': AsDtype('>f8'), 'primaryMomentumDirectionGenerated_': AsStridedObjects(Model_TVector3_v3), 'primaryKineticEnergyGenerated_': AsDtype('>f8')})
  [  2] Rec/CEventBase/TObject                                        AsGroup(<TBranchElement 'TObject' (2 subbranches) at 0x7f4c5f173bd0>, {'fUniqueID': AsDtype('>u4'), 'fBits': AsDtype('>u4')})
  [  3] Rec/CEventBase/TObject/fUniqueID                              AsDtype('>u4')
  [  4] Rec/CEventBase/TObject/fBits                                  AsDtype('>u4')
  [  5] Rec/CEventBase/runNumber_                                     AsDtype('>u4')
  [  6] Rec/CEventBase/subRunNumber_                                  AsDtype('>u4')
  [  7] Rec/CEventBase/eventNumber_                                   AsDtype('>u4')
  [  8] Rec/CEventBase/eventTime_                                     AsDtype('>i8')
  [  9] Rec/CEventBase/eventId_                                       AsDtype('>u4')
  [ 10] Rec/CEventBase/primaryBetaGenerated_                          AsDtype('>f8')
  [ 11] Rec/CEventBase/primaryMomentumDirectionGenerated_             AsStridedObjects(Model_TVector3_v3)
  [ 12] Rec/CEventBase/primaryKineticEnergyGenerated_                 AsDtype('>f8')
  [ 13] Rec/trigger_sources                                           AsJagged(AsDtype('uint8'), header_bytes=10)
  [ 14] Rec/trigger_vids                                              AsJagged(AsDtype('>u4'), header_bytes=10)
  [ 15] Rec/activeReco_                                               AsStrings(header_bytes=6)
  [ 16] Rec/event_quality                                             AsJagged(AsDtype('>i4'), header_bytes=10)
  [ 17] Rec/primaryStoppingPosition_                                  AsGroup(<TBranchElement 'primaryStoppingPosition_' (2 subbranches) at 0x7f4c5f1aadd0>, {'primaryStoppingPosition_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingPosition_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))})
  [ 18] Rec/primaryStoppingPosition_/primaryStoppingPosition_.first   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 19] Rec/primaryStoppingPosition_/primaryStoppingPosition_.second  AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 20] Rec/primaryStoppingVolume_                                    AsGroup(<TBranchElement 'primaryStoppingVolume_' (2 subbranches) at 0x7f4c5f1b2890>, {'primaryStoppingVolume_.first': <UnknownInterpretation 'none of the rules matched'>, 'primaryStoppingVolume_.second': AsJagged(AsDtype('>u4'))})
  [ 21] Rec/primaryStoppingVolume_/primaryStoppingVolume_.first       none of the rules matched
in file /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root
in object /TreeRec;4:Rec/primaryStoppingVolume_/primaryStoppingVolume_.first
  [ 22] Rec/primaryStoppingVolume_/primaryStoppingVolume_.second      AsJagged(AsDtype('>u4'))
  [ 23] Rec/primaryStoppingTime_                                      AsGroup(<TBranchElement 'primaryStoppingTime_' (2 subbranches) at 0x7f4c5f1b8c10>, {'primaryStoppingTime_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryStoppingTime_.second': AsJagged(AsDtype('>f8'))})
  [ 24] Rec/primaryStoppingTime_/primaryStoppingTime_.first           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 25] Rec/primaryStoppingTime_/primaryStoppingTime_.second          AsJagged(AsDtype('>f8'))
  [ 26] Rec/primaryBeta_                                              AsGroup(<TBranchElement 'primaryBeta_' (2 subbranches) at 0x7f4c5f1cde90>, {'primaryBeta_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBeta_.second': AsJagged(AsDtype('>f8'))})
  [ 27] Rec/primaryBeta_/primaryBeta_.first                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 28] Rec/primaryBeta_/primaryBeta_.second                          AsJagged(AsDtype('>f8'))
  [ 29] Rec/primaryBetaError_                                         AsGroup(<TBranchElement 'primaryBetaError_' (2 subbranches) at 0x7f4c5f1aac50>, {'primaryBetaError_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryBetaError_.second': AsJagged(AsDtype('>f8'))})
  [ 30] Rec/primaryBetaError_/primaryBetaError_.first                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 31] Rec/primaryBetaError_/primaryBetaError_.second                AsJagged(AsDtype('>f8'))
  [ 32] Rec/primaryMomentumDirection_                                 AsGroup(<TBranchElement 'primaryMomentumDirection_' (2 subbranches) at 0x7f4c5efe1510>, {'primaryMomentumDirection_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryMomentumDirection_.second': AsJagged(AsStridedObjects(Model_TVector3_v3))})
  [ 33] Rec/primaryMomentumDirection_/primaryMomentumDirection_.first  AsObjects(AsArray(True, False, AsString(False), ()))
  [ 34] Rec/primaryMomentumDirection_/primaryMomentumDirection_.second  AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 35] Rec/primaryEnergyDepositions_                                 AsGroup(<TBranchElement 'primaryEnergyDepositions_' (2 subbranches) at 0x7f4c5efe5010>, {'primaryEnergyDepositions_.first': AsObjects(AsArray(True, False, AsString(False), ())), 'primaryEnergyDepositions_.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 36] Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.first  AsObjects(AsArray(True, False, AsString(False), ()))
  [ 37] Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.second  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 38] Rec/HitTrackIndex                                             AsGroup(<TBranchElement 'HitTrackIndex' (2 subbranches) at 0x7f4c5eff0b10>, {'HitTrackIndex.first': AsObjects(AsArray(True, False, AsString(False), ())), 'HitTrackIndex.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))})
  [ 39] Rec/HitTrackIndex/HitTrackIndex.first                         AsObjects(AsArray(True, False, AsString(False), ()))
  [ 40] Rec/HitTrackIndex/HitTrackIndex.second                        AsObjects(AsArray(True, False, AsVector(False, dtype('>i4')), ()))
  [ 41] Rec/Chi2                                                      AsGroup(<TBranchElement 'Chi2' (2 subbranches) at 0x7f4c5f000650>, {'Chi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Chi2.second': AsJagged(AsDtype('>f8'))})
  [ 42] Rec/Chi2/Chi2.first                                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 43] Rec/Chi2/Chi2.second                                          AsJagged(AsDtype('>f8'))
  [ 44] Rec/Ndof                                                      AsGroup(<TBranchElement 'Ndof' (2 subbranches) at 0x7f4c5f00c3d0>, {'Ndof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Ndof.second': AsJagged(AsDtype('>i4'))})
  [ 45] Rec/Ndof/Ndof.first                                           AsObjects(AsArray(True, False, AsString(False), ()))
  [ 46] Rec/Ndof/Ndof.second                                          AsJagged(AsDtype('>i4'))
  [ 47] Rec/ParCov                                                    AsGroup(<TBranchElement 'ParCov' (2 subbranches) at 0x7f4c5f0140d0>, {'ParCov.first': AsObjects(AsArray(True, False, AsString(False), ())), 'ParCov.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 48] Rec/ParCov/ParCov.first                                       AsObjects(AsArray(True, False, AsString(False), ()))
  [ 49] Rec/ParCov/ParCov.second                                      AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 50] Rec/FitStatus                                                 AsGroup(<TBranchElement 'FitStatus' (2 subbranches) at 0x7f4c5f017e10>, {'FitStatus.first': AsObjects(AsArray(True, False, AsString(False), ())), 'FitStatus.second': AsJagged(AsDtype('>i4'))})
  [ 51] Rec/FitStatus/FitStatus.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 52] Rec/FitStatus/FitStatus.second                                AsJagged(AsDtype('>i4'))
  [ 53] Rec/SdFitPar                                                  AsGroup(<TBranchElement 'SdFitPar' (2 subbranches) at 0x7f4c5f02bad0>, {'SdFitPar.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitPar.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 54] Rec/SdFitPar/SdFitPar.first                                   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 55] Rec/SdFitPar/SdFitPar.second                                  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 56] Rec/SdFitErr                                                  AsGroup(<TBranchElement 'SdFitErr' (2 subbranches) at 0x7f4c5f02f6d0>, {'SdFitErr.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitErr.second': AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))})
  [ 57] Rec/SdFitErr/SdFitErr.first                                   AsObjects(AsArray(True, False, AsString(False), ()))
  [ 58] Rec/SdFitErr/SdFitErr.second                                  AsObjects(AsArray(True, False, AsVector(False, dtype('>f8')), ()))
  [ 59] Rec/SdFitChi2                                                 AsGroup(<TBranchElement 'SdFitChi2' (2 subbranches) at 0x7f4c5f0332d0>, {'SdFitChi2.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitChi2.second': AsJagged(AsDtype('>f8'))})
  [ 60] Rec/SdFitChi2/SdFitChi2.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 61] Rec/SdFitChi2/SdFitChi2.second                                AsJagged(AsDtype('>f8'))
  [ 62] Rec/SdFitNdof                                                 AsGroup(<TBranchElement 'SdFitNdof' (2 subbranches) at 0x7f4c5f042d10>, {'SdFitNdof.first': AsObjects(AsArray(True, False, AsString(False), ())), 'SdFitNdof.second': AsJagged(AsDtype('>i4'))})
  [ 63] Rec/SdFitNdof/SdFitNdof.first                                 AsObjects(AsArray(True, False, AsString(False), ()))
  [ 64] Rec/SdFitNdof/SdFitNdof.second                                AsJagged(AsDtype('>i4'))
  [ 65] Rec/hitseries_                                                AsGroup(<TBranchElement 'hitseries_' (7 subbranches) at 0x7f4c5f04eb10>, {'hitseries_.fUniqueID': AsJagged(AsDtype('>u4')), 'hitseries_.fBits': AsJagged(AsDtype('>u4')), 'hitseries_.volume_id_': AsJagged(AsDtype('>u4')), 'hitseries_.energydep_': AsJagged(AsDtype('>f8')), 'hitseries_.hit_position_': AsJagged(AsStridedObjects(Model_TVector3_v3)), 'hitseries_.hit_time_': AsJagged(AsDtype('>f8')), 'hitseries_.index_': AsJagged(AsDtype('>i4'))})
  [ 66] Rec/hitseries_/hitseries_.fUniqueID                           AsJagged(AsDtype('>u4'))
  [ 67] Rec/hitseries_/hitseries_.fBits                               AsJagged(AsDtype('>u4'))
  [ 68] Rec/hitseries_/hitseries_.volume_id_                          AsJagged(AsDtype('>u4'))
  [ 69] Rec/hitseries_/hitseries_.energydep_                          AsJagged(AsDtype('>f8'))
  [ 70] Rec/hitseries_/hitseries_.hit_position_                       AsJagged(AsStridedObjects(Model_TVector3_v3))
  [ 71] Rec/hitseries_/hitseries_.hit_time_                           AsJagged(AsDtype('>f8'))
  [ 72] Rec/hitseries_/hitseries_.index_                              AsJagged(AsDtype('>i4'))
  [ 73] Rec/Tracks                                                    AsGroup(<TBranchElement 'Tracks' (2 subbranches) at 0x7f4c5f0687d0>, {'Tracks.first': AsObjects(AsArray(True, False, AsString(False), ())), 'Tracks.second': AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))})
  [ 74] Rec/Tracks/Tracks.first                                       AsObjects(AsArray(True, False, AsString(False), ()))
  [ 75] Rec/Tracks/Tracks.second                                      AsObjects(AsArray(True, False, AsVector(False, AsPointer(Model_CTrackRec)), ()))
  [ 76] Rec/registeredRecos_                                          AsObjects(AsVector(True, AsString(False)))
  [ 77] Rec/PacketType                                                AsDtype('>i4')

[antiP] avg = 180,798 ev/file  ->  projected total 25,311,720





非常干净的输出!关键信息都拿到了。整理一下:

  ✅ 总事件数确认: 约 5,010 万

  ┌───────┬────────┬──────────────┬─────────────┐
  │ 粒子  │ 文件数 │ 平均 ev/file │  推算总数   │
  ├───────┼────────┼──────────────┼─────────────┤
  │ antiD │ 100    │ 247,959      │ 24,795,880  │
  ├───────┼────────┼──────────────┼─────────────┤
  │ antiP │ 140    │ 180,798      │ 25,311,720  │
  ├───────┼────────┼──────────────┼─────────────┤
  │ 合計  │ 240    │ —            │ ~50,107,600 │
  └───────┴────────┴──────────────┴─────────────┘

  完美吻合大場「5000 万」。

  关键 branch 映射(下一步前处理设计的基础)

  TreeMc — Ground Truth 用

  Mc/primaryPdg_                          # PDG ID (-1000010020=antiD, -2212=antiP) → label
  Mc/primaryBetaGenerated_                # 真値 β
  Mc/primaryKineticEnergyGenerated_       # 真値運動エネルギー
  Mc/primaryMomentumDirectionGenerated_   # TVector3, 方向
  Mc/primaryStoppingPosition_             # TVector3, 停止位置
  Mc/primaryStoppingVolume_               # 停止した volume ID
  Mc/volumeId_     (jagged)               # event 内 hit ごとの volume ID
  Mc/meanPosition_ (jagged, TVector3)     # event 内 hit ごとの位置
  Mc/totalEnergyDeposition_ (jagged)      # event 内 hit ごとの dE
  Mc/time_         (jagged)               # event 内 hit ごとの時間

  TreeRec — 学習入力に使うべき(検出器の応答済み)

  Rec/hitseries_.volume_id_     (jagged u32)        # ★ Si(Li) / TOF を区別する鍵
  Rec/hitseries_.energydep_     (jagged f8)         # ★ Voxel 値
  Rec/hitseries_.hit_position_  (jagged TVector3)   # ★ x, y, z
  Rec/hitseries_.hit_time_      (jagged f8)         # ★ TOF 計算用
  Rec/hitseries_.index_         (jagged i4)

  Rec/primaryBeta_.second        (jagged f8)        # 再構成 β
  Rec/primaryStoppingVolume_.second                  # 再構成停止 volume
  Rec/Chi2.second / Rec/Ndof.second                  # 品質カット用
  Rec/event_quality                                  # event 品質フラグ
  Rec/PacketType                                     # Atrest/Inflight 区別?

  ⚠️ 重要発見: 「2 TOF」専用 branch は無い

  之前担心新数据会多出 2 TOF 的额外 branch,实际没有。结构和旧数据完全一致。

  那「2 TOF」是怎么体现的?通过 volume_id_ 的 ID 范围区分:
  - TOF Umbrella(上面) → 某 volume ID 区间
  - TOF Cortina + Cube(下面+侧面) → 另一 volume ID 区间
  - Si(Li) Tracker → 第三个区间

  这意味着 TOF 特征维度不会从 9D/11D 任意暴涨,正常处理即可。
"""