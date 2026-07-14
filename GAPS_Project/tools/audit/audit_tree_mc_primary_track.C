#include <iostream>
#include <map>
#include <string>

#include "TChain.h"

#include "CEventBase.hh"
#include "CEventMc.hh"
#include "CTrackMc.hh"
#include "GGeometryObject.hh"

void audit_tree_mc_primary_track(const char* root_path, Long64_t max_entries=20000) {
  TChain tree("TreeMc");
  tree.Add(root_path);

  CEventBase* event_base = new CEventMc;
  tree.SetBranchAddress("Mc", &event_base);

  Long64_t n_entries = tree.GetEntries();
  Long64_t n_loop = (max_entries > 0 && max_entries < n_entries) ? max_entries : n_entries;

  Long64_t n_event = 0;
  Long64_t n_no_track = 0;
  Long64_t n_track0_primary = 0;
  Long64_t n_track0_same_as_primary = 0;
  Long64_t n_track0_not_primary = 0;
  Long64_t n_primary_missing = 0;
  Long64_t n_bad_volume_hit = 0;
  Long64_t n_hits_total = 0;

  std::map<int, Long64_t> pdg_count;
  std::map<int, Long64_t> parent_count;
  std::map<int, Long64_t> vol_prefix1;
  std::map<int, Long64_t> vol_prefix2;
  std::map<int, Long64_t> bad_vol_examples;

  for (Long64_t entry = 0; entry < n_loop; ++entry) {
    tree.GetEntry(entry);
    auto* event = dynamic_cast<CEventMc*>(event_base);
    if (!event) continue;

    ++n_event;

    if (event->GetNTracks() == 0) {
      ++n_no_track;
      continue;
    }

    CTrackMc* t0 = event->GetTrack(0);
    CTrackMc* tp = event->GetPrimaryTrack();

    if (!tp) {
      ++n_primary_missing;
    }

    if (t0 && t0->IsPrimary()) {
      ++n_track0_primary;
    } else {
      ++n_track0_not_primary;
    }

    if (t0 && tp && t0 == tp) {
      ++n_track0_same_as_primary;
    }

    if (t0) {
      pdg_count[t0->GetPdg()]++;
      parent_count[t0->GetParentId()]++;

      const auto& vids = t0->GetVolumeId();
      for (auto vid : vids) {
        ++n_hits_total;
        vol_prefix1[int(vid / 100000000)]++;
        vol_prefix2[int(vid / 10000000)]++;

        bool is_det = GGeometryObject::IsTofVolume(vid) || GGeometryObject::IsTrackerVolume(vid);
        if (!is_det) {
          ++n_bad_volume_hit;
          if (bad_vol_examples.size() < 20) bad_vol_examples[vid]++;
        }
      }
    }
  }

  std::cout << "root: " << root_path << "\n";
  std::cout << "entries_total: " << n_entries << "\n";
  std::cout << "entries_looped: " << n_loop << "\n";
  std::cout << "events_seen: " << n_event << "\n";
  std::cout << "no_track: " << n_no_track << "\n";
  std::cout << "primary_missing: " << n_primary_missing << "\n";
  std::cout << "track0_is_primary: " << n_track0_primary << "\n";
  std::cout << "track0_not_primary: " << n_track0_not_primary << "\n";
  std::cout << "track0_same_pointer_as_GetPrimaryTrack: " << n_track0_same_as_primary << "\n";
  std::cout << "hits_total_in_track0: " << n_hits_total << "\n";
  std::cout << "bad_non_tof_non_tracker_hits: " << n_bad_volume_hit << "\n";

  std::cout << "\ntrack0 pdg count:\n";
  for (auto& kv : pdg_count) std::cout << "  " << kv.first << " " << kv.second << "\n";

  std::cout << "\ntrack0 parent id count:\n";
  for (auto& kv : parent_count) std::cout << "  " << kv.first << " " << kv.second << "\n";

  std::cout << "\nvolume_id // 100000000:\n";
  for (auto& kv : vol_prefix1) std::cout << "  " << kv.first << " " << kv.second << "\n";

  std::cout << "\nvolume_id // 10000000:\n";
  for (auto& kv : vol_prefix2) std::cout << "  " << kv.first << " " << kv.second << "\n";

  if (!bad_vol_examples.empty()) {
    std::cout << "\nbad volume examples:\n";
    for (auto& kv : bad_vol_examples) std::cout << "  " << kv.first << " " << kv.second << "\n";
  }
}
