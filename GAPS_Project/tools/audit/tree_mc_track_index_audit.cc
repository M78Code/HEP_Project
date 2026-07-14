#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TChain.h"

#include "CEventBase.hh"
#include "CEventMc.hh"
#include "CTrackMc.hh"
#include "GGeometryObject.hh"

namespace {

struct RunningStats {
  long long n = 0;
  double sum = 0.0;
  double min = 0.0;
  double max = 0.0;

  void add(double x) {
    if (n == 0) {
      min = max = x;
    } else {
      if (x < min) min = x;
      if (x > max) max = x;
    }
    ++n;
    sum += x;
  }

  double mean() const { return n ? sum / n : 0.0; }
};

struct TrackIndexSummary {
  long long tracks_seen = 0;
  long long tracks_primary = 0;
  long long tracks_parent0 = 0;
  long long hits = 0;
  long long tof_hits = 0;
  long long tracker_hits = 0;
  long long non_tof_tracker_hits = 0;
  double edep_sum = 0.0;
  std::map<int, long long> pdg_count;
  std::map<unsigned int, long long> parent_count;
  std::map<unsigned int, long long> trackid_count;
  std::map<int, long long> volume_prefix_count;
};

double sum_vector(const std::vector<double>& xs) {
  double s = 0.0;
  for (double x : xs) s += x;
  return s;
}

void update_summary(TrackIndexSummary& s, const CTrackMc* trk) {
  if (!trk) return;

  ++s.tracks_seen;
  if (trk->IsPrimary()) ++s.tracks_primary;
  if (trk->GetParentId() == 0) ++s.tracks_parent0;
  s.pdg_count[trk->GetPdg()]++;
  s.parent_count[trk->GetParentId()]++;
  s.trackid_count[trk->GetTrackId()]++;

  const auto edeps = trk->GetEnergyDeposition();
  const auto vids = trk->GetVolumeId();
  s.hits += static_cast<long long>(edeps.size());
  s.edep_sum += sum_vector(edeps);

  for (auto vid : vids) {
    s.volume_prefix_count[static_cast<int>(vid / 10000000)]++;
    if (GGeometryObject::IsTofVolume(vid)) {
      ++s.tof_hits;
    } else if (GGeometryObject::IsTrackerVolume(vid)) {
      ++s.tracker_hits;
    } else {
      ++s.non_tof_tracker_hits;
    }
  }
}

void print_top_map(const std::string& name, const std::map<int, long long>& m, int limit = 12) {
  std::vector<std::pair<int, long long>> items(m.begin(), m.end());
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  std::cout << name << "\n";
  int n = 0;
  for (const auto& kv : items) {
    if (n++ >= limit) break;
    std::cout << "  " << kv.first << " " << kv.second << "\n";
  }
}

void print_top_map_uint(const std::string& name, const std::map<unsigned int, long long>& m, int limit = 12) {
  std::vector<std::pair<unsigned int, long long>> items(m.begin(), m.end());
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  std::cout << name << "\n";
  int n = 0;
  for (const auto& kv : items) {
    if (n++ >= limit) break;
    std::cout << "  " << kv.first << " " << kv.second << "\n";
  }
}

void print_summary(int idx, const TrackIndexSummary& s) {
  if (s.tracks_seen == 0) return;

  std::cout << "\n==== track_index " << idx << " ====\n";
  std::cout << "tracks_seen: " << s.tracks_seen << "\n";
  std::cout << "tracks_primary: " << s.tracks_primary << "\n";
  std::cout << "tracks_parent0: " << s.tracks_parent0 << "\n";
  std::cout << "hits: " << s.hits << "\n";
  std::cout << "edep_sum: " << std::setprecision(12) << s.edep_sum << "\n";
  std::cout << "tof_hits: " << s.tof_hits << "\n";
  std::cout << "tracker_hits: " << s.tracker_hits << "\n";
  std::cout << "non_tof_tracker_hits: " << s.non_tof_tracker_hits << "\n";
  print_top_map("pdg_count:", s.pdg_count);
  print_top_map_uint("parent_id_count:", s.parent_count);
  print_top_map_uint("track_id_count:", s.trackid_count);
  print_top_map("volume_id // 10000000:", s.volume_prefix_count);
}

}  // namespace

void tree_mc_track_index_audit(const char* root_path, Long64_t start_entry, Long64_t max_entries, int max_track_index) {
  TChain tree("TreeMc");
  tree.Add(root_path);

  CEventBase* event_base = new CEventMc;
  tree.SetBranchAddress("Mc", &event_base);

  const Long64_t entries_total = tree.GetEntries();
  if (start_entry < 0) start_entry = 0;
  if (start_entry > entries_total) start_entry = entries_total;
  const Long64_t end_entry =
      (max_entries > 0) ? std::min(entries_total, start_entry + max_entries) : entries_total;
  const Long64_t n_loop = end_entry - start_entry;
  std::vector<TrackIndexSummary> by_index(max_track_index + 1);

  long long events_seen = 0;
  long long no_track = 0;
  long long events_with_secondary_tracks = 0;
  long long events_track0_has_non_detector_hits = 0;
  long long events_any_track_has_non_detector_hits = 0;
  std::map<unsigned int, long long> ntracks_count;
  RunningStats track0_hit_fraction;
  RunningStats track0_edep_fraction;

  for (Long64_t entry = start_entry; entry < end_entry; ++entry) {
    tree.GetEntry(entry);
    auto* event = dynamic_cast<CEventMc*>(event_base);
    if (!event) continue;
    ++events_seen;

    const unsigned int n_tracks = event->GetNTracks();
    ntracks_count[n_tracks]++;
    if (n_tracks == 0) {
      ++no_track;
      continue;
    }
    if (n_tracks > 1) ++events_with_secondary_tracks;

    long long all_hits = 0;
    long long t0_hits = 0;
    double all_edep = 0.0;
    double t0_edep = 0.0;
    bool track0_bad_volume = false;
    bool any_bad_volume = false;

    for (unsigned int i = 0; i < n_tracks; ++i) {
      CTrackMc* trk = event->GetTrack(i);
      if (!trk) continue;

      if (static_cast<int>(i) <= max_track_index) {
        update_summary(by_index[i], trk);
      }

      const auto edeps = trk->GetEnergyDeposition();
      const auto vids = trk->GetVolumeId();
      const auto hit_count = static_cast<long long>(edeps.size());
      const double edep = sum_vector(edeps);
      all_hits += hit_count;
      all_edep += edep;
      if (i == 0) {
        t0_hits = hit_count;
        t0_edep = edep;
      }

      for (auto vid : vids) {
        const bool is_detector = GGeometryObject::IsTofVolume(vid) || GGeometryObject::IsTrackerVolume(vid);
        if (!is_detector) {
          any_bad_volume = true;
          if (i == 0) track0_bad_volume = true;
        }
      }
    }

    if (track0_bad_volume) ++events_track0_has_non_detector_hits;
    if (any_bad_volume) ++events_any_track_has_non_detector_hits;
    if (all_hits > 0) track0_hit_fraction.add(static_cast<double>(t0_hits) / static_cast<double>(all_hits));
    if (all_edep > 0.0) track0_edep_fraction.add(t0_edep / all_edep);
  }

  std::cout << "root: " << root_path << "\n";
  std::cout << "entries_total: " << entries_total << "\n";
  std::cout << "start_entry: " << start_entry << "\n";
  std::cout << "end_entry: " << end_entry << "\n";
  std::cout << "entries_looped: " << n_loop << "\n";
  std::cout << "events_seen: " << events_seen << "\n";
  std::cout << "no_track: " << no_track << "\n";
  std::cout << "events_with_GetTrack1plus: " << events_with_secondary_tracks << "\n";
  std::cout << "events_track0_has_non_tof_non_tracker_hits: " << events_track0_has_non_detector_hits << "\n";
  std::cout << "events_any_track_has_non_tof_non_tracker_hits: " << events_any_track_has_non_detector_hits << "\n";

  std::cout << "\ntrack0_hit_fraction_of_all_tracks:\n";
  std::cout << "  n " << track0_hit_fraction.n << "\n";
  std::cout << "  mean " << track0_hit_fraction.mean() << "\n";
  std::cout << "  min " << track0_hit_fraction.min << "\n";
  std::cout << "  max " << track0_hit_fraction.max << "\n";

  std::cout << "\ntrack0_edep_fraction_of_all_tracks:\n";
  std::cout << "  n " << track0_edep_fraction.n << "\n";
  std::cout << "  mean " << track0_edep_fraction.mean() << "\n";
  std::cout << "  min " << track0_edep_fraction.min << "\n";
  std::cout << "  max " << track0_edep_fraction.max << "\n";

  std::cout << "\nnumber_of_tracks_per_event:\n";
  for (const auto& kv : ntracks_count) {
    std::cout << "  " << kv.first << " " << kv.second << "\n";
  }

  for (int i = 0; i <= max_track_index; ++i) {
    print_summary(i, by_index[i]);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " input.root [start_entry] [max_entries] [max_track_index]\n";
    return 1;
  }

  const char* root_path = argv[1];
  Long64_t start_entry = (argc >= 3) ? std::stoll(argv[2]) : 0;
  Long64_t max_entries = (argc >= 4) ? std::stoll(argv[3]) : 200000;
  int max_track_index = (argc >= 5) ? std::stoi(argv[4]) : 12;

  tree_mc_track_index_audit(root_path, start_entry, max_entries, max_track_index);
  return 0;
}
