#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

class CTrackRec;

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " input.root [max_entries]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const Long64_t max_entries = argc == 3 ? std::stoll(argv[2]) : 10000;
  const char* branch_name = "Rec/Tracks/Tracks.second";

  TFile input(input_path.c_str(), "READ");
  if (input.IsZombie()) {
    std::cerr << "cannot open: " << input_path << "\n";
    return 2;
  }

  auto* tree = dynamic_cast<TTree*>(input.Get("TreeRec"));
  if (tree == nullptr) {
    std::cerr << "TreeRec not found: " << input_path << "\n";
    return 3;
  }
  if (tree->GetBranch(branch_name) == nullptr) {
    std::cerr << "branch not found: " << branch_name << "\n";
    return 4;
  }

  const Long64_t entries_total = tree->GetEntries();
  const Long64_t entries_requested = std::min(entries_total, max_entries);
  TTreeReader reader(tree);
  TTreeReaderValue<std::vector<CTrackRec*>> tracks(reader, branch_name);

  Long64_t events_read = 0;
  Long64_t events_with_tracks = 0;
  Long64_t total_track_ptrs = 0;
  Long64_t null_track_ptrs = 0;
  std::vector<std::uint32_t> track_counts;
  track_counts.reserve(static_cast<std::size_t>(entries_requested));

  while (events_read < entries_requested && reader.Next()) {
    const auto& event_tracks = *tracks;
    const auto count = static_cast<std::uint32_t>(event_tracks.size());
    track_counts.push_back(count);
    total_track_ptrs += count;
    if (count > 0) {
      ++events_with_tracks;
    }
    for (const auto* track : event_tracks) {
      if (track == nullptr) {
        ++null_track_ptrs;
      }
    }
    ++events_read;
  }

  if (events_read == 0 && entries_requested > 0) {
    std::cerr << "failed to read " << branch_name << "\n";
    return 5;
  }

  std::sort(track_counts.begin(), track_counts.end());
  const auto min_tracks = track_counts.empty() ? 0U : track_counts.front();
  const auto median_tracks = track_counts.empty()
      ? 0U
      : track_counts[track_counts.size() / 2];
  const auto max_tracks = track_counts.empty() ? 0U : track_counts.back();

  std::cout << "input: " << input_path << "\n";
  std::cout << "TreeRec entries_total: " << entries_total << "\n";
  std::cout << "entries_requested: " << entries_requested << "\n";
  std::cout << "events_read: " << events_read << "\n";
  std::cout << "CTrackRec dictionary: "
            << (TClass::GetClass("CTrackRec") != nullptr ? "available" : "missing")
            << "\n";
  std::cout << "events_with_tracks: " << events_with_tracks << "\n";
  std::cout << "events_with_tracks_fraction: "
            << (events_read == 0 ? 0.0
                                 : static_cast<double>(events_with_tracks) / events_read)
            << "\n";
  std::cout << "track_ptrs_total: " << total_track_ptrs << "\n";
  std::cout << "null_track_ptrs: " << null_track_ptrs << "\n";
  std::cout << "tracks_per_event_min_median_max: " << min_tracks << " "
            << median_tracks << " " << max_tracks << "\n";
  return 0;
}
