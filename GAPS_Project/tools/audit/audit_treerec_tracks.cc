#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CEventBase.hh"
#include "CEventRec.hh"
#include "CTrackRec.hh"
#include "TFile.h"
#include "TTree.h"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " input.root [max_entries]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const Long64_t max_entries = argc == 3 ? std::stoll(argv[2]) : 10000;

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

  CEventBase* event_base = new CEventRec;
  if (tree->SetBranchAddress("Rec", &event_base) < 0) {
    std::cerr << "failed to bind top-level Rec branch\n";
    return 4;
  }

  const Long64_t entries_total = tree->GetEntries();
  const Long64_t entries_requested = std::min(entries_total, max_entries);
  Long64_t events_read = 0;
  Long64_t events_with_reco = 0;
  Long64_t events_with_tracks = 0;
  Long64_t total_track_ptrs = 0;
  Long64_t null_track_ptrs = 0;
  std::map<std::string, Long64_t> reco_events;
  std::map<std::string, Long64_t> reco_events_with_tracks;
  std::map<std::string, Long64_t> reco_track_ptrs;
  std::vector<std::uint32_t> event_track_counts;
  event_track_counts.reserve(static_cast<std::size_t>(entries_requested));

  for (Long64_t entry = 0; entry < entries_requested; ++entry) {
    if (tree->GetEntry(entry) <= 0) {
      continue;
    }
    ++events_read;

    auto* event = dynamic_cast<CEventRec*>(event_base);
    if (event == nullptr) {
      std::cerr << "Rec object is not CEventRec at entry " << entry << "\n";
      return 5;
    }

    const auto reco_names = event->ListAvailableReconstructions();
    if (!reco_names.empty()) {
      ++events_with_reco;
    }

    std::uint32_t event_tracks = 0;
    for (const auto& reco_name : reco_names) {
      event->ChooseReconstruction(reco_name, true);
      const auto n_tracks = static_cast<std::uint32_t>(event->GetNTracks());
      ++reco_events[reco_name];
      reco_track_ptrs[reco_name] += n_tracks;
      event_tracks += n_tracks;
      if (n_tracks > 0) {
        ++reco_events_with_tracks[reco_name];
      }
      for (std::uint32_t index = 0; index < n_tracks; ++index) {
        if (event->GetTrack(index) == nullptr) {
          ++null_track_ptrs;
        }
      }
    }

    total_track_ptrs += event_tracks;
    if (event_tracks > 0) {
      ++events_with_tracks;
    }
    event_track_counts.push_back(event_tracks);
  }

  std::sort(event_track_counts.begin(), event_track_counts.end());
  const auto min_tracks = event_track_counts.empty() ? 0U : event_track_counts.front();
  const auto median_tracks = event_track_counts.empty()
      ? 0U
      : event_track_counts[event_track_counts.size() / 2];
  const auto max_tracks = event_track_counts.empty() ? 0U : event_track_counts.back();

  std::cout << "input: " << input_path << "\n";
  std::cout << "TreeRec entries_total: " << entries_total << "\n";
  std::cout << "entries_requested: " << entries_requested << "\n";
  std::cout << "events_read: " << events_read << "\n";
  std::cout << "events_with_reco: " << events_with_reco << "\n";
  std::cout << "events_with_reco_fraction: "
            << (events_read == 0 ? 0.0
                                 : static_cast<double>(events_with_reco) / events_read)
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
  std::cout << "reconstruction_summary:\n";
  for (const auto& [name, events] : reco_events) {
    std::cout << "  name=" << name << " events=" << events
              << " events_with_tracks=" << reco_events_with_tracks[name]
              << " track_ptrs=" << reco_track_ptrs[name] << "\n";
  }

  return 0;
}
