#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "CEventMc.hh"
#include "CTrackMc.hh"
#include "GGeometryObject.hh"

namespace fs = std::filesystem;

namespace {

constexpr double kMergeWindowNs = 5.0;

struct Deposit {
  uint32_t volume_id;
  double time;
  std::size_t index;
};

struct Group {
  uint32_t volume_id;
  double time;
  std::vector<std::size_t> members;
};

struct Summary {
  std::int64_t files = 0;
  std::int64_t events = 0;
  std::int64_t events_with_tof = 0;
  std::int64_t events_with_actual_merge = 0;
  std::int64_t events_with_result_difference = 0;
  std::int64_t events_with_grouping_difference = 0;
  std::int64_t events_with_time_only_difference = 0;
  std::int64_t tof_deposits = 0;
  std::int64_t actual_groups = 0;
  std::int64_t intended_groups = 0;
  std::int64_t nonfinite_tof_times = 0;
  double max_finite_time_difference = 0.0;

  Summary& operator+=(const Summary& other) {
    files += other.files;
    events += other.events;
    events_with_tof += other.events_with_tof;
    events_with_actual_merge += other.events_with_actual_merge;
    events_with_result_difference += other.events_with_result_difference;
    events_with_grouping_difference += other.events_with_grouping_difference;
    events_with_time_only_difference += other.events_with_time_only_difference;
    tof_deposits += other.tof_deposits;
    actual_groups += other.actual_groups;
    intended_groups += other.intended_groups;
    nonfinite_tof_times += other.nonfinite_tof_times;
    max_finite_time_difference =
        std::max(max_finite_time_difference, other.max_finite_time_difference);
    return *this;
  }
};

std::vector<Group> merge_hits(const std::vector<Deposit>& hits,
                              bool use_intended_earliest_time) {
  std::vector<Group> groups;
  std::vector<bool> consumed(hits.size(), false);

  for (std::size_t i = 0; i < hits.size(); ++i) {
    if (consumed[i]) continue;

    Group merged{hits[i].volume_id, hits[i].time, {hits[i].index}};
    for (std::size_t j = 0; j < hits.size(); ++j) {
      if (i == j || consumed[j]) continue;
      if (merged.volume_id != hits[j].volume_id) continue;
      const double delta_time = std::abs(merged.time - hits[j].time);
      if (!(delta_time <= kMergeWindowNs)) continue;

      if (use_intended_earliest_time) {
        merged.time = std::min(merged.time, hits[j].time);
      } else {
        // Reproduce GRecoHit::operator+: the right-hand hit time wins.
        merged.time = hits[j].time;
      }
      merged.members.push_back(hits[j].index);
      consumed[j] = true;
    }
    groups.push_back(std::move(merged));
  }
  return groups;
}

bool same_members(const std::vector<Group>& lhs,
                  const std::vector<Group>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].volume_id != rhs[i].volume_id ||
        lhs[i].members != rhs[i].members) {
      return false;
    }
  }
  return true;
}

bool same_time(double lhs, double rhs) {
  return (std::isnan(lhs) && std::isnan(rhs)) || lhs == rhs;
}

Summary scan_file(const fs::path& path, std::int64_t max_events) {
  TFile file(path.c_str(), "READ");
  if (file.IsZombie()) throw std::runtime_error("cannot open " + path.string());
  auto* tree = dynamic_cast<TTree*>(file.Get("TreeMc"));
  if (tree == nullptr) throw std::runtime_error("TreeMc missing in " + path.string());

  CEventMc* event = nullptr;
  if (tree->SetBranchAddress("Mc", &event) < 0) {
    throw std::runtime_error("Mc branch missing in " + path.string());
  }

  Summary summary;
  summary.files = 1;
  const auto entries = max_events < 0
                           ? tree->GetEntries()
                           : std::min<Long64_t>(tree->GetEntries(), max_events);

  for (Long64_t entry = 0; entry < entries; ++entry) {
    if (tree->GetEntry(entry) <= 0 || event == nullptr) {
      throw std::runtime_error("failed to read entry " +
                               std::to_string(entry) + " in " + path.string());
    }
    ++summary.events;
    std::vector<Deposit> deposits;

    for (const auto* track : event->GetTracks()) {
      const auto volume_ids = track->GetVolumeId();
      const auto energies = track->GetEnergyDeposition();
      const auto times = track->GetGlobalTime();
      if (volume_ids.size() != energies.size() ||
          volume_ids.size() != times.size()) {
        throw std::runtime_error("track-array length mismatch at entry " +
                                 std::to_string(entry) + " in " + path.string());
      }
      for (std::size_t i = 0; i < volume_ids.size(); ++i) {
        if (energies[i] == 0 ||
            !GGeometryObject::IsTofVolume(volume_ids[i])) {
          continue;
        }
        if (!std::isfinite(times[i])) ++summary.nonfinite_tof_times;
        deposits.push_back(
            {volume_ids[i], times[i], deposits.size()});
      }
    }

    summary.tof_deposits += deposits.size();
    if (deposits.empty()) continue;
    ++summary.events_with_tof;

    const auto actual = merge_hits(deposits, false);
    const auto intended = merge_hits(deposits, true);
    summary.actual_groups += actual.size();
    summary.intended_groups += intended.size();
    if (actual.size() < deposits.size()) ++summary.events_with_actual_merge;

    const bool members_equal = same_members(actual, intended);
    bool times_equal = members_equal;
    if (members_equal) {
      for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!same_time(actual[i].time, intended[i].time)) times_equal = false;
        if (std::isfinite(actual[i].time) && std::isfinite(intended[i].time)) {
          summary.max_finite_time_difference = std::max(
              summary.max_finite_time_difference,
              std::abs(actual[i].time - intended[i].time));
        }
      }
    }
    if (!members_equal || !times_equal) {
      ++summary.events_with_result_difference;
      if (!members_equal) {
        ++summary.events_with_grouping_difference;
      } else {
        ++summary.events_with_time_only_difference;
      }
    }
  }
  tree->ResetBranchAddresses();
  return summary;
}

double fraction(std::int64_t numerator, std::int64_t denominator) {
  return denominator == 0 ? 0.0
                          : static_cast<double>(numerator) / denominator;
}

void print_summary(const std::string& name, const Summary& s) {
  std::cout << std::fixed << std::setprecision(8)
            << "SUMMARY name=" << name
            << " files=" << s.files
            << " events=" << s.events
            << " events_with_tof=" << s.events_with_tof
            << " events_with_actual_merge=" << s.events_with_actual_merge
            << " merge_fraction_all="
            << fraction(s.events_with_actual_merge, s.events)
            << " events_with_result_difference="
            << s.events_with_result_difference
            << " difference_fraction_all="
            << fraction(s.events_with_result_difference, s.events)
            << " events_with_grouping_difference="
            << s.events_with_grouping_difference
            << " events_with_time_only_difference="
            << s.events_with_time_only_difference
            << " tof_deposits=" << s.tof_deposits
            << " actual_groups=" << s.actual_groups
            << " intended_groups=" << s.intended_groups
            << " nonfinite_tof_times=" << s.nonfinite_tof_times
            << " max_finite_time_difference_ns="
            << s.max_finite_time_difference << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0]
              << " INPUT_DIR [MAX_EVENTS_PER_FILE]\n";
    return 2;
  }

  const fs::path input_dir = argv[1];
  const std::int64_t max_events = argc == 3 ? std::stoll(argv[2]) : -1;
  if (!fs::is_directory(input_dir)) {
    std::cerr << "not a directory: " << input_dir << '\n';
    return 2;
  }

  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".root") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    std::cerr << "no ROOT files in " << input_dir << '\n';
    return 2;
  }

  try {
    Summary total;
    for (const auto& path : paths) {
      const auto file_summary = scan_file(path, max_events);
      print_summary(path.filename().string(), file_summary);
      total += file_summary;
    }
    print_summary("ALL", total);
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
