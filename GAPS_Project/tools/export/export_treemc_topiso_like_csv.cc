#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TChain.h"
#include "TGeoManager.h"
#include "TObjString.h"
#include "TVector3.h"

#include "CEventBase.hh"
#include "CEventMc.hh"
#include "CTrackBase.hh"
#include "GGeometry.hh"
#include "GGeometryObject.hh"

namespace {

struct Args {
  std::string input;
  std::string output;
  std::string geometry_file;
  long long max_events = -1;
  long long start_entry = 0;
  int target_label = -1;
};

void print_usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " --input ROOT_OR_GLOB --output CSV "
      << "[--geometry-file ROOT] [--max-events N] [--start-entry N] "
      << "[--target-label 0|1]\n\n"
      << "Export TreeMc events to a topiso1457-like CSV:\n"
      << "  col 0       : random seed\n"
      << "  col 1       : ROOT entry index\n"
      << "  col 2       : label (antiP=0, antiD=1)\n"
      << "  col 3       : at-rest flag, fixed to 0\n"
      << "  col 4       : generated primary beta\n"
      << "  col 5       : stopping layer\n"
      << "  col 6:1446  : 1440 Si(Li) fixed-grid energy channels\n"
      << "  col 1446:1457: 11 TOF/event features\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (key == "--input") {
      args.input = require_value("--input");
    } else if (key == "--output") {
      args.output = require_value("--output");
    } else if (key == "--geometry-file") {
      args.geometry_file = require_value("--geometry-file");
    } else if (key == "--max-events") {
      args.max_events = std::stoll(require_value("--max-events"));
    } else if (key == "--start-entry") {
      args.start_entry = std::stoll(require_value("--start-entry"));
    } else if (key == "--target-label") {
      args.target_label = std::stoi(require_value("--target-label"));
    } else if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      print_usage(argv[0]);
      std::exit(2);
    }
  }

  if (args.input.empty() || args.output.empty()) {
    print_usage(argv[0]);
    std::exit(2);
  }
  if (args.geometry_file.empty()) {
    args.geometry_file = args.input;
  }
  if (args.target_label != -1 && args.target_label != 0 && args.target_label != 1) {
    std::cerr << "--target-label must be 0, 1, or omitted\n";
    std::exit(2);
  }
  if (args.start_entry < 0) {
    std::cerr << "--start-entry must be non-negative\n";
    std::exit(2);
  }
  return args;
}

int label_from_pdg(int pdg) {
  if (pdg == -2212) return 0;
  if (pdg == -1000010020) return 1;
  return -1;
}

std::vector<int> build_tracker_channel_order(const std::string& geometry_source) {
  TGeoManager* geo = LoadGeometryRoot(geometry_source);
  if (!geo) {
    std::cerr << "failed to load geometry from " << geometry_source << "\n";
    std::exit(3);
  }
  GGeometryMapPtr geometry = UnpackGeometry(geo, true);

  GGeometryMapPtr active_strips = FilterMap(
      geometry, [](GGeometryObject g) -> bool {
        return g.IsActive() && g.GetType() == GGeometryType::STRIP;
      });

  std::vector<int> tracker_volume_id[60][2];

  for (GGeometryElement element : *active_strips) {
    const int volid = element.first;
    if (volid % 10 != 0) continue;

    const int iinfo = static_cast<int>(volid / 10000) % 1000;
    const int jinfo = static_cast<int>(volid / 100) % 100;
    const int i = static_cast<int>((iinfo / 100) * 6 + (iinfo % 100) / 6);
    const int j = (jinfo < 2) ? 0 : 1;

    if (0 <= i && i < 60 && 0 <= j && j < 2) {
      tracker_volume_id[i][j].push_back(volid);
    }
  }

  std::vector<int> order;
  order.reserve(1440);
  for (int i = 0; i < 60; ++i) {
    for (int j = 0; j < 2; ++j) {
      std::sort(tracker_volume_id[i][j].begin(), tracker_volume_id[i][j].end());
      if (tracker_volume_id[i][j].size() < 12) {
        std::cerr << "bad tracker channel map: i=" << i << " j=" << j
                  << " size=" << tracker_volume_id[i][j].size() << "\n";
        std::exit(3);
      }
      for (int k = 0; k < 12; ++k) {
        order.push_back(tracker_volume_id[i][j][k]);
      }
    }
  }

  if (order.size() != 1440) {
    std::cerr << "bad tracker channel count: " << order.size() << "\n";
    std::exit(3);
  }
  return order;
}

bool valid_tracker_segment(int hit_volid) {
  return (hit_volid % 100 < 8) &&
         ((hit_volid / 100) % 100 < 4) &&
         ((hit_volid / 10000) % 100 < 36);
}

int channel_volume_id(int hit_volid) {
  return static_cast<int>(static_cast<double>(hit_volid) / 100.0) * 100;
}

struct EventFeatures {
  bool stopped = false;
  bool toptrigger = false;
  int stop_layer = -1;
  int n_top_umbrella = 0;
  int n_top_cube = 0;
  double e_top_umbrella = 0.0;
  double e_top_cube = 0.0;
  double tof = 0.0;
  TVector3 p_top_cube;
  TVector3 p_top_umbrella;
};

EventFeatures compute_event_features(CTrackBase* primary) {
  EventFeatures out;

  bool hit_top_umbrella = false;
  bool hit_top_cube = false;
  double t_top_umbrella = 1e32;
  double t_top_cube = 1e32;
  int top_umbrella_id = 0;
  int top_cube_id = 0;

  const auto vids = primary->GetVolumeId();
  const auto edeps = primary->GetEnergyDeposition();
  const auto times = primary->GetGlobalTime();
  const auto positions = primary->GetPosition();
  const auto kinetic_energy = primary->GetKineticEnergy();
  const auto step_lengths = primary->GetStepLength();

  for (std::size_t k = 0; k < vids.size(); ++k) {
    const int volid = static_cast<int>(vids[k]);
    const double edep = edeps[k];
    const double time = times[k];
    const TVector3 position = positions[k];

    if (k < kinetic_energy.size() && kinetic_energy[k] == 0.0 &&
        GGeometryObject::IsTrackerVolume(volid)) {
      for (std::size_t p = 0; p < step_lengths.size(); ++p) {
        if (step_lengths[p] == 0.0) {
          out.stopped = true;
          out.stop_layer = (volid % 10000000) / 1000000;
          break;
        }
      }
    }

    if (GGeometryObject::IsUmbrellaVolume(volid) && volid / 1000000 == 100) {
      hit_top_umbrella = true;
      out.e_top_umbrella += edep;
      if (top_umbrella_id == 0 || top_umbrella_id != volid) {
        ++out.n_top_umbrella;
        top_umbrella_id = volid;
      }
      if (time < t_top_umbrella) {
        t_top_umbrella = time;
        out.p_top_umbrella = position;
      }
    }

    if (GGeometryObject::IsCubeVolume(volid) && volid / 1000000 == 110) {
      hit_top_cube = true;
      out.e_top_cube += edep;
      if (top_cube_id == 0 || top_cube_id != volid) {
        ++out.n_top_cube;
        top_cube_id = volid;
      }
      if (time < t_top_cube) {
        t_top_cube = time;
        out.p_top_cube = position;
      }
    }
  }

  out.toptrigger = hit_top_umbrella && hit_top_cube && (t_top_umbrella < t_top_cube);
  out.tof = t_top_cube - t_top_umbrella;
  return out;
}

std::map<int, double> collect_tracker_energy(CEventMc* event) {
  std::map<int, double> by_channel;

  for (unsigned int track_index = 0; track_index < event->GetNTracks(); ++track_index) {
    CTrackBase* track = event->GetTrack(track_index);
    const auto vids = track->GetVolumeId();
    const auto edeps = track->GetEnergyDeposition();

    for (std::size_t k = 0; k < vids.size(); ++k) {
      const int hit_volid = static_cast<int>(vids[k]);
      const double hit_edep = edeps[k];
      if (hit_edep <= 0.0) continue;
      if (!GGeometryObject::IsTrackerVolume(hit_volid)) continue;
      if (!valid_tracker_segment(hit_volid)) continue;
      by_channel[channel_volume_id(hit_volid)] += hit_edep;
    }
  }

  return by_channel;
}

void write_event_row(std::ofstream& out,
                     CEventMc* event,
                     Long64_t entry,
                     int label,
                     const EventFeatures& feat,
                     const std::vector<int>& tracker_order,
                     const std::map<int, double>& tracker_energy) {
  out << event->GetRandSeed() << ","
      << entry << ","
      << label << ","
      << 0 << ","
      << event->GetPrimaryBetaGenerated() << ","
      << feat.stop_layer << ",";

  for (const int channel : tracker_order) {
    const auto it = tracker_energy.find(channel);
    out << (it == tracker_energy.end() ? 0.0 : it->second) << ",";
  }

  out << feat.n_top_umbrella << ","
      << feat.n_top_cube << ","
      << feat.e_top_umbrella << ","
      << feat.e_top_cube << ","
      << feat.tof << ","
      << feat.p_top_cube.x() << ","
      << feat.p_top_cube.y() << ","
      << feat.p_top_cube.z() << ","
      << feat.p_top_umbrella.x() << ","
      << feat.p_top_umbrella.y() << ","
      << feat.p_top_umbrella.z()
      << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);

  const std::vector<int> tracker_order = build_tracker_channel_order(args.geometry_file);

  TChain tree("TreeMc");
  tree.Add(args.input.c_str());

  CEventBase* event_base = new CEventMc;
  tree.SetBranchAddress("Mc", &event_base);

  const Long64_t n_entries = tree.GetEntries();
  if (n_entries <= 0) {
    std::cerr << "no entries: " << args.input << "\n";
    return 1;
  }

  std::ofstream out(args.output);
  if (!out) {
    std::cerr << "cannot open output: " << args.output << "\n";
    return 1;
  }

  long long seen = 0;
  long long written = 0;
  long long no_track = 0;
  long long not_target = 0;
  long long not_requested_label = 0;
  long long not_toptrigger = 0;
  long long not_stopped = 0;

  for (Long64_t entry = args.start_entry; entry < n_entries; ++entry) {
    if (args.max_events >= 0 && written >= args.max_events) break;

    tree.GetEntry(entry);
    auto* event = dynamic_cast<CEventMc*>(event_base);
    if (!event || event->GetNTracks() == 0) {
      ++no_track;
      continue;
    }
    ++seen;

    CTrackBase* primary = event->GetTrack(0);
    const int label = label_from_pdg(primary->GetPdg());
    if (label < 0) {
      ++not_target;
      continue;
    }
    if (args.target_label != -1 && label != args.target_label) {
      ++not_requested_label;
      continue;
    }

    const EventFeatures feat = compute_event_features(primary);
    if (!feat.toptrigger) {
      ++not_toptrigger;
      continue;
    }
    if (!feat.stopped) {
      ++not_stopped;
      continue;
    }

    const std::map<int, double> tracker_energy = collect_tracker_energy(event);
    write_event_row(out, event, entry, label, feat, tracker_order, tracker_energy);
    ++written;
  }

  std::cerr << "input: " << args.input << "\n";
  std::cerr << "output: " << args.output << "\n";
  std::cerr << "entries_total: " << n_entries << "\n";
  std::cerr << "start_entry: " << args.start_entry << "\n";
  std::cerr << "events_seen: " << seen << "\n";
  std::cerr << "written_atrest_toptrigger: " << written << "\n";
  std::cerr << "no_track: " << no_track << "\n";
  std::cerr << "not_target: " << not_target << "\n";
  std::cerr << "not_requested_label: " << not_requested_label << "\n";
  std::cerr << "not_toptrigger: " << not_toptrigger << "\n";
  std::cerr << "not_stopped: " << not_stopped << "\n";

  return 0;
}
