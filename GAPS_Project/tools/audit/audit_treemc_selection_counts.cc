#include <iomanip>
#include <iostream>
#include <string>

#include "TChain.h"

#include "CEventBase.hh"
#include "CEventMc.hh"
#include "CTrackBase.hh"
#include "GGeometryObject.hh"

namespace {

struct Args {
  std::string input;
  long long max_entries = 0;
  int target_label = -1;
  double beta_min = 0.2;
  double beta_max = 0.5;
};

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --input ROOT_FILE [--max-entries N] [--target-label 0|1]"
            << " [--beta-min X] [--beta-max X]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (key == "--input") {
      args.input = value("--input");
    } else if (key == "--max-entries") {
      args.max_entries = std::stoll(value("--max-entries"));
    } else if (key == "--target-label") {
      args.target_label = std::stoi(value("--target-label"));
    } else if (key == "--beta-min") {
      args.beta_min = std::stod(value("--beta-min"));
    } else if (key == "--beta-max") {
      args.beta_max = std::stod(value("--beta-max"));
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      usage(argv[0]);
      std::exit(2);
    }
  }

  if (args.input.empty()) {
    usage(argv[0]);
    std::exit(2);
  }
  if (args.target_label != -1 && args.target_label != 0 && args.target_label != 1) {
    std::cerr << "--target-label must be 0, 1, or omitted\n";
    std::exit(2);
  }
  return args;
}

int label_from_pdg(int pdg) {
  if (pdg == -2212) return 0;
  if (pdg == -1000010020) return 1;
  return -1;
}

bool is_stopped_in_tracker(CTrackBase* primary, int& stop_layer) {
  stop_layer = -1;

  const auto& volume_ids = primary->GetVolumeId();
  const auto& kinetic_energies = primary->GetKineticEnergy();
  const auto& step_lengths = primary->GetStepLength();

  bool has_zero_step_length = false;
  for (double step_length : step_lengths) {
    if (step_length == 0.0) {
      has_zero_step_length = true;
      break;
    }
  }
  if (!has_zero_step_length) return false;

  for (std::size_t i = 0; i < kinetic_energies.size() && i < volume_ids.size(); ++i) {
    const int volume_id = static_cast<int>(volume_ids[i]);
    if (kinetic_energies[i] == 0.0 && GGeometryObject::IsTrackerVolume(volume_id)) {
      stop_layer = (volume_id % 10000000) / 1000000;
      return true;
    }
  }
  return false;
}

bool has_top_trigger(CTrackBase* primary) {
  const auto& volume_ids = primary->GetVolumeId();
  const auto& times = primary->GetGlobalTime();

  bool hit_top_umbrella = false;
  bool hit_top_cube = false;
  double top_umbrella_time = 1e99;
  double top_cube_time = 1e99;

  for (std::size_t i = 0; i < volume_ids.size() && i < times.size(); ++i) {
    const int volume_id = static_cast<int>(volume_ids[i]);

    if (GGeometryObject::IsUmbrellaVolume(volume_id) && volume_id / 1000000 == 100) {
      hit_top_umbrella = true;
      if (times[i] < top_umbrella_time) top_umbrella_time = times[i];
    }

    if (GGeometryObject::IsCubeVolume(volume_id) && volume_id / 1000000 == 110) {
      hit_top_cube = true;
      if (times[i] < top_cube_time) top_cube_time = times[i];
    }
  }

  return hit_top_umbrella && hit_top_cube && (top_umbrella_time < top_cube_time);
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);

  TChain tree("TreeMc");
  tree.Add(args.input.c_str());

  CEventBase* event_base = new CEventMc;
  tree.SetBranchAddress("Mc", &event_base);

  const Long64_t entries_total = tree.GetEntries();
  const Long64_t entries_loop =
      (args.max_entries > 0 && args.max_entries < entries_total)
          ? args.max_entries
          : entries_total;

  long long events_seen = 0;
  long long no_track = 0;
  long long bad_getentry = 0;
  long long other_pdg = 0;
  long long selected_label = 0;
  long long beta_in_range = 0;
  long long stopped = 0;
  long long toptrigger = 0;
  long long stopped_toptrigger = 0;
  long long stopped_no_top = 0;
  long long top_no_stopped = 0;
  long long labels[2] = {0, 0};

  for (Long64_t entry = 0; entry < entries_loop; ++entry) {
    const int got = tree.GetEntry(entry);
    if (got <= 0) {
      ++bad_getentry;
      continue;
    }

    auto* event = dynamic_cast<CEventMc*>(event_base);
    if (!event || event->GetNTracks() == 0) {
      ++no_track;
      continue;
    }
    ++events_seen;

    CTrackBase* primary = event->GetTrack(0);
    const int label = label_from_pdg(primary->GetPdg());
    if (label < 0) {
      ++other_pdg;
      continue;
    }
    ++labels[label];
    if (args.target_label != -1 && label != args.target_label) continue;
    ++selected_label;

    const double beta = event->GetPrimaryBetaGenerated();
    if (!(args.beta_min < beta && beta < args.beta_max)) continue;
    ++beta_in_range;

    int stop_layer = -1;
    const bool is_stopped = is_stopped_in_tracker(primary, stop_layer);
    const bool is_top = has_top_trigger(primary);

    if (is_stopped) ++stopped;
    if (is_top) ++toptrigger;
    if (is_stopped && is_top) ++stopped_toptrigger;
    if (is_stopped && !is_top) ++stopped_no_top;
    if (!is_stopped && is_top) ++top_no_stopped;
  }

  std::cout << std::setprecision(12);
  std::cout << "input,entries_total,entries_loop,events_seen,no_track,bad_getentry,"
            << "other_pdg,label0,label1,selected_label,beta_in_range,stopped,"
            << "toptrigger,stopped_toptrigger,stopped_no_top,top_no_stopped\n";
  std::cout << args.input << ","
            << entries_total << ","
            << entries_loop << ","
            << events_seen << ","
            << no_track << ","
            << bad_getentry << ","
            << other_pdg << ","
            << labels[0] << ","
            << labels[1] << ","
            << selected_label << ","
            << beta_in_range << ","
            << stopped << ","
            << toptrigger << ","
            << stopped_toptrigger << ","
            << stopped_no_top << ","
            << top_no_stopped << "\n";

  return 0;
}
