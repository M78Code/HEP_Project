#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "TChain.h"

#include "CEventBase.hh"
#include "CEventMc.hh"
#include "CTrackBase.hh"
#include "GGeometry.hh"

namespace {

std::string file_id_from_path(const std::string& path) {
  std::string name = path;
  const size_t slash = name.find_last_of('/');
  if (slash != std::string::npos) {
    name = name.substr(slash + 1);
  }
  if (name.size() > 5 && name.substr(name.size() - 5) == ".root") {
    name = name.substr(0, name.size() - 5);
  }
  const size_t underscore = name.find_last_of('_');
  if (underscore != std::string::npos) {
    return name.substr(underscore + 1);
  }
  return name;
}

int label_from_pdg(const int pdg) {
  if (pdg == -2212) {
    return 0;
  }
  if (pdg == -1000010020) {
    return 1;
  }
  return -1;
}

int stoplayer_from_track(CTrackBase* track, bool& stopped_in_tracker) {
  stopped_in_tracker = false;
  int stoplayer = -1;

  const auto& volume_ids = track->GetVolumeId();
  const auto& kinetic_energies = track->GetKineticEnergy();
  const auto& step_lengths = track->GetStepLength();

  for (size_t i = 0; i < kinetic_energies.size() && i < volume_ids.size(); ++i) {
    const int volume_id = volume_ids.at(i);
    if (kinetic_energies.at(i) == 0 && GGeometryObject::IsTrackerVolume(volume_id)) {
      stoplayer = (volume_id % 10000000) / 1000000;
      stopped_in_tracker = true;
      return stoplayer;
    }
  }

  for (size_t i = 0; i < step_lengths.size() && i < volume_ids.size(); ++i) {
    const int volume_id = volume_ids.at(i);
    if (step_lengths.at(i) == 0 && GGeometryObject::IsTrackerVolume(volume_id)) {
      stoplayer = (volume_id % 10000000) / 1000000;
      stopped_in_tracker = true;
      return stoplayer;
    }
  }

  return stoplayer;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0]
              << " ROOT_PATH OUTPUT_CSV MAX_ENTRIES REQUIRE_STOPPED [META_CSV]\n"
              << "  MAX_ENTRIES=0 means all entries\n"
              << "  REQUIRE_STOPPED=1 keeps only events stopped in tracker\n"
              << "  META_CSV optionally writes: file_id,entry,label,stoplayer,beta,pdg,n_steps,primary_beta,generated_beta\n";
    return 1;
  }

  const std::string root_path = argv[1];
  const std::string output_path = argv[2];
  const Long64_t max_entries = std::stoll(argv[3]);
  const bool require_stopped = std::stoi(argv[4]) != 0;
  const std::string meta_path = argc >= 6 ? argv[5] : "";
  const std::string file_id = file_id_from_path(root_path);

  TChain tree("TreeMc");
  tree.Add(root_path.c_str());

  CEventBase* event = new CEventMc;
  tree.SetBranchAddress("Mc", &event);

  std::ofstream output(output_path);
  if (!output) {
    std::cerr << "cannot open output: " << output_path << "\n";
    return 1;
  }

  std::ofstream meta_output;
  if (!meta_path.empty()) {
    meta_output.open(meta_path);
    if (!meta_output) {
      std::cerr << "cannot open meta output: " << meta_path << "\n";
      return 1;
    }
  }

  const Long64_t total_entries = tree.GetEntries();
  const Long64_t entries_to_loop =
      (max_entries > 0 && max_entries < total_entries) ? max_entries : total_entries;

  Long64_t used_events = 0;
  Long64_t skipped_no_track = 0;
  Long64_t skipped_label = 0;
  Long64_t skipped_no_top = 0;
  Long64_t skipped_not_stopped = 0;
  Long64_t rows = 0;

  output << std::setprecision(12);
  meta_output << std::setprecision(12);

  for (Long64_t entry = 0; entry < entries_to_loop; ++entry) {
    tree.GetEntry(entry);

    if (event->GetNTracks() == 0) {
      ++skipped_no_track;
      continue;
    }

    CTrackBase* primary_track = event->GetTrack(0);
    const int label = label_from_pdg(primary_track->GetPdg());
    if (label < 0) {
      ++skipped_label;
      continue;
    }

    bool stopped_in_tracker = false;
    const int stoplayer = stoplayer_from_track(primary_track, stopped_in_tracker);
    if (require_stopped && !stopped_in_tracker) {
      ++skipped_not_stopped;
      continue;
    }

    const auto& volume_ids = primary_track->GetVolumeId();
    const auto& positions = primary_track->GetPosition();
    const auto& energy_depositions = primary_track->GetEnergyDeposition();
    const auto& times = primary_track->GetGlobalTime();

    bool hit_top_umbrella = false;
    bool hit_top_cube = false;
    double top_umbrella_time = 0.0;
    double top_cube_time = 0.0;

    for (size_t i = 0; i < volume_ids.size() && i < times.size(); ++i) {
      const int volume_id = volume_ids.at(i);

      if (!hit_top_umbrella && GGeometryObject::IsUmbrellaVolume(volume_id)) {
        hit_top_umbrella = true;
        top_umbrella_time = times.at(i);
      }

      if (!hit_top_cube && GGeometryObject::IsCubeVolume(volume_id)) {
        hit_top_cube = true;
        top_cube_time = times.at(i);
      }
    }

    if (!hit_top_umbrella || !hit_top_cube) {
      ++skipped_no_top;
      continue;
    }

    const double tof = top_cube_time - top_umbrella_time;
    ++used_events;

    if (meta_output) {
      const double primary_beta = event->GetPrimaryBeta();
      const double generated_beta = event->GetPrimaryBetaGenerated();
      const double beta = primary_beta > 0.0 ? primary_beta : generated_beta;
      meta_output << file_id << "," << entry << "," << label << "," << stoplayer << ","
                  << beta << "," << primary_track->GetPdg() << "," << volume_ids.size()
                  << "," << primary_beta << "," << generated_beta << "\n";
    }

    for (size_t i = 0;
         i < volume_ids.size() && i < positions.size() && i < energy_depositions.size();
         ++i) {
      const auto& position = positions.at(i);
      output << file_id << "," << entry << "," << label << "," << stoplayer << ","
             << volume_ids.at(i) << "," << position.x() << "," << position.y() << ","
             << position.z() << "," << tof << "," << energy_depositions.at(i) << "\n";
      ++rows;
    }
  }

  std::cerr << "root: " << root_path << "\n";
  std::cerr << "output: " << output_path << "\n";
  if (!meta_path.empty()) {
    std::cerr << "meta_output: " << meta_path << "\n";
  }
  std::cerr << "entries_total: " << total_entries << "\n";
  std::cerr << "entries_looped: " << entries_to_loop << "\n";
  std::cerr << "used_events: " << used_events << "\n";
  std::cerr << "rows: " << rows << "\n";
  std::cerr << "skipped_no_track: " << skipped_no_track << "\n";
  std::cerr << "skipped_label: " << skipped_label << "\n";
  std::cerr << "skipped_no_top: " << skipped_no_top << "\n";
  std::cerr << "skipped_not_stopped: " << skipped_not_stopped << "\n";

  return 0;
}
