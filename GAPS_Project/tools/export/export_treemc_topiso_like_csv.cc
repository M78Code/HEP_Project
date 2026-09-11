#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

#include "TChain.h"
#include "TChainElement.h"
#include "TGeoManager.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TTree.h"
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
  std::string output_npy_dir;
  std::string geometry_file;
  std::string selection = "stopped-toptrigger";
  long long max_events = -1;
  long long start_entry = 0;
  int target_label = -1;
  bool provenance_only = false;
};

void print_usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0
      << " --input ROOT_OR_GLOB (--output CSV | --output-npy-dir DIR) "
      << "[--geometry-file ROOT] [--max-events N] [--start-entry N] "
      << "[--target-label 0|1] "
      << "[--selection none|toptrigger|stopped|stopped-toptrigger] "
      << "[--provenance-only]\n\n"
      << "Export TreeMc events to a topiso1457-like CSV:\n"
      << "  col 0       : random seed\n"
      << "  col 1       : ROOT entry index\n"
      << "  col 2       : label (antiP=0, antiD=1)\n"
      << "  col 3       : at-rest flag, fixed to 0\n"
      << "  col 4       : generated primary beta\n"
      << "  col 5       : stopping layer\n"
      << "  col 6:1446  : 1440 Si(Li) fixed-grid energy channels\n"
      << "  col 1446:1457: 11 TOF/event features\n\n"
      << "The direct NPY mode writes voxels.npy, tof_primary.npy, labels.npy,\n"
      << "betas.npy, and provenance arrays without a CSV intermediate. It\n"
      << "requires a positive --max-events value. --provenance-only omits\n"
      << "the voxel and TOF feature arrays.\n";
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
    } else if (key == "--output-npy-dir") {
      args.output_npy_dir = require_value("--output-npy-dir");
    } else if (key == "--geometry-file") {
      args.geometry_file = require_value("--geometry-file");
    } else if (key == "--selection") {
      args.selection = require_value("--selection");
    } else if (key == "--max-events") {
      args.max_events = std::stoll(require_value("--max-events"));
    } else if (key == "--start-entry") {
      args.start_entry = std::stoll(require_value("--start-entry"));
    } else if (key == "--target-label") {
      args.target_label = std::stoi(require_value("--target-label"));
    } else if (key == "--provenance-only") {
      args.provenance_only = true;
    } else if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      print_usage(argv[0]);
      std::exit(2);
    }
  }

  if (args.input.empty() || (args.output.empty() == args.output_npy_dir.empty())) {
    print_usage(argv[0]);
    std::exit(2);
  }
  if (!args.output_npy_dir.empty() && args.max_events <= 0) {
    std::cerr << "--output-npy-dir requires --max-events greater than zero\n";
    std::exit(2);
  }
  if (args.provenance_only && args.output_npy_dir.empty()) {
    std::cerr << "--provenance-only requires --output-npy-dir\n";
    std::exit(2);
  }
  if (args.selection != "none" && args.selection != "toptrigger" &&
      args.selection != "stopped" &&
      args.selection != "stopped-toptrigger") {
    std::cerr << "invalid --selection: " << args.selection << "\n";
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

std::vector<std::string> chain_source_files(TChain& tree) {
  std::vector<std::string> paths;
  TObjArray* files = tree.GetListOfFiles();
  if (files == nullptr) return paths;
  paths.reserve(static_cast<std::size_t>(files->GetEntries()));
  for (int index = 0; index < files->GetEntries(); ++index) {
    auto* element = dynamic_cast<TChainElement*>(files->At(index));
    if (element != nullptr) paths.emplace_back(element->GetTitle());
  }
  return paths;
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

std::string npy_shape(const std::vector<std::size_t>& shape) {
  std::ostringstream out;
  out << "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out << ", ";
    out << shape[i];
  }
  if (shape.size() == 1) out << ",";
  out << ")";
  return out.str();
}

class NpyStream {
 public:
  NpyStream(const std::string& path,
            const std::string& dtype,
            const std::vector<std::size_t>& shape)
      : out_(path, std::ios::binary) {
    if (!out_) throw std::runtime_error("cannot open " + path);

    const std::string dict = "{'descr': '" + dtype +
                             "', 'fortran_order': False, 'shape': " +
                             npy_shape(shape) + ", }";
    constexpr std::size_t prefix_size = 10;
    const std::size_t remainder = (prefix_size + dict.size() + 1) % 16;
    const std::size_t padding = remainder == 0 ? 0 : 16 - remainder;
    const std::string header = dict + std::string(padding, ' ') + "\n";
    if (header.size() > 65535) {
      throw std::runtime_error("NPY header is too large");
    }

    const char magic[] = {static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y'};
    out_.write(magic, sizeof(magic));
    const char version[] = {1, 0};
    out_.write(version, sizeof(version));
    const std::uint16_t length = static_cast<std::uint16_t>(header.size());
    const char length_bytes[] = {
        static_cast<char>(length & 0xff),
        static_cast<char>((length >> 8) & 0xff),
    };
    out_.write(length_bytes, sizeof(length_bytes));
    out_.write(header.data(), static_cast<std::streamsize>(header.size()));
  }

  template <typename T>
  void write(const T* data, std::size_t count) {
    out_.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(sizeof(T) * count));
    if (!out_) throw std::runtime_error("failed while writing NPY data");
  }

  template <typename T>
  void write_scalar(const T value) {
    write(&value, 1);
  }

  void flush() {
    out_.flush();
    if (!out_) throw std::runtime_error("failed while flushing NPY data");
  }

 private:
  std::ofstream out_;
};

float legacy_csv_float(double value) {
  char buffer[64];
  const int length = std::snprintf(buffer, sizeof(buffer), "%.6g", value);
  if (length <= 0 || length >= static_cast<int>(sizeof(buffer))) {
    throw std::runtime_error("failed to reproduce legacy CSV precision");
  }
  return std::strtof(buffer, nullptr);
}

std::string join_path(const std::string& directory, const std::string& name) {
  return directory.empty() || directory.back() == '/'
             ? directory + name
             : directory + "/" + name;
}

class DirectNpyOutput {
 public:
  DirectNpyOutput(const std::string& output_dir,
                  std::size_t n_events,
                  const std::vector<int>& tracker_order,
                  bool provenance_only)
      : output_dir_(output_dir),
        expected_(n_events),
        provenance_only_(provenance_only),
        labels_(join_path(output_dir, "labels.npy"), "<i8", {n_events}),
        betas_(join_path(output_dir, "betas.npy"), "<f4", {n_events}),
        random_seeds_(join_path(output_dir, "random_seeds.npy"), "<i8", {n_events}),
        chain_entries_(join_path(output_dir, "chain_entries.npy"), "<i8", {n_events}),
        source_file_indices_(join_path(output_dir, "source_file_indices.npy"), "<i4", {n_events}),
        source_entries_(join_path(output_dir, "source_entries.npy"), "<i8", {n_events}) {
    if (!provenance_only_) {
      voxels_ = std::make_unique<NpyStream>(
          join_path(output_dir, "voxels.npy"),
          "<f4",
          std::vector<std::size_t>{n_events, 10, 12, 12});
      tof_primary_ = std::make_unique<NpyStream>(
          join_path(output_dir, "tof_primary.npy"),
          "<f4",
          std::vector<std::size_t>{n_events, 11});
    }
    for (std::size_t i = 0; i < tracker_order.size(); ++i) {
      channel_indices_.emplace(tracker_order[i], i);
    }
  }

  static void prepare_directory(const std::string& output_dir) {
    struct stat info {};
    if (stat(output_dir.c_str(), &info) == 0) {
      throw std::runtime_error("output directory already exists: " + output_dir);
    }
    if (errno != ENOENT) {
      throw std::runtime_error("cannot inspect output directory: " + output_dir +
                               ": " + std::strerror(errno));
    }
    if (mkdir(output_dir.c_str(), 0775) != 0) {
      throw std::runtime_error("cannot create output directory: " + output_dir +
                               ": " + std::strerror(errno));
    }
  }

  void write(CEventMc* event,
             Long64_t chain_entry,
             Long64_t source_entry,
             int source_file_index,
             int label,
             const EventFeatures& feat,
             const std::map<int, double>& tracker_energy) {
    if (written_ >= expected_) {
      throw std::runtime_error("attempted to write too many NPY events");
    }

    if (!provenance_only_) {
      std::array<float, 1440> voxel{};
      for (const auto& item : tracker_energy) {
        const auto index = channel_indices_.find(item.first);
        if (index != channel_indices_.end()) {
          voxel[index->second] = legacy_csv_float(item.second);
        }
      }

      const std::array<float, 11> tof = {
          static_cast<float>(feat.n_top_umbrella),
          static_cast<float>(feat.n_top_cube),
          legacy_csv_float(feat.e_top_umbrella),
          legacy_csv_float(feat.e_top_cube),
          legacy_csv_float(feat.tof),
          legacy_csv_float(feat.p_top_cube.x()),
          legacy_csv_float(feat.p_top_cube.y()),
          legacy_csv_float(feat.p_top_cube.z()),
          legacy_csv_float(feat.p_top_umbrella.x()),
          legacy_csv_float(feat.p_top_umbrella.y()),
          legacy_csv_float(feat.p_top_umbrella.z()),
      };

      voxels_->write(voxel.data(), voxel.size());
      tof_primary_->write(tof.data(), tof.size());
    }
    labels_.write_scalar<std::int64_t>(label);
    betas_.write_scalar<float>(legacy_csv_float(event->GetPrimaryBetaGenerated()));
    random_seeds_.write_scalar<std::int64_t>(event->GetRandSeed());
    chain_entries_.write_scalar<std::int64_t>(chain_entry);
    source_file_indices_.write_scalar<std::int32_t>(source_file_index);
    source_entries_.write_scalar<std::int64_t>(source_entry);
    ++written_;
  }

  std::size_t written() const { return written_; }
  std::size_t expected() const { return expected_; }
  bool provenance_only() const { return provenance_only_; }

  void mark_complete(const Args& args,
                     const std::vector<std::string>& source_files) {
    if (!provenance_only_) {
      voxels_->flush();
      tof_primary_->flush();
    }
    labels_.flush();
    betas_.flush();
    random_seeds_.flush();
    chain_entries_.flush();
    source_file_indices_.flush();
    source_entries_.flush();

    std::ofstream manifest(join_path(output_dir_, "export_manifest.json"));
    if (!manifest) throw std::runtime_error("cannot write export manifest");
    manifest << "{\n"
             << "  \"source\": \"TreeMc direct fixed-grid export\",\n"
             << "  \"input\": \"" << args.input << "\",\n"
             << "  \"geometry_file\": \"" << args.geometry_file << "\",\n"
             << "  \"events\": " << written_ << ",\n"
             << "  \"target_label\": " << args.target_label << ",\n"
             << "  \"selection\": \"" << args.selection << "\",\n"
             << "  \"provenance_only\": "
             << (provenance_only_ ? "true" : "false") << ",\n"
             << "  \"start_entry\": " << args.start_entry << ",\n"
             << "  \"source_files\": " << source_files.size() << ",\n"
             << "  \"legacy_csv_significant_digits\": 6,\n"
             << "  \"voxel_shape\": [10, 12, 12],\n"
             << "  \"tof_primary_features\": 11\n"
             << "}\n";
    std::ofstream source_list(join_path(output_dir_, "source_files.txt"));
    if (!source_list) throw std::runtime_error("cannot write source file list");
    for (const std::string& path : source_files) source_list << path << "\n";
    std::ofstream success(join_path(output_dir_, "_SUCCESS"));
    if (!success) throw std::runtime_error("cannot write success marker");
  }

 private:
  std::string output_dir_;
  std::size_t expected_;
  bool provenance_only_;
  std::size_t written_ = 0;
  std::unordered_map<int, std::size_t> channel_indices_;
  std::unique_ptr<NpyStream> voxels_;
  std::unique_ptr<NpyStream> tof_primary_;
  NpyStream labels_;
  NpyStream betas_;
  NpyStream random_seeds_;
  NpyStream chain_entries_;
  NpyStream source_file_indices_;
  NpyStream source_entries_;
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

bool passes_selection(const EventFeatures& features,
                      const std::string& selection) {
  if (selection == "none") return true;
  if (selection == "toptrigger") return features.toptrigger;
  if (selection == "stopped") return features.stopped;
  return features.stopped && features.toptrigger;
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
  const std::vector<std::string> source_files = chain_source_files(tree);

  CEventBase* event_base = new CEventMc;
  tree.SetBranchAddress("Mc", &event_base);

  const Long64_t n_entries = tree.GetEntries();
  if (n_entries <= 0) {
    std::cerr << "no entries: " << args.input << "\n";
    return 1;
  }

  std::ofstream csv_out;
  std::unique_ptr<DirectNpyOutput> npy_out;
  if (!args.output.empty()) {
    csv_out.open(args.output);
    if (!csv_out) {
      std::cerr << "cannot open output: " << args.output << "\n";
      return 1;
    }
  } else {
    try {
      DirectNpyOutput::prepare_directory(args.output_npy_dir);
      npy_out = std::make_unique<DirectNpyOutput>(
          args.output_npy_dir,
          static_cast<std::size_t>(args.max_events),
          tracker_order,
          args.provenance_only);
    } catch (const std::exception& error) {
      std::cerr << "cannot initialize direct NPY output: " << error.what() << "\n";
      return 1;
    }
  }

  long long seen = 0;
  long long written = 0;
  long long no_track = 0;
  long long not_target = 0;
  long long not_requested_label = 0;
  long long not_toptrigger = 0;
  long long not_stopped = 0;
  long long failed_selection = 0;

  for (Long64_t entry = args.start_entry; entry < n_entries; ++entry) {
    if (args.max_events >= 0 && written >= args.max_events) break;

    tree.GetEntry(entry);
    const Long64_t source_entry =
        tree.GetTree() == nullptr ? -1 : tree.GetTree()->GetReadEntry();
    const int source_file_index = tree.GetTreeNumber();
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
    if (!feat.toptrigger) ++not_toptrigger;
    if (!feat.stopped) ++not_stopped;
    if (!passes_selection(feat, args.selection)) {
      ++failed_selection;
      continue;
    }

    std::map<int, double> tracker_energy;
    if (!npy_out || !npy_out->provenance_only()) {
      tracker_energy = collect_tracker_energy(event);
    }
    try {
      if (npy_out) {
        npy_out->write(event,
                       entry,
                       source_entry,
                       source_file_index,
                       label,
                       feat,
                       tracker_energy);
      } else {
        write_event_row(
            csv_out, event, entry, label, feat, tracker_order, tracker_energy);
      }
    } catch (const std::exception& error) {
      std::cerr << "failed to write entry " << entry << ": " << error.what() << "\n";
      return 1;
    }
    ++written;
  }

  if (npy_out) {
    if (npy_out->written() != npy_out->expected()) {
      std::cerr << "direct NPY output is incomplete: wrote " << npy_out->written()
                << " events, expected " << npy_out->expected() << "\n";
      return 1;
    }
    try {
      npy_out->mark_complete(args, source_files);
    } catch (const std::exception& error) {
      std::cerr << "cannot finalize direct NPY output: " << error.what() << "\n";
      return 1;
    }
  }

  std::cerr << "input: " << args.input << "\n";
  std::cerr << "output: "
            << (args.output.empty() ? args.output_npy_dir : args.output) << "\n";
  std::cerr << "entries_total: " << n_entries << "\n";
  std::cerr << "start_entry: " << args.start_entry << "\n";
  std::cerr << "selection: " << args.selection << "\n";
  std::cerr << "provenance_only: " << args.provenance_only << "\n";
  std::cerr << "events_seen: " << seen << "\n";
  std::cerr << "written_selected: " << written << "\n";
  std::cerr << "no_track: " << no_track << "\n";
  std::cerr << "not_target: " << not_target << "\n";
  std::cerr << "not_requested_label: " << not_requested_label << "\n";
  std::cerr << "not_toptrigger: " << not_toptrigger << "\n";
  std::cerr << "not_stopped: " << not_stopped << "\n";
  std::cerr << "failed_selection: " << failed_selection << "\n";

  return 0;
}
