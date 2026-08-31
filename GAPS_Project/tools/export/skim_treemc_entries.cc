#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "TChain.h"
#include "TClass.h"
#include "TFile.h"
#include "TObject.h"
#include "TTree.h"

namespace {

struct Args {
  std::vector<std::string> inputs;
  std::string entry_list;
  std::string metadata_file;
  std::string output_prefix;
  std::string tree_name = "TreeMc";
  int shards = 1;
};

struct OutputShard {
  std::unique_ptr<TFile> file;
  TTree* event_tree = nullptr;
  TTree* selection_tree = nullptr;
  Long64_t source_entry = -1;
  Long64_t source_local_entry = -1;
  Long64_t selection_index = -1;
  Int_t source_file_index = -1;
  Long64_t written = 0;
  std::string path;
};

void print_usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0
      << " --input INPUT.root [--input CONTINUATION.root ...]"
      << " --entry-list ENTRIES.txt --output-prefix PATH"
      << " [--metadata-file INPUT.root] [--tree-name TreeMc]"
      << " [--shards N]\n\n"
      << "Select arbitrary global TChain entries and write one or more small "
      << "ROOT files. Entries are sorted and deduplicated before reading. "
      << "Each output contains the selected event tree, SelectionMetadata, "
      << "GGeometry, and SimulationParameterTree when available.\n";
}

std::string require_value(int argc, char** argv, int& index,
                          const char* option) {
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << option << "\n";
    std::exit(2);
  }
  return argv[++index];
}

Args parse_args(int argc, char** argv) {
  Args args;

  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--input") {
      args.inputs.push_back(require_value(argc, argv, i, "--input"));
    } else if (option == "--entry-list") {
      args.entry_list = require_value(argc, argv, i, "--entry-list");
    } else if (option == "--metadata-file") {
      args.metadata_file = require_value(argc, argv, i, "--metadata-file");
    } else if (option == "--output-prefix") {
      args.output_prefix = require_value(argc, argv, i, "--output-prefix");
    } else if (option == "--tree-name") {
      args.tree_name = require_value(argc, argv, i, "--tree-name");
    } else if (option == "--shards") {
      args.shards = std::stoi(require_value(argc, argv, i, "--shards"));
    } else if (option == "--help" || option == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown option: " << option << "\n";
      print_usage(argv[0]);
      std::exit(2);
    }
  }

  if (args.inputs.empty() || args.entry_list.empty() ||
      args.output_prefix.empty()) {
    print_usage(argv[0]);
    std::exit(2);
  }
  if (args.metadata_file.empty()) {
    args.metadata_file = args.inputs.front();
  }
  if (args.shards <= 0) {
    std::cerr << "--shards must be positive\n";
    std::exit(2);
  }
  return args;
}

std::vector<Long64_t> load_entries(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    std::cerr << "unable to open entry list: " << path << "\n";
    std::exit(3);
  }

  std::vector<Long64_t> entries;
  Long64_t entry = -1;
  while (stream >> entry) {
    if (entry < 0) {
      std::cerr << "negative entry in " << path << ": " << entry << "\n";
      std::exit(3);
    }
    entries.push_back(entry);
  }
  if (!stream.eof()) {
    std::cerr << "invalid entry-list content in " << path << "\n";
    std::exit(3);
  }
  if (entries.empty()) {
    std::cerr << "entry list is empty: " << path << "\n";
    std::exit(3);
  }

  std::sort(entries.begin(), entries.end());
  const auto unique_end = std::unique(entries.begin(), entries.end());
  const std::size_t duplicate_count =
      static_cast<std::size_t>(entries.end() - unique_end);
  entries.erase(unique_end, entries.end());

  if (duplicate_count > 0) {
    std::cout << "deduplicated entries: " << duplicate_count << "\n";
  }
  return entries;
}

std::string shard_path(const std::string& prefix, int shard, int shards) {
  int width = 2;
  for (int n = shards - 1; n >= 100; n /= 10) {
    ++width;
  }

  std::ostringstream path;
  path << prefix << "_shard" << std::setw(width) << std::setfill('0')
       << shard << ".root";
  return path.str();
}

void copy_metadata(TObject* geometry, TTree* simulation,
                   OutputShard& output) {
  if (geometry) {
    output.file->cd();
    geometry->Write("GGeometry", TObject::kOverwrite);
  }

  if (simulation) {
    output.file->cd();
    TTree* clone = simulation->CloneTree(-1, "fast");
    clone->Write("SimulationParameterTree", TObject::kOverwrite);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);

  if (!TClass::GetClass("CEventMc")) {
    std::cerr
        << "CEventMc dictionary is unavailable. Link this executable with "
        << "libGAPSCommon using --no-as-needed so the library is loaded at "
        << "process startup.\n";
    return 4;
  }

  const std::vector<Long64_t> entries = load_entries(args.entry_list);

  TChain input(args.tree_name.c_str());
  for (const std::string& path : args.inputs) {
    if (input.Add(path.c_str()) == 0) {
      std::cerr << "unable to add input: " << path << "\n";
      return 5;
    }
  }

  const Long64_t input_entries = input.GetEntries();
  if (input_entries <= 0) {
    std::cerr << "input chain is empty\n";
    return 5;
  }
  if (entries.back() >= input_entries) {
    std::cerr << "selected entry " << entries.back()
              << " is outside input range [0, " << input_entries << ")\n";
    return 5;
  }

  std::cout << "tree: " << args.tree_name << "\n";
  std::cout << "input files: " << args.inputs.size() << "\n";
  std::cout << "input entries: " << input_entries << "\n";
  std::cout << "selected entries: " << entries.size() << "\n";
  std::cout << "output shards: " << args.shards << "\n";

  std::vector<std::unique_ptr<OutputShard>> outputs;
  outputs.reserve(args.shards);

  for (int shard = 0; shard < args.shards; ++shard) {
    auto output = std::make_unique<OutputShard>();
    output->path = shard_path(args.output_prefix, shard, args.shards);
    output->file = std::make_unique<TFile>(output->path.c_str(), "CREATE");
    if (!output->file || output->file->IsZombie()) {
      std::cerr << "unable to create output (already exists?): "
                << output->path << "\n";
      return 6;
    }

    output->file->cd();
    output->event_tree = input.CloneTree(0);
    if (!output->event_tree) {
      std::cerr << "unable to clone input tree structure\n";
      return 6;
    }

    output->selection_tree = new TTree(
        "SelectionMetadata", "Source entries selected into this skim");
    output->selection_tree->Branch("source_entry", &output->source_entry);
    output->selection_tree->Branch("source_file_index",
                                   &output->source_file_index);
    output->selection_tree->Branch("source_local_entry",
                                   &output->source_local_entry);
    output->selection_tree->Branch("selection_index",
                                   &output->selection_index);
    outputs.push_back(std::move(output));
  }

  for (std::size_t index = 0; index < entries.size(); ++index) {
    const Long64_t selected = entries[index];
    const Long64_t local_entry = input.LoadTree(selected);
    if (local_entry < 0 || input.GetEntry(selected) <= 0) {
      std::cerr << "unable to read selected entry: " << selected << "\n";
      return 7;
    }

    OutputShard& output = *outputs[index % outputs.size()];
    output.source_entry = selected;
    output.source_file_index = input.GetTreeNumber();
    output.source_local_entry = local_entry;
    output.selection_index = static_cast<Long64_t>(index);

    output.event_tree->Fill();
    output.selection_tree->Fill();
    ++output.written;
  }

  TFile metadata(args.metadata_file.c_str(), "READ");
  if (metadata.IsZombie()) {
    std::cerr << "unable to open metadata file: " << args.metadata_file
              << "\n";
    return 8;
  }

  // TGeoManager is registered in ROOT global state. Fetch it once: repeated
  // TFile::Get calls can delete and recreate the active geometry.
  TObject* geometry = metadata.Get("GGeometry");
  if (!geometry) {
    std::cerr << "warning: GGeometry is missing from "
              << metadata.GetName() << "\n";
  }
  TTree* simulation = dynamic_cast<TTree*>(
      metadata.Get("SimulationParameterTree"));

  Long64_t total_written = 0;
  for (auto& output : outputs) {
    output->file->cd();
    output->event_tree->Write(args.tree_name.c_str(), TObject::kOverwrite);
    output->selection_tree->Write("SelectionMetadata", TObject::kOverwrite);
    copy_metadata(geometry, simulation, *output);

    total_written += output->written;
    std::cout << output->path << ": " << output->written << " entries\n";
  }

  // Old GGeometry files register objects in ROOT's global directory state.
  // Close their owning input file before output directories tear down.
  metadata.Close();

  for (auto& output : outputs) {
    output->file->Close();
  }

  if (total_written != static_cast<Long64_t>(entries.size())) {
    std::cerr << "entry-count mismatch: selected=" << entries.size()
              << " written=" << total_written << "\n";
    return 9;
  }

  std::cout << "complete: " << total_written << " entries\n";
  return 0;
}
