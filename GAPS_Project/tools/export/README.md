# TreeMc export tools

## `skim_treemc_entries.cc`

Select arbitrary global `TChain` entries into one or more small ROOT files.
The tool sorts and deduplicates the entry list, reads the source chain once,
and distributes events round-robin across output shards. Each shard contains:

- the selected `TreeMc` entries;
- `SelectionMetadata` with the global entry, source-file index, local entry,
  and sorted-selection index;
- `GGeometry`, when present in the metadata source;
- `SimulationParameterTree`, when present in the metadata source.

The executable must load `libGAPSCommon` at process startup. On gp1, build it
from the repository root with:

```bash
source ~/setup_ynakagami_root.sh

MYSIM=$HOME/simpledet_treerec_fix
MYSRC=$MYSIM/SimpleDet
MYBUILD=$MYSIM/build

mkdir -p build/tools

g++ -std=c++17 -O2 \
  $(root-config --cflags) \
  -I"$MYSRC/common/include" \
  tools/export/skim_treemc_entries.cc \
  -L"$MYBUILD/common" \
  -Wl,--no-as-needed -lGAPSCommon -Wl,--as-needed \
  -Wl,-rpath,"$MYBUILD/common" \
  $(root-config --libs) \
  -o build/tools/skim_treemc_entries
```

Example using one input ROOT and six output shards:

```bash
build/tools/skim_treemc_entries \
  --input input.root \
  --entry-list selected.entries \
  --metadata-file input.root \
  --output-prefix /tmp/antip_selected1000 \
  --shards 6
```

For a continued `TreeMc`, repeat `--input` in chain order:

```bash
build/tools/skim_treemc_entries \
  --input dbar_part0.root \
  --input dbar_part1.root \
  --entry-list selected.entries \
  --metadata-file dbar_part0.root \
  --output-prefix /tmp/antid_selected1000 \
  --shards 6
```

Outputs are created with ROOT `CREATE` mode and are never overwritten.
