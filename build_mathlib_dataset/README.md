# Lean Mathlib Graph Extraction Pipeline

A comprehensive pipeline for extracting and building dependency graphs from Lean 4 Mathlib with source code snippets.

## Overview

This project extracts declarations (theorems, definitions, etc.) from Lean 4's Mathlib and constructs a rich dependency graph that includes:
- Full qualified names and type signatures
- Source code snippets with accurate line ranges
- Dependency relationships between declarations
- File-level import structures
- Classification of declaration types

## Features

- **Parallel Extraction**: Multi-threaded processing for fast extraction from large codebases
- **Source Code Integration**: Enriches each constant with its actual Lean source code
- **Smart Snippet Detection**: Intelligent heuristics to extract complete declarations including docstrings and attributes
- **Dependency Analysis**: Tracks both explicit and implicit dependencies between declarations
- **File-level Graph**: Creates file nodes representing Lean modules and their imports
- **Comprehensive Metadata**: Statistics and metadata for quick inspection

## Directory Structure

```
.
├── run_p2_pipeline.sh          # Main entry point script
├── graph_pipeline/                     # Python pipeline modules
│   ├── run_full_pipeline.py    # Main pipeline orchestrator
│   ├── lake_extractor.py       # Lean extraction wrapper
│   ├── import_parser.py        # Import statement parser
│   └── toolkit_skiplist.py     # Skip list for common declarations
├── find_code/                   # Code snippet extraction
│   └── extract_lake_code.py    # Enriches constants with source code
└── LeanScripts/                 # Lean extraction scripts
    ├── ExtractConstants.lean    # Lean metaprogram for extraction
    ├── lakefile.lean            # Lake build configuration
    ├── lean-toolchain           # Lean version specification
    └── lake-manifest.json       # Dependency manifest
```

The sample graph is at `$REPO_ROOT/graph-demo/example_graph.json`.

## Dependencies

### System Requirements
- **Lean 4**: Version matching your target Mathlib (e.g., 4.22.0 for mathlib4-4.22.0)
- **Mathlib**: A local copy of mathlib4 with its own `lean-toolchain` and `lake-manifest.json`
- **Lake**: Lean's build tool (comes with Lean 4)
- **Python**: 3.8 or higher

**Important**: The pipeline uses the Lean environment from the `--lake-project-dir` parameter (your target Mathlib directory), not from `LeanScripts/`. The `LeanScripts/` directory only provides the `ExtractConstants.lean` script.

### Python Dependencies
No external Python packages required - uses only standard library modules:
- `json`, `logging`, `subprocess`, `pathlib`
- `argparse`, `dataclasses`, `concurrent.futures`
- `collections`, `datetime`, `re`, `time`

### Lean Dependencies
- **mathlib4**: Version 4.22.0 (tag `v4.22.0`, commit `79e94a09...`)
- **batteries**, **aesop**, **Qq**, **Cli**, etc.: Automatically resolved by Lake
- Full dependency list in your target Mathlib project's `lake-manifest.json`

## Setup

### 1. Install Lean 4

Install elan (Lean version manager):
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.profile  # or restart your terminal
```

Verify installation:
```bash
elan --version
lake --version
```

### 2. Clone and Build Mathlib4 (Target Library)

The pipeline extracts data from a local copy of Mathlib. Clone the specific version you want to analyze:

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
export MATHLIB_ROOT="$REPO_ROOT/build_mathlib_dataset/mathlib4-4.22.0"

# Clone mathlib4 repository into a deterministic location under REPO_ROOT
git clone https://github.com/leanprover-community/mathlib4.git "$MATHLIB_ROOT"
cd "$MATHLIB_ROOT"

# Checkout the v4.22.0 tag
git checkout v4.22.0

# Download pre-built cache (IMPORTANT: saves hours of compilation)
lake exe cache get

# Build mathlib (this ensures all dependencies are ready)
lake build
```

**Important Notes**:
- `lake exe cache get` downloads pre-compiled `.olean` files, saving hours of build time
- The build may still take 10-30 minutes to verify and finalize
- Requires ~5.5 GB disk space for Mathlib with cache

### 3. Verify Setup

After building, verify you have the correct structure:
```bash
ls "$MATHLIB_ROOT"/
# Should contain: Mathlib/, lakefile.lean, lean-toolchain, lake-manifest.json, .lake/

cat "$MATHLIB_ROOT/lean-toolchain"
# Should output: leanprover/lean4:v4.22.0
```

### Note on LeanScripts Directory

The `LeanScripts/` directory in this repository contains configuration files for Lean 4.22.0, but **these are NOT used during pipeline execution**. The pipeline uses the Lean environment from your target Mathlib directory (specified by `--lake-project-dir`). The `LeanScripts/` directory only provides the `ExtractConstants.lean` script file.

## Usage

All commands below are written with absolute script paths and do not depend on current working directory.

### Basic Usage

Use absolute-by-construction paths:

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
export MATHLIB_ROOT="$REPO_ROOT/build_mathlib_dataset/mathlib4-4.22.0"

bash "$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT"
```

### Full Command with All Options

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
export MATHLIB_ROOT="$REPO_ROOT/build_mathlib_dataset/mathlib4-4.22.0"

bash "$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT" \
  --output-root "$REPO_ROOT/build_mathlib_dataset/generated_nodes/mathlib4-4.22.0" \
  --graph-output "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0/graph.json" \
  --stats-output "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0/stats.json" \
  --version mathlib4-4.22.0 \
  --timeout 900 \
  --workers 8
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mathlib-source` | Path to Mathlib directory containing .lean files | Required |
| `--lake-project-dir` | Directory where `lake` commands run | Parent of mathlib-source |
| `--output-root` | Output directory for extracted nodes | `$REPO_ROOT/build_mathlib_dataset/generated_nodes/mathlib4-4.22.0-full-withcode` |
| `--graph-output` | Path to final graph JSON file | `$REPO_ROOT/build_mathlib_dataset/generated_graphs/lake_graph_with_code.json` |
| `--stats-output` | Path to statistics JSON file | `$REPO_ROOT/build_mathlib_dataset/generated_graphs/lake_graph_with_code_stats.json` |
| `--version` | Version label for metadata | `mathlib4-4.22.0-full` |
| `--timeout` | Timeout per file extraction (seconds) | 900 |
| `--workers` | Number of parallel workers | CPU count - 1 |
| `--limit` | Limit number of files (for testing) | None |
| `--skip-extract` | Skip Lean extraction step | - |
| `--skip-snippet` | Skip code snippet enrichment step | - |
| `--skip-graph` | Skip graph construction step | - |

### Example: Test Run on Limited Files

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
export MATHLIB_ROOT="$REPO_ROOT/build_mathlib_dataset/mathlib4-4.22.0"

bash "$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT" \
  --limit 10 \
  --workers 2
```

For a faster smoke test (about 1-2 minutes on a typical machine), use:

```bash
bash "$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT" \
  --limit 5 \
  --workers 2
```

## Pipeline Stages

The pipeline consists of three main stages:

### Stage 1: Lean Constant Extraction (--skip-extract to skip)

For each `.lean` file in Mathlib:
1. Invokes `lake env lean --run LeanScripts/ExtractConstants.lean <module>`
2. Extracts declarations with metadata (name, type, position, dependencies)
3. Outputs `lake_constants.json` for each module
4. Supports parallel processing with configurable workers

### Stage 2: Code Snippet Enrichment (--skip-snippet to skip)

Runs `find_code/extract_lake_code.py`:
1. Reads each `lake_constants.json`
2. Parses the corresponding `.lean` source file
3. Uses intelligent heuristics to extract complete code snippets
4. Includes docstrings, attributes, and multi-line declarations
5. Splits declarations into statement and proof parts
6. Outputs enriched `lake_constants_withcode.json`

### Stage 3: Graph Construction (--skip-graph to skip)

Executed by `graph_pipeline/run_full_pipeline.py`:
1. Loads all enriched constants
2. Creates file nodes from import statements
3. Builds edges between nodes based on dependencies
4. Classifies edge types (DEPENDS_ON, IMPORTS, CONTAINS, USES_*)
5. Generates final graph JSON with metadata
6. Produces statistics summary

### Statistics JSON

Provides counts and ratios for:
- Node types (theorem, definition, instance, etc.)
- Edge types (DEPENDS_ON, IMPORTS, CONTAINS, etc.)
- Skippable nodes/edges
- Processing metadata

## Advanced Usage

### Using Different Lean/Mathlib Versions

This project is configured for:
- **Lean 4.22.0** (language version)
- **Mathlib 4.22.0** (library version)

To use different versions:
- Update the target Mathlib checkout (`--lake-project-dir`) to the desired Lean/Mathlib version
- In that target Mathlib directory, run `lake update` and `lake build`
- Re-run this pipeline against the updated target directory

### Extending the Skip List

Edit `graph_pipeline/toolkit_skiplist.py` to add declarations that should be marked as skippable (e.g., basic logical operations).

### Custom Processing

You can run individual stages:

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
export MATHLIB_ROOT="$REPO_ROOT/build_mathlib_dataset/mathlib4-4.22.0"

# Extract constants only
python "$REPO_ROOT/build_mathlib_dataset/graph_pipeline/run_full_pipeline.py" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT" \
  --skip-snippet --skip-graph

# Extract code snippets only (after extraction)
python "$REPO_ROOT/build_mathlib_dataset/find_code/extract_lake_code.py" \
  --mathlib-root "$REPO_ROOT/build_mathlib_dataset/generated_nodes/output/Mathlib"

# Build graph only (after both previous steps)
python "$REPO_ROOT/build_mathlib_dataset/graph_pipeline/run_full_pipeline.py" \
  --mathlib-source "$MATHLIB_ROOT/Mathlib" \
  --lake-project-dir "$MATHLIB_ROOT" \
  --skip-extract --skip-snippet
```

## Troubleshooting

### "lake: command not found"
Ensure Lean 4 and Lake are installed and in your PATH. Run `elan update` to install.

### "`lake exe cache get` fails"
You can still continue by running `lake build` directly in the target Mathlib directory.

### "Module not found" errors during extraction
Make sure `lake build` completed successfully in your target Mathlib directory (`--lake-project-dir`).
Do not run `lake build` under `LeanScripts/`; that directory is only a script container in this submission package.

### Timeout errors
Increase `--timeout` value for complex modules that take longer to process.

### Memory issues
Reduce `--workers` count or process in batches using `--limit`.

## Performance Notes

- Full Mathlib extraction (~100k+ declarations) typically takes ~30 minutes depending on CPU cores
- Memory usage: ~2-4 GB for full extraction
- Disk space: ~500 MB for full graph output
- Parallelization scales well up to number of CPU cores

## Citation

If you use this tool in your research, please cite appropriately.

