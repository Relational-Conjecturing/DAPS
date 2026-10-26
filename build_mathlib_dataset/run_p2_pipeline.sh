#!/usr/bin/env bash
# ============================================================================
# Lean Mathlib Graph Extraction Pipeline
# ============================================================================
#
# Usage:
#   ./run_p2_pipeline.sh --mathlib-source /absolute/mathlib4/Mathlib \
#                        --lake-project-dir /absolute/mathlib4
#
# Or configure the default paths below and run:
#   ./run_p2_pipeline.sh
#
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# EXTERNAL DEPENDENCY: Mathlib4 Source
# ============================================================================
# You need a local copy of mathlib4. To set it up:
#
#   git clone https://github.com/leanprover-community/mathlib4.git ../mathlib4-4.22.0
#   cd ../mathlib4-4.22.0
#   git checkout v4.22.0
#   lake exe cache get
#   lake build
#   cd -
#
# Then update the path below to point to your mathlib4 directory:
# ============================================================================
MATHLIB_SOURCE_DEFAULT="$REPO_ROOT/../mathlib4-4.22.0/Mathlib"

# ============================================================================
# Other default configurations (usually no need to modify)
# ============================================================================
LAKE_PROJECT_DIR_DEFAULT="$(dirname "$MATHLIB_SOURCE_DEFAULT")"
OUTPUT_ROOT_DEFAULT="$REPO_ROOT/generated_nodes/mathlib4-4.22.0-full-withcode"
GRAPH_OUTPUT_DEFAULT="$REPO_ROOT/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json"
STATS_OUTPUT_DEFAULT="$REPO_ROOT/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_stats.json"
VERSION_DEFAULT="mathlib4-4.22.0-full"
TIMEOUT_DEFAULT=900

WORKERS=""

# ============================================================================
# Parse command-line arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mathlib-source) MATHLIB_SOURCE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --lake-project-dir) LAKE_PROJECT_DIR="$2"; shift 2 ;;
    --graph-output) GRAPH_OUTPUT="$2"; shift 2 ;;
    --stats-output) STATS_OUTPUT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --skip-extract) SKIP_EXTRACT="--skip-extract"; shift ;;
    --skip-snippet) SKIP_SNIPPET="--skip-snippet"; shift ;;
    --skip-graph) SKIP_GRAPH="--skip-graph"; shift ;;
    --limit) LIMIT="--limit $2"; shift 2 ;;
    --workers) WORKERS="--workers $2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --mathlib-source PATH   Path to Mathlib directory (default: $MATHLIB_SOURCE_DEFAULT)"
      echo "  --lake-project-dir PATH Directory where lake commands run (default: parent of mathlib-source)"
      echo "  --output-root PATH      Output directory for extracted nodes"
      echo "  --graph-output PATH     Path to final graph JSON file"
      echo "  --stats-output PATH     Path to statistics JSON file"
      echo "  --version STRING        Version label for metadata"
      echo "  --timeout SECONDS       Timeout per file extraction (default: 900)"
      echo "  --workers N             Number of parallel workers"
      echo "  --limit N               Limit number of files (for testing)"
      echo "  --skip-extract          Skip Lean extraction step"
      echo "  --skip-snippet          Skip code snippet enrichment step"
      echo "  --skip-graph            Skip graph construction step"
      echo "  -h, --help              Show this help message"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

MATHLIB_SOURCE="${MATHLIB_SOURCE:-$MATHLIB_SOURCE_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
LAKE_PROJECT_DIR="${LAKE_PROJECT_DIR:-$LAKE_PROJECT_DIR_DEFAULT}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-$GRAPH_OUTPUT_DEFAULT}"
STATS_OUTPUT="${STATS_OUTPUT:-$STATS_OUTPUT_DEFAULT}"
VERSION="${VERSION:-$VERSION_DEFAULT}"
TIMEOUT="${TIMEOUT:-$TIMEOUT_DEFAULT}"
SKIP_EXTRACT="${SKIP_EXTRACT:-}"
SKIP_SNIPPET="${SKIP_SNIPPET:-}"
SKIP_GRAPH="${SKIP_GRAPH:-}"
LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-}"

# Validate effective mathlib source *after* CLI args are parsed.
if [[ ! -d "$MATHLIB_SOURCE" ]]; then
  echo "ERROR: Mathlib source not found at: $MATHLIB_SOURCE" >&2
  echo "" >&2
  echo "Please either:" >&2
  echo "  1. Clone mathlib4 to the expected location:" >&2
  echo "     git clone https://github.com/leanprover-community/mathlib4.git ../mathlib4-4.22.0" >&2
  echo "     cd ../mathlib4-4.22.0 && git checkout v4.22.0 && lake exe cache get && lake build" >&2
  echo "" >&2
  echo "  2. Or specify the path explicitly:" >&2
  echo "     ./run_p2_pipeline.sh --mathlib-source /absolute/mathlib4/Mathlib \\" >&2
  echo "                          --lake-project-dir /absolute/mathlib4" >&2
  echo "" >&2
  echo "  3. Or edit MATHLIB_SOURCE_DEFAULT in this script to point to your mathlib4." >&2
  exit 1
fi

mkdir -p "$(dirname "$GRAPH_OUTPUT")"
mkdir -p "$(dirname "$STATS_OUTPUT")"

CMD=(
  python "$REPO_ROOT/graph_pipeline/run_full_pipeline.py"
  --mathlib-source "$MATHLIB_SOURCE"
  --lake-project-dir "$LAKE_PROJECT_DIR"
  --output-root "$OUTPUT_ROOT"
  --graph-output "$GRAPH_OUTPUT"
  --stats-output "$STATS_OUTPUT"
  --version "$VERSION"
  --timeout "$TIMEOUT"
)

[[ -n "$SKIP_EXTRACT" ]] && CMD+=("$SKIP_EXTRACT")
[[ -n "$SKIP_SNIPPET" ]] && CMD+=("$SKIP_SNIPPET")
[[ -n "$SKIP_GRAPH" ]] && CMD+=("$SKIP_GRAPH")
[[ -n "$LIMIT" ]] && CMD+=($LIMIT)
[[ -n "$WORKERS" ]] && CMD+=($WORKERS)

echo "Running pipeline:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
