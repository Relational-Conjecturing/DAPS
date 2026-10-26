#!/usr/bin/env bash
# Flash generation for manual/Goedel inputs; reuse the main run's first four baseline samples.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash run_formalizer_ablation.sh A.jsonl B.jsonl C.jsonl [--dry-run]"
  exit 0
fi
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash run_formalizer_ablation.sh A.jsonl B.jsonl C.jsonl [--dry-run]" >&2
  exit 2
fi
DRY_RUN="${DRY_RUN:-0}"
if [[ "${4:-}" == --dry-run ]]; then DRY_RUN=1; elif [[ $# -eq 4 ]]; then echo "Unknown argument: $4" >&2; exit 2; fi
export PYTHON="${PYTHON:-python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../../imo_runs/formalizer_outputs}"
export RUN_DATE="${RUN_DATE:-$(date +%Y%m%d_%H%M%S)}"
export SAMPLES=4
export DRY_RUN
STAGE="${STAGE:-all}"
if [[ ! -f "$1" || ! -f "$2" ]]; then echo "An input JSONL does not exist" >&2; exit 1; fi
if [[ "$DRY_RUN" != 1 && ( "$STAGE" == all || "$STAGE" == summarize ) ]]; then
  if [[ ! -f "$3" ]]; then echo "Missing baseline: $3" >&2; exit 1; fi
fi
INPUT_JSONL="$1" SELECTOR_LABEL=manual bash "$SCRIPT_DIR/run_evaluation.sh" flash
INPUT_JSONL="$2" SELECTOR_LABEL=goedel bash "$SCRIPT_DIR/run_evaluation.sh" flash
if [[ "$STAGE" != all && "$STAGE" != summarize ]]; then exit 0; fi
if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] formalizer summary reuses main-run baseline sample indices 0..3"
  exit 0
fi
manifest="$OUTPUT_ROOT/formalizer_runs_${RUN_DATE}.json"
"$PYTHON" - "$manifest" "$3" "$OUTPUT_ROOT" "$RUN_DATE" <<'PY'
import json
from pathlib import Path
import sys

path, baseline, output, run_date = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
runs = [{"model": "Gemini-3.6-Flash", "selector": "Gemini-3-Pro", "graded_file": str(baseline.resolve())}]
for selector, label in (("goedel", "Goedel-Formalizer-V2-32B"), ("manual", "Manual")):
    source = output / "gemini-3.6-flash/matched" / f"{selector}_{run_date}_graded.jsonl"
    if not source.is_file():
        raise SystemExit(f"Missing graded file: {source}")
    runs.append({"model": "Gemini-3.6-Flash", "selector": label, "graded_file": str(source.resolve())})
path.write_text(json.dumps({"samples": 4, "runs": runs}, indent=2) + "\n")
PY
"$PYTHON" "$SCRIPT_DIR/summarize_scores.py" --manifest "$manifest" \
  --output "$OUTPUT_ROOT/formalizer_summary_${RUN_DATE}.json"
