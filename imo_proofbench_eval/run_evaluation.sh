#!/usr/bin/env bash
# Paper pipeline: generation -> Gemini judge -> per-problem score aggregation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE=all
DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in
    all|qwen|deepseek|kimi|flash) MODEL_CHOICE="$arg" ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      echo "Usage: bash run_evaluation.sh [all|qwen|deepseek|kimi|flash] [--dry-run]"
      echo "Set INPUT_JSONL to your input file; OUTPUT_ROOT and RUN_DATE select the output run."
      echo "SELECTOR_LABEL is optional. STAGE=all|collect|grade|summarize (default all)."
      echo "Credentials: NOVITA_API_KEY for Novita; GEMINI_API_KEY for Flash and grading."
      exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-python}"
: "${INPUT_JSONL:?Set INPUT_JSONL to the JSONL file to evaluate}"
if [[ ! -f "$INPUT_JSONL" ]]; then echo "Input JSONL not found: $INPUT_JSONL" >&2; exit 1; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../../imo_runs/outputs}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d_%H%M%S)}"
STAGE="${STAGE:-all}"
SAMPLES="${SAMPLES:-8}"
TOP_K="${TOP_K:-16}"
if [[ -z "${SELECTOR_LABEL:-}" ]]; then
  SELECTOR_LABEL="${INPUT_JSONL##*/}"
  SELECTOR_LABEL="${SELECTOR_LABEL%.jsonl}"
fi
NOVITA_KEY_ENV="${NOVITA_KEY_ENV:-NOVITA_API_KEY}"
GEMINI_KEY_ENV="${GEMINI_KEY_ENV:-GEMINI_API_KEY}"
NOVITA_BASE_URL="${NOVITA_BASE_URL:-https://api.novita.ai/openai}"
GEMINI_BASE_URL="${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}"
case "$STAGE" in all|collect|grade|summarize) ;; *) echo "Invalid STAGE=$STAGE" >&2; exit 2 ;; esac
if [[ -z "$SELECTOR_LABEL" || "$SELECTOR_LABEL" == */* || "$SELECTOR_LABEL" == . || "$SELECTOR_LABEL" == .. ]]; then
  echo "SELECTOR_LABEL must be a single output label" >&2; exit 2
fi
models=("$MODEL_CHOICE")
if [[ "$MODEL_CHOICE" == all ]]; then models=(qwen deepseek kimi flash); fi

run() {
  printf '[command]'
  printf ' [%s]' "$@"
  printf '\n'
  "$@"
}

require_key() {
  if [[ -z "${!1:-}" ]]; then echo "Missing API key env: $1" >&2; exit 1; fi
}

if [[ "$DRY_RUN" != 1 ]]; then
  if [[ "$STAGE" == all || "$STAGE" == grade ]]; then require_key "$GEMINI_KEY_ENV"; fi
  if [[ "$STAGE" == all || "$STAGE" == collect ]]; then
    for alias in "${models[@]}"; do
      if [[ "$alias" == flash ]]; then require_key "$GEMINI_KEY_ENV"; else require_key "$NOVITA_KEY_ENV"; fi
    done
  fi
fi

echo "[config] models=${models[*]} selector=$SELECTOR_LABEL samples=$SAMPLES top_k=$TOP_K stage=$STAGE dry_run=$DRY_RUN"
echo "[paths] input=$INPUT_JSONL output=$OUTPUT_ROOT run_date=$RUN_DATE"
common=(--input-jsonl "$INPUT_JSONL" --selector "$SELECTOR_LABEL" --output-root "$OUTPUT_ROOT" --mode matched --top-k "$TOP_K"
        --samples "$SAMPLES" --run-date "$RUN_DATE" --request-timeout "${REQUEST_TIMEOUT:-900}"
        --resume-fix-invalid --record-input-messages)
if [[ "$DRY_RUN" == 1 ]]; then common+=(--dry-run); fi
manifest_args=()
for alias in "${models[@]}"; do
  case "$alias" in
    qwen) model_id=qwen/qwen3-235b-a22b-fp8; label=Qwen3-235B; rpm="${QWEN_RPM:-34}" ;;
    deepseek) model_id=deepseek/deepseek-v3.1; label=DeepSeek-V3.1; rpm="${DEEPSEEK_RPM:-33}" ;;
    kimi) model_id=moonshotai/kimi-k2-instruct; label=Kimi-K2-Instruct; rpm="${KIMI_RPM:-33}" ;;
    flash) model_id=gemini-3.6-flash; label=Gemini-3.6-Flash; rpm="${FLASH_RPM:-40}" ;;
  esac
  model_dir="$OUTPUT_ROOT/${model_id//\//_}"
  if [[ "$STAGE" == all || "$STAGE" == collect ]]; then
    if [[ "$alias" == flash ]]; then
      run "$PYTHON" "$SCRIPT_DIR/collect_gemini_flash.py" "${common[@]}" --model "$model_id" \
        --base-url "$GEMINI_BASE_URL" --api-key-env "$GEMINI_KEY_ENV" --rate-limit-per-min "$rpm"
    else
      run "$PYTHON" "$SCRIPT_DIR/collect_novita.py" "${common[@]}" --model "$model_id" \
        --base-url "$NOVITA_BASE_URL" --api-key-env "$NOVITA_KEY_ENV" --rate-limit-per-min "$rpm"
    fi
  fi
  if [[ "$STAGE" == all || "$STAGE" == grade ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "[dry-run] grade generated answers in $model_dir with gemini-2.5-pro, samples=$SAMPLES"
    else
      run "$PYTHON" "$SCRIPT_DIR/grade_gemini.py" --answers-jsonl "$model_dir/matched/${SELECTOR_LABEL}_${RUN_DATE}.jsonl" --benchmark-jsonl "$INPUT_JSONL" \
        --run-date "$RUN_DATE" --model gemini-2.5-pro --base-url "$GEMINI_BASE_URL" \
        --api-key-env "$GEMINI_KEY_ENV" --samples "$SAMPLES" --rate-limit-per-min "${JUDGE_RPM:-30}" \
        --workers "${JUDGE_WORKERS:-12}" --max-retries "${JUDGE_RETRIES:-4}" --sleep "${JUDGE_SLEEP:-0.2}"
    fi
  fi
  manifest_args+=("$label" "$SELECTOR_LABEL" "$model_dir/matched/${SELECTOR_LABEL}_${RUN_DATE}_graded.jsonl")
done

if [[ "$STAGE" == all || "$STAGE" == summarize ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] aggregate complete per-problem scores; no API calls or output files written"
    exit 0
  fi
  manifest="$OUTPUT_ROOT/runs_${RUN_DATE}.json"
  "$PYTHON" - "$manifest" "$SAMPLES" "${manifest_args[@]}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
fields = sys.argv[3:]
runs = [{"model": fields[i], "selector": fields[i + 1], "graded_file": str(Path(fields[i + 2]).resolve())}
        for i in range(0, len(fields), 3)]
missing = [run["graded_file"] for run in runs if not Path(run["graded_file"]).is_file()]
if missing:
    raise SystemExit("Missing graded files: " + ", ".join(missing))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"samples": int(sys.argv[2]), "runs": runs}, indent=2) + "\n")
print(f"[saved] {path}")
PY
  run "$PYTHON" "$SCRIPT_DIR/summarize_scores.py" --manifest "$manifest" \
    --output "$OUTPUT_ROOT/summary_${RUN_DATE}.json"
fi
