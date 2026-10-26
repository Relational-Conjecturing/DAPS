#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Default to packaged sample dataset for reproducible smoke tests.
DATA_DIR="${DATA_DIR:-$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/crawled_repos}"
SPLITS_PATH="${SPLITS_PATH:-$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json}"

# Phase2 selector checkpoint (absolute-by-construction default; override as needed)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt}"

# Phase1 text encoder (absolute-by-construction default; override as needed)
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final}"
ALLOW_RANDOM_INIT="${ALLOW_RANDOM_INIT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RESUME_PATH="${RESUME_PATH:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"

CMD=(
  python train_and_eval_blueprint.py
  --data-dir "$DATA_DIR"
  --checkpoint-path "$CHECKPOINT_PATH"
  --text-encoder-path "$TEXT_ENCODER_PATH"
)

if [[ -n "$SPLITS_PATH" ]]; then
  CMD+=(--splits-path "$SPLITS_PATH")
fi
if [[ -n "$OUTPUT_ROOT" ]]; then
  CMD+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "$RESUME_PATH" ]]; then
  CMD+=(--resume "$RESUME_PATH")
fi
if [[ "$EVAL_ONLY" == "1" ]]; then
  CMD+=(--eval-only)
fi
if [[ "$ALLOW_RANDOM_INIT" == "1" ]]; then
  CMD+=(--allow-random-init)
fi

exec "${CMD[@]}"
