#!/usr/bin/env bash
# ==============================================================================
# Phase 2: Train RelGT Premise Selector with Subgraph (setting: alibaba-gte-modernbert-ft, raw, pattern=none)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

GRAPH_PATH_DEFAULT="$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json"
GRAPH_PATH="${GRAPH_PATH:-$GRAPH_PATH_DEFAULT}"
if [[ ! -f "$GRAPH_PATH" ]]; then
  echo "ERROR: Graph JSON not found at: $GRAPH_PATH" >&2
  echo "Please run: cd \"$REPO_ROOT/build_mathlib_dataset\" && ./run_p2_pipeline.sh" >&2
  exit 1
fi

# Output/cache paths
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/runs/phase2_subgraph}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$SCRIPT_DIR/runs/feature_cache/mathlib4-4.22.0-full}"
SUBGRAPH_CACHE_DIR="${SUBGRAPH_CACHE_DIR:-$SCRIPT_DIR/runs/subgraph_cache/mathlib4-4.22.0-full}"
MEMORY_BANK_DIR="${MEMORY_BANK_DIR:-$OUT_DIR/memory_bank}"

# Core setting (fixed for this run)
EMBEDDING="${EMBEDDING:-alibaba-gte-modernbert-ft}"
PROOF_PATTERN_MODE="${PROOF_PATTERN_MODE:-none}"
SUBGRAPH_MODE="${SUBGRAPH_MODE:-0}"
SELECT_TOP_K="${SELECT_TOP_K:-64}"

# Hyperparameters for the submission default run
LR="${LR:-0.0006}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_NODES="${MAX_NODES:-300}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
PROJ_DIM="${PROJ_DIM:-256}"
LOCAL_LAYERS="${LOCAL_LAYERS:-3}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
SUB_NODE_MAX_LENGTH="${SUB_NODE_MAX_LENGTH:-1024}"
PROOF_PATTERN_MAX_LENGTH="${PROOF_PATTERN_MAX_LENGTH:-768}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-512}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
GLOBAL_NEGS="${GLOBAL_NEGS:-30}"
RANDOM_HARD_NEGS="${RANDOM_HARD_NEGS:-30}"
CROSS_BATCH_NEGS="${CROSS_BATCH_NEGS:-32}"
CROSS_BATCH_QUEUE="${CROSS_BATCH_QUEUE:-4096}"
MAX_CANDIDATES_PER_BATCH="${MAX_CANDIDATES_PER_BATCH:-1500}"
TRAIN_REPEAT="${TRAIN_REPEAT:-2}"
RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-60}"

SPLIT_MODE="${SPLIT_MODE:-fixed_holdout}"
VAL_COUNT="${VAL_COUNT:-2000}"
TEST_COUNT="${TEST_COUNT:-2000}"
HOLDOUT_PER_SPLIT="${HOLDOUT_PER_SPLIT:-1000}"

NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-1}"
EVAL_DYNAMIC_FRACTION="${EVAL_DYNAMIC_FRACTION:-0.05}"

DEVICE="${DEVICE:-cuda}"
TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda}"
MEMORY_BANK_BUILD_DEVICE="${MEMORY_BANK_BUILD_DEVICE:-}"

mkdir -p "$OUT_DIR" "$FEATURE_CACHE_DIR" "$SUBGRAPH_CACHE_DIR" "$MEMORY_BANK_DIR"

TRAIN_ARGS=(
  --graph-path "$GRAPH_PATH"
  --dataset-mode mathlib
  --device "$DEVICE"
  --text-encoder-device "$TEXT_ENCODER_DEVICE"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --warmup-ratio "$WARMUP_RATIO"
  --max-nodes "$MAX_NODES"
  --subgraph-mode "$SUBGRAPH_MODE"
  --hidden-dim "$HIDDEN_DIM"
  --proj-dim "$PROJ_DIM"
  --local-layers "$LOCAL_LAYERS"
  --heads "$HEADS"
  --dropout "$DROPOUT"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --pin-memory
  --eval-num-workers "$EVAL_NUM_WORKERS"
  --eval-prefetch-factor "$EVAL_PREFETCH_FACTOR"
  --eval-dynamic-fraction "$EVAL_DYNAMIC_FRACTION"
  --out-dir "$OUT_DIR"
  --feature-cache-dir "$FEATURE_CACHE_DIR"
  --subgraph-cache-dir "$SUBGRAPH_CACHE_DIR"
  --memory-bank-dir "$MEMORY_BANK_DIR"
  --memory-bank-per-query 0
  --memory-bank-topk 512
  --memory-bank-chunk-size 2048
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --loss-reduction positive
  --loss-type positive_infonce
  --diversity-weight 0.0
  --cross-batch-negs "$CROSS_BATCH_NEGS"
  --cross-batch-queue "$CROSS_BATCH_QUEUE"
  --max-candidates-per-batch "$MAX_CANDIDATES_PER_BATCH"
  --global-negatives-per-query "$GLOBAL_NEGS"
  --random-negatives-per-query 0
  --random-hard-negatives-per-query "$RANDOM_HARD_NEGS"
  --hard-negatives-per-query 0
  --train-repeat "$TRAIN_REPEAT"
  --resource-monitor-interval "$RESOURCE_MONITOR_INTERVAL"
  --embedding "$EMBEDDING"
  --max-length "$MAX_LENGTH"
  --sub-node-max-length "$SUB_NODE_MAX_LENGTH"
  --precompute-batch-size "$PRECOMPUTE_BATCH_SIZE"
  --use-amp
  --split-mode "$SPLIT_MODE"
  --val-count "$VAL_COUNT"
  --test-count "$TEST_COUNT"
  --holdout-per-split "$HOLDOUT_PER_SPLIT"
  --proof-pattern-mode "$PROOF_PATTERN_MODE"
  --proof-pattern-max-length "$PROOF_PATTERN_MAX_LENGTH"
)

if [[ -n "$SELECT_TOP_K" ]]; then
  TRAIN_ARGS+=(--select-top-k "$SELECT_TOP_K")
fi

if [[ -n "$MEMORY_BANK_BUILD_DEVICE" ]]; then
  TRAIN_ARGS+=(--memory-bank-build-device "$MEMORY_BANK_BUILD_DEVICE")
fi

# Detect number of GPUs (optional)
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    [[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
  else
    NUM_GPUS=1
  fi
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Running DDP with torchrun on $NUM_GPUS GPUs..."
  torchrun --nproc_per_node="$NUM_GPUS" -m premise_selector.train "${TRAIN_ARGS[@]}"
else
  echo "Running single-process training..."
  python -m premise_selector.train "${TRAIN_ARGS[@]}"
fi
