#!/usr/bin/env bash
# ==============================================================================
# Train Text Encoder (gte-modernbert-base) for Premise Selection
# ==============================================================================
# Usage (single GPU):
#   bash "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/run_train.sh"
#
# Usage (multi-GPU DDP):
#   NUM_GPUS=4 bash "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/run_train.sh"
#
# Environment variables:
#   NUM_GPUS          - Number of GPUs (default: 1, auto-detects if not set)
#   GRAPH_JSON        - Path to graph JSON (default: auto-detected)
#   OUTPUT_DIR        - Output directory (default: $REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/...)
#   LR                - Learning rate (default: 4e-4)
#   BATCH_SIZE        - Per-device batch size (default: 32)
#   NUM_EPOCHS        - Number of epochs (default: 2)
#   MINI_BATCH_SIZE   - Mini-batch size for CachedMNRL (default: 32)
#   GRAD_ACCUM_STEPS  - Gradient accumulation steps (default: 2)
#   GRAD_CHECKPOINTING- Enable gradient checkpointing: 1/0 (default: 1)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect graph JSON path
GRAPH_JSON_DEFAULT="$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json"
GRAPH_JSON="${GRAPH_JSON:-$GRAPH_JSON_DEFAULT}"
if [[ ! -f "$GRAPH_JSON" ]]; then
    echo "ERROR: Graph JSON not found at: $GRAPH_JSON"
    echo ""
    echo "Please run the dataset generation pipeline first:"
    echo "  cd \"$REPO_ROOT/build_mathlib_dataset\" && ./run_p2_pipeline.sh"
    echo ""
    echo "Or specify the path manually:"
    echo "  GRAPH_JSON=\"$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json\" ./run_train.sh"
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-Alibaba-NLP/gte-modernbert-base}"
LR="${LR:-4e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
NUM_NEGATIVES="${NUM_NEGATIVES:-3}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-1024}"
PREMISE_MAX_LEN="${PREMISE_MAX_LEN:-1024}"
LEAN4_CODE_MAX_CHARS="${LEAN4_CODE_MAX_CHARS:-2000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"
SEED="${SEED:-42}"

# Output directory
MODEL_SLUG="${MODEL_NAME//\//-}"
GC_TAG=""
if [[ "$GRAD_CHECKPOINTING" == "1" ]]; then
    GC_TAG="-gc"
fi
OUTPUT_DIR_DEFAULT="$SCRIPT_DIR/runs/${MODEL_SLUG}-lr${LR}-bs${BATCH_SIZE}-nneg${NUM_NEGATIVES}-ne${NUM_EPOCHS}-wr${WARMUP_RATIO}-mb${MINI_BATCH_SIZE}-ga${GRAD_ACCUM_STEPS}${GC_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

# Detect number of GPUs
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        [[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
    else
        NUM_GPUS=1
    fi
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

echo "============================================"
echo "Training Configuration:"
echo "  Model:           $MODEL_NAME"
echo "  Graph JSON:      $GRAPH_JSON"
echo "  Output:          $OUTPUT_DIR"
echo "  GPUs:            $NUM_GPUS"
echo "  Batch size:      $BATCH_SIZE (per device)"
echo "  Mini-batch:      $MINI_BATCH_SIZE"
echo "  Learning rate:   $LR"
echo "  Warmup ratio:    $WARMUP_RATIO"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Grad accum:      $GRAD_ACCUM_STEPS"
echo "  Grad checkpoint: $GRAD_CHECKPOINTING"
echo "  Negatives:       $NUM_NEGATIVES"
echo "  Query max len:   $QUERY_MAX_LEN"
echo "  Premise max len: $PREMISE_MAX_LEN"
echo "============================================"

TRAIN_ARGS=(
    --graph-json "$GRAPH_JSON"
    --output-dir "$OUTPUT_DIR"
    --model-name "$MODEL_NAME"
    --train-batch-size "$BATCH_SIZE"
    --eval-batch-size 64
    --lr "$LR"
    --warmup-ratio "$WARMUP_RATIO"
    --num-epochs "$NUM_EPOCHS"
    --num-negatives-per-state "$NUM_NEGATIVES"
    --mini-batch-size "$MINI_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --query-max-length "$QUERY_MAX_LEN"
    --premise-max-length "$PREMISE_MAX_LEN"
    --lean4-code-max-chars "$LEAN4_CODE_MAX_CHARS"
    --include-lean4-code
    --seed "$SEED"
    --bf16
    --report-to none
)

if [[ "$GRAD_CHECKPOINTING" == "1" ]]; then
    TRAIN_ARGS+=(--gradient-checkpointing)
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "Running with torchrun (DDP) on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node="$NUM_GPUS" train_text_encoder.py "${TRAIN_ARGS[@]}"
else
    echo "Running single-GPU training..."
    python train_text_encoder.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR/final"
