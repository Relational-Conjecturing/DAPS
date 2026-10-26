# train_DAPS_selector: Single-Page Reproduction Guide

This file is the primary README for running selector training and evaluation.
All commands below are designed to work from any current working directory.

## Scope of Packaged Data

The packaged sample data for this submission is intentionally small and only for smoke tests.
These files are stored at repository root paths (mainly under `$REPO_ROOT/datasets/`,
plus `$REPO_ROOT/graph-demo/`), not inside the `train_DAPS_selector/` folder:
- phase2 sample graph: small graph demo (`$REPO_ROOT/graph-demo/example_graph.json`)
- DAPS-Collection test sample: 10 queries
- phase3 blueprint sample: 10 queries (train=8, val=1, test=1)

For simplicity, the full corpora used for paper-scale training and evaluation are not included in this package.

## 1) Set Absolute Root

```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/SUBMIT_CODE_DAPS"
```

## 2) Python Environment

```bash
python -m venv "$REPO_ROOT/.venv_selector"
source "$REPO_ROOT/.venv_selector/bin/activate"
pip install -U pip
pip install -r "$REPO_ROOT/requirements.txt"
```

## 3) Preflight Checks

```bash
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["REPO_ROOT"]).resolve()
checks = [
    root / "build_mathlib_dataset/run_p2_pipeline.sh",
    root / "train_DAPS_selector/phase1_train_text_encoder/run_train.sh",
    root / "train_DAPS_selector/phase2_train_with_subgraph/run_train.sh",
    root / "train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh",
    root / "datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl",
    root / "graph-demo/example_graph.json",
    root / "datasets/mathlib4-random-demo/10-examples/mathlib_random_test_demo_10.jsonl",
    root / "datasets/mathlib4-heldout-demo/10-examples/mathlib_heldout_test_demo_10.jsonl",
    root / "datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json",
]
missing = [str(p) for p in checks if not p.exists()]
print("MISSING:" if missing else "OK")
for m in missing:
    print(m)
PY
```

## 4) Build Mathlib Graph (required for full phase1 and phase2)

```bash
bash "$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh"
```

Default full graph path:

```bash
$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json
```

This submission provides a fixed Mathlib split consistent with the paper evaluation:

```bash
$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/metadata/splits/splits_6687f6e53f5af7b1.json
```

Both phase1 and phase2 use this split JSON by default. If it is missing, both read or generate the same seed-42 JSON in this directory; the generated filename is printed during loading.

## 5) Train Commands

Note: this section focuses on training entrypoints.
- Phase1 and Phase2 `run_train.sh` are training-first commands (with in-training validation as implemented by the trainer).
- Phase3 uses `train_and_eval_blueprint.py`, so its run command performs training plus built-in validation and test.
- For standalone retrieval metric reports on Mathlib and DAPS-Collection, use Section 6 evaluation commands.

### 5.1 Phase 1 (text encoder)

```bash
bash "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/run_train.sh"
```

Optional explicit override:

```bash
GRAPH_JSON="$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
OUTPUT_DIR="$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/custom_phase1" \
NUM_GPUS=4 \
bash "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/run_train.sh"
```

Expected default phase1 checkpoint:

```bash
$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final
```

Default phase1 hyperparameters:
- `model-name=Alibaba-NLP/gte-modernbert-base`
- `lr=4e-4`
- `train-batch-size=32`
- `eval-batch-size=64`
- `mini-batch-size=32`
- `gradient-accumulation-steps=2`
- `gradient-checkpointing=enabled`
- `num-epochs=2`
- `warmup-ratio=0.04`
- `num-negatives-per-state=3`
- `query-max-length=1024`
- `premise-max-length=1024`
- `lean4-code-max-chars=2000`

### 5.2 Phase 2 (subgraph selector)

```bash
bash "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/run_train.sh"
```

Phase2 uses `OUT_DIR` exactly as supplied and saves checkpoints directly inside it.
Default training uses `EPOCHS=80`; the expected checkpoint is:

```bash
$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt
```

Optional explicit override (reported large-scale setting):

```bash
GRAPH_PATH="$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
OUT_DIR="$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/custom_phase2" \
NUM_GPUS=16 \
bash "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/run_train.sh"
```


Fixed phase2 setting in this submission:
- `FILTER_VARIANTS=("raw")`
- `PATTERN_MODES=("none")`
- embedding key: `alibaba-gte-modernbert-ft` (loaded from phase1 output)


### 5.3 Phase 3 (blueprint continual training, optional)

The archived Phase3 model can be evaluated with `--resume .../daps_selector_blueprint.pt --eval-only`.
`--checkpoint-path` initializes Phase3 from a Phase2 checkpoint. It loads the shared encoder, retrieval heads and codebook weights, while keeping fresh Blueprint-specific node centroid assignments (`premise_encoder.conv.c_idx`) and node-type embeddings (`premise_encoder.type_encoder.embedding.weight`). 

Use `--resume` to restore a Phase3 checkpoint, including its node/type state.

Full dataset mode:

Set `DATA_DIR` and `SPLITS_PATH` to the complete Blueprint dataset and its split file.

```bash
DATA_DIR="/ABSOLUTE/PATH/TO/FULL_BLUEPRINT_DATASET/full_collection" \
SPLITS_PATH="/ABSOLUTE/PATH/TO/FULL_BLUEPRINT_DATASET/splits.json" \
CHECKPOINT_PATH="$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt" \
TEXT_ENCODER_PATH="$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final" \
bash "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh"
```

Included sample mode (train=8, val=1, test=1; total=10):

```bash
DATA_DIR="$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/crawled_repos" \
SPLITS_PATH="$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json" \
CHECKPOINT_PATH="$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt" \
TEXT_ENCODER_PATH="$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final" \
bash "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh"
```

Missing checkpoints raise an error by default. For a smoke test with random initialization, explicitly choose a nonexistent checkpoint path and allow the fallback:

```bash
ALLOW_RANDOM_INIT=1 \
CHECKPOINT_PATH="$REPO_ROOT/no_phase2_checkpoint.pt" \
bash "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh"
```

Phase3 reported metrics:
- `global/recall@k`
- `global/recall@k_with_lean`
- `global/recall@k_nlp_only`

Phase3 outputs (under `.../outputs/blueprint_run_YYYYmmdd_HHMMSS/`):
- `config.json`
- `model_best.pt`, `model_last.pt`
- `metrics_epoch_*.json`, `metrics.csv`
- `test_metrics.json`
- `sample_query.json`

## 6) Evaluation Commands

This repository supports two selector variants:
- Base DAPS (Phase1 + Phase2)
- Continual DAPS (Phase1 + Phase2 + optional Phase3 Blueprint continual learning)

### 6.1 Base DAPS (Phase1 + Phase2): full metrics report

This report evaluates three test subsets together:
- `Mathlib4-H` (mathlib holdout subset from split cache `test_holdout`)
- `Mathlib4-R` (mathlib random subset from split cache `test_random`)
- `DAPS_collection` (DAPS-Collection test JSONL)

Generate premise embeddings from the phase2 checkpoint (one-time):

```bash
python "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/eval_selector.py" \
  --graph-path "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
  --split-cache-json "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/metadata/splits/splits_6687f6e53f5af7b1.json" \
  --daps-collection-test-jsonl "$REPO_ROOT/datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl" \
  --checkpoint "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt" \
  --embeddings-out "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/embeddings_new/premise_embeddings_epoch_80_alibaba-gte-modernbert-ft_len1024.pt"
```

Then run the phase2 selector report:

```bash
python "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/evaluate_phase2_selector.py" \
  --phase2-checkpoint "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt" \
  --phase2-embeddings "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/embeddings_new/premise_embeddings_epoch_80_alibaba-gte-modernbert-ft_len1024.pt" \
  --split-cache-json "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/metadata/splits/splits_6687f6e53f5af7b1.json" \
  --daps-collection-test-jsonl "$REPO_ROOT/datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl"
```

### 6.2 Evaluate a specific phase2 checkpoint

```bash
python "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/eval_selector.py" \
  --graph-path "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
  --split-cache-json "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/metadata/splits/splits_6687f6e53f5af7b1.json" \
  --daps-collection-test-jsonl "$REPO_ROOT/datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl" \
  --checkpoint "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt"
```
`eval_selector.py` and `evaluate_phase2_selector.py` use `--split-cache-json` as the standard split source.  
The optional `--mathlib-test-*` arguments accept manually supplied query-ID files.

The released `Mathlib/mathlib_test_heldout.jsonl` and `Mathlib/mathlib_test_random.jsonl` are readable exports with `query_key`, `query_text`, and premise names in `positives`. Mathlib evaluation derives its queries and gold from the split JSON, graph and filter configuration; it does not read these two exports.

### 6.3 Continual DAPS (+Phase3): Blueprint evaluation

Standard training plus validation and test run:

```bash
DATA_DIR="$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/crawled_repos" \
SPLITS_PATH="$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json" \
CHECKPOINT_PATH="$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/runs/phase2_subgraph/checkpoint_epoch_80.pt" \
TEXT_ENCODER_PATH="$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final" \
bash "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh"
```

Eval-only from an existing continual-learning checkpoint:

```bash
python "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/train_and_eval_blueprint.py" \
  --data-dir "/ABSOLUTE/PATH/TO/FULL_BLUEPRINT_DATASET/full_collection" \
  --splits-path "/ABSOLUTE/PATH/TO/FULL_BLUEPRINT_DATASET/splits.json" \
  --resume "/ABSOLUTE/PATH/TO/PHASE3_OUTPUT/model_last.pt" \
  --eval-only
```

Notes:
- `--resume` should point to a phase3 output checkpoint such as `model_last.pt`.
- Eval-only outputs are written under the phase3 output directory and include `test_metrics.json`.

## 7) Direct Python Entry Support (any cwd)

The key Python entry scripts are callable via absolute path without `cd`:

```bash
python "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/train_text_encoder.py" --help
python "$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/premise_selector/train.py" --help
python "$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/train_and_eval_blueprint.py" --help
```

## 8) Dataset Format

### 8.1 Phase2 Mathlib Demo Data

Packaged lightweight demo files:
- `$REPO_ROOT/graph-demo/example_graph.json` (small graph demo)
- `$REPO_ROOT/datasets/mathlib4-heldout-demo/10-examples/mathlib_heldout_test_demo_10.jsonl`
- `$REPO_ROOT/datasets/mathlib4-random-demo/10-examples/mathlib_random_test_demo_10.jsonl`

These are smoke-test subsets only, not full Mathlib evaluation corpora.

How they are used:
- Default phase2 training (`run_train.sh`) uses the full graph under `build_mathlib_dataset/generated_graphs/...`.
- If the default full graph path is missing, `eval_selector.py` can fall back to `$REPO_ROOT/graph-demo/example_graph.json` for smoke testing.

### 8.2 Phase3 Blueprint Dataset Contract

Expected full dataset root:

```bash
/ABSOLUTE/PATH/TO/FULL_BLUEPRINT_DATASET/full_collection/
```

Each `*.jsonl` file corresponds to one repository.

Required fields per JSONL record:
- `id` (string)
- `dependencies` (list[string])
- `proof_status` (`all_formalized` / `statement_formalized` / `nlp_only`)
- `kind` (`ellipse` / `box`)
- `nlp_text` (string)

Optional fields used by text construction:
- `lean4_code` (string)
- `lean_decl_name` (string)

If split file is missing, it is auto-generated. Expected split JSON structure:

```json
{
  "seed": 42,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "train": [{"id": "...", "repo": "...", "full_id": "repo/id"}],
  "val": [...],
  "test": [...],
  "stats": {"train_count": 0, "val_count": 0, "test_count": 0}
}
```

Included blueprint sample subset:
- `$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/crawled_repos/`
- `$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json`

The packaged `datasets/DAPS-Blueprint/10-examples` subset is trimmed to 10 total queries for lightweight submission smoke tests.
It currently only includes 3 repo slices.
