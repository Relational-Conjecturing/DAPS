# Preparations
## absolute path convention
```bash
export REPO_ROOT="/ABSOLUTE/PATH/TO/REPO_DAPS"
```

## core dependencies
- Python 3.10+
- `pip install -r "$REPO_ROOT/requirements.txt"`
- Lean toolchain via `elan` + `lake`
- Optional: `wandb`

# Resources
All demo files listed below are lightweight subsets for smoke tests and quick sanity checks.

## mathlib graph demo
- Graph demo root: `$REPO_ROOT/graph-demo`
- Mathlib4 dependency graph sample: `$REPO_ROOT/graph-demo/example_graph.json`

## blueprint graph demo
- 10-example blueprint demo:
  - `$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/crawled_repos/`
  - `$REPO_ROOT/datasets/DAPS-Blueprint/10-examples/splits_sample_train8_val1_test1.json`

## benchmarks demo
- Mathlib random demo:
  - 10-query JSONL: `$REPO_ROOT/datasets/mathlib4-random-demo/10-examples/mathlib_random_test_demo_10.jsonl`
- Mathlib heldout demo:
  - 10-query JSONL: `$REPO_ROOT/datasets/mathlib4-heldout-demo/10-examples/mathlib_heldout_test_demo_10.jsonl`
- DAPS-Collection demo:
  - 10-query JSONL: `$REPO_ROOT/datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl`

# Graph Construction
`build_mathlib_dataset`
- Usage doc: `$REPO_ROOT/build_mathlib_dataset/README.md`
- Pipeline script: `$REPO_ROOT/build_mathlib_dataset/run_p2_pipeline.sh`
- Core implementation: `$REPO_ROOT/build_mathlib_dataset/graph_pipeline/`
- This stage should be completed before Phase 1 and Phase 2, because it produces the Mathlib graph artifacts used by selector training and evaluation.

# DAPS Implementation
## training
- Full selector reproduction guide: `$REPO_ROOT/train_DAPS_selector/README.md`
- Phase 1 text encoder pretraining: `$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/run_train.sh`
- Phase 2 subgraph-aware selector training: `$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/run_train.sh`
- Phase 3 blueprint continual learning (optional): `$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh`

## inference
DAPS supports two evaluation tracks:
- Base DAPS (trained with Phase 1 + Phase 2):
  - embedding + retrieval evaluation: `$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/eval_selector.py`
  - full metrics report: `$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/evaluate_phase2_selector.py`
- Continual DAPS (Phase 3 on top of Phase 1 + Phase 2):
  - training plus validation and test entrypoint: `$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/run_train.sh`
  - eval-only entrypoint (with a continual-learning checkpoint): `$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/train_and_eval_blueprint.py` with `--eval-only --resume /ABSOLUTE/PATH/TO/model_last.pt`

## selector test phase
- For Mathlib and DAPS-Collection retrieval metrics, use the phase2 scripts above.
- For Blueprint continual-learning metrics, use phase3 run outputs (`test_metrics.json`, `metrics.csv`).

## selector verification
- Phase2 evaluation scripts:
  - `$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/eval_selector.py`
  - `$REPO_ROOT/train_DAPS_selector/phase2_train_with_subgraph/evaluate_phase2_selector.py`
- Phase3 evaluation path:
  - `$REPO_ROOT/train_DAPS_selector/phase3_train_DAPS_blueprint/train_and_eval_blueprint.py` with `--eval-only --resume /ABSOLUTE/PATH/TO/model_last.pt`

# Downstream Utility Evaluation

[`imo_proofbench_eval`](imo_proofbench_eval/) uses IMO-ProofBench to evaluate how premises retrieved by DAPS affect downstream LLMs' informal mathematical reasoning. It includes answer generation, grading and score aggregation; see its [README](imo_proofbench_eval/README.md) for usage.
