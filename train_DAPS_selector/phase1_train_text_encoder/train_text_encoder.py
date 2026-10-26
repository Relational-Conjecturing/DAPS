#!/usr/bin/env python3
# ==============================================================================
# Text Encoder Training for Premise Selection
# ==============================================================================
# Usage (single GPU):
#   export REPO_ROOT="/abs/SUBMIT_CODE"
#   python train_text_encoder.py \
#       --graph-json "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
#       --model-name Alibaba-NLP/gte-modernbert-base \
#       --output-dir "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/gte-modernbert"
#
# Usage (multi-GPU with torchrun):
#   export REPO_ROOT="/abs/SUBMIT_CODE"
#   torchrun --nproc_per_node=4 train_text_encoder.py \
#       --graph-json "$REPO_ROOT/build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/lake_graph_with_code_mathlib4-4.22.0-full.json" \
#       --model-name Alibaba-NLP/gte-modernbert-base \
#       --output-dir "$REPO_ROOT/train_DAPS_selector/phase1_train_text_encoder/runs/gte-modernbert"
# ==============================================================================
from __future__ import annotations

import os

# Keep help/CLI robust in environments where TensorFlow is present but not usable.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from functools import partial

import datasets
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import _backward_hook
from transformers import set_seed
from typing import Iterable

from common.graph_loader import build_premise_text, load_mathlib_graph_lite
from common.constants import DEFAULT_POSITIVE_EDGE_TYPES, DEFAULT_PREMISE_TYPES, DEFAULT_QUERY_TYPES
from common.splits import DEFAULT_SPLIT_CACHE_JSON, load_or_create_mathlib_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_GRAPH_JSON = REPO_ROOT / "build_mathlib_dataset" / "generated_graphs" / "mathlib4-4.22.0-full" / "lake_graph_with_code_mathlib4-4.22.0-full.json"


# -----------------------------------------------------------------------------
# MaskedCachedMultipleNegativesRankingLoss
# -----------------------------------------------------------------------------
class MaskedCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """CMNRL with retrieval mask support to exclude known positives from negatives."""

    def calculate_loss_and_cache_gradients(
        self, reps: list[list[torch.Tensor]], retrieval_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings_a = torch.cat(reps[0])
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])

        batch_size = len(embeddings_a)
        labels = torch.tensor(range(batch_size), dtype=torch.long, device=embeddings_a.device)
        losses: list[torch.Tensor] = []
        for b in range(0, batch_size, self.mini_batch_size):
            e = b + self.mini_batch_size
            scores: torch.Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            if retrieval_mask is not None:
                scores[~retrieval_mask[b:e]] = -torch.inf
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses).requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_loss(
        self, reps: list[list[torch.Tensor]], retrieval_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings_a = torch.cat(reps[0])
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])

        batch_size = len(embeddings_a)
        labels = torch.tensor(range(batch_size), dtype=torch.long, device=embeddings_a.device)
        losses: list[torch.Tensor] = []
        for b in range(0, batch_size, self.mini_batch_size):
            e = b + self.mini_batch_size
            scores: torch.Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            if retrieval_mask is not None:
                scores[~retrieval_mask[b:e]] = -torch.inf
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            losses.append(loss_mbatch)

        return sum(losses)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        reps = []
        self.random_states = []
        retrieval_mask = None
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)
            if "retrieval_mask" in sentence_feature:
                retrieval_mask = sentence_feature["retrieval_mask"]

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, retrieval_mask)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps, retrieval_mask)
        return loss


# -----------------------------------------------------------------------------
# Dataset and Collator classes
# -----------------------------------------------------------------------------
def _clean_query_text(text: str) -> str:
    """Clean query text (statement field)."""
    return (text or "").lstrip()


@dataclass(frozen=True)
class Example:
    state_id: int
    premise_id: int


class MathlibDataset:
    """Dataset for Mathlib4 premise selection training."""

    def __init__(
        self,
        *,
        graph,
        state_ids: Sequence[int],
        num_negatives_per_state: int,
        seed: int,
        include_lean4_code: bool,
        include_statement: bool,
        lean4_code_max_chars: int,
        train: bool,
    ):
        self.graph = graph
        self.state_ids = [int(s) for s in state_ids]
        self.state_ids_set = set(self.state_ids)
        self.train = train
        self.num_negatives_per_state = num_negatives_per_state if train else 0
        self.lean4_code_max_chars = lean4_code_max_chars
        self.include_lean4_code = include_lean4_code
        self.include_statement = include_statement

        # Build examples: one per (query, positive_premise) pair
        self.examples: List[Example] = []
        for sid in self.state_ids:
            for pid in self.graph.positives.get(sid, []):
                self.examples.append(Example(state_id=sid, premise_id=pid))

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self._premise_cache: Dict[int, str] = {}
        self._query_cache: Dict[int, str] = {}

        self._premise_id_list: List[int] = list(self.graph.premise_ids)
        self._positive_sets: Dict[int, Set[int]] = {k: set(v) for k, v in self.graph.positives.items()}

    def __len__(self) -> int:
        return len(self.examples)

    def to_hf_dataset(self) -> datasets.Dataset:
        examples = [{"state_id": ex.state_id, "premise_id": ex.premise_id} for ex in self.examples]
        features = datasets.Features({"state_id": datasets.Value("int64"), "premise_id": datasets.Value("int64")})
        return datasets.Dataset.from_list(examples, features=features)

    def _sample_one_negative(self, state_id: int) -> int:
        pos = self._positive_sets.get(state_id, set())
        for _ in range(32):
            idx = int(torch.randint(len(self._premise_id_list), (1,), generator=self.generator).item())
            pid = self._premise_id_list[idx]
            if pid not in pos and pid != state_id:
                return pid
        fallback = [pid for pid in self._premise_id_list if pid not in pos and pid != state_id]
        if not fallback:
            return self._premise_id_list[0]
        idx = int(torch.randint(len(fallback), (1,), generator=self.generator).item())
        return fallback[idx]

    def sample_negatives(self, state_id: int, k: int) -> List[int]:
        return [self._sample_one_negative(state_id) for _ in range(k)]

    def get_query_text(self, state_id: int) -> str:
        if state_id not in self._query_cache:
            node = self.graph.nodes[state_id]
            # Use statement as query
            self._query_cache[state_id] = _clean_query_text(node.statement or "")
        return self._query_cache[state_id]

    def get_premise_text(self, premise_id: int) -> str:
        if premise_id not in self._premise_cache:
            node = self.graph.nodes[premise_id]
            self._premise_cache[premise_id] = build_premise_text(
                node,
                include_lean4_code=self.include_lean4_code,
                include_statement=self.include_statement,
                lean4_code_max_chars=self.lean4_code_max_chars,
            )
        return self._premise_cache[premise_id]

    def positive_set(self, state_id: int) -> Set[int]:
        return self._positive_sets.get(state_id, set())


class TextEncoderCollator:
    """Data collator for text encoder training."""

    def __init__(
        self,
        *,
        model: SentenceTransformer,
        train_dataset: MathlibDataset,
        all_positive_sets: Dict[int, Set[int]],
        num_negatives_per_state: int,
        retrieval_mask_mode: str,
        query_max_length: int,
        premise_max_length: int,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.all_positive_sets = all_positive_sets
        self.num_negatives_per_state = num_negatives_per_state
        self.retrieval_mask_mode = retrieval_mask_mode
        self.query_max_length = query_max_length
        self.premise_max_length = premise_max_length
        # For SentenceTransformerTrainer API compatibility
        self.valid_label_columns = ["label", "score"]
        self._warned_columns = set()

    def __call__(self, features: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
        state_ids = [int(f["state_id"]) for f in features]
        premise_ids = [int(f["premise_id"]) for f in features]
        batch_size = len(state_ids)

        do_neg_sampling = bool(
            self.num_negatives_per_state > 0 and all(sid in self.train_dataset.state_ids_set for sid in state_ids)
        )
        neg_slots = self.num_negatives_per_state if do_neg_sampling else 0
        sampled_negs: List[List[int]] = []
        if do_neg_sampling:
            for sid in state_ids:
                sampled_negs.append(self.train_dataset.sample_negatives(sid, neg_slots))
        else:
            sampled_negs = [[] for _ in state_ids]

        query_texts = [self.train_dataset.get_query_text(sid) for sid in state_ids]
        pos_texts = [self.train_dataset.get_premise_text(pid) for pid in premise_ids]
        neg_texts_by_slot: List[List[str]] = []
        for neg_idx in range(neg_slots):
            neg_texts_by_slot.append([self.train_dataset.get_premise_text(neg_list[neg_idx]) for neg_list in sampled_negs])

        batch: Dict[str, torch.Tensor] = {}

        def _tok(texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
            return self.model.tokenize(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        for key, val in _tok(query_texts, self.query_max_length).items():
            batch[f"state_{key}"] = val
        for key, val in _tok(pos_texts, self.premise_max_length).items():
            batch[f"premise_{key}"] = val
        for neg_idx, texts in enumerate(neg_texts_by_slot):
            for key, val in _tok(texts, self.premise_max_length).items():
                batch[f"negative_premise_{neg_idx}_{key}"] = val

        # Build retrieval mask
        num_cols = 1 + neg_slots
        retrieval_mask = torch.ones((batch_size, num_cols, batch_size), dtype=torch.bool)
        for i, sid in enumerate(state_ids):
            positives = self.all_positive_sets.get(sid, set())
            for j in range(batch_size):
                doc_ids = [premise_ids[j]] + sampled_negs[j]
                for col_idx, pid in enumerate(doc_ids):
                    if col_idx == 0 and j == i:
                        continue
                    if pid in positives and self.retrieval_mask_mode != "none":
                        retrieval_mask[i, col_idx, j] = False
        batch["state_retrieval_mask"] = retrieval_mask.view(batch_size, -1)
        return batch


# -----------------------------------------------------------------------------
# Dataset building
# -----------------------------------------------------------------------------
def build_datasets(
    graph_path: Union[str, Path],
    *,
    seed: int,
    num_negatives_per_state: int,
    include_lean4_code: bool,
    include_statement: bool,
    lean4_code_max_chars: int,
    split_cache_json: Optional[Union[str, Path]] = None,
) -> Tuple[MathlibDataset, MathlibDataset, MathlibDataset, Dict[int, Set[int]]]:
    """Build train/val/test datasets with fixed holdout splits."""
    graph = load_mathlib_graph_lite(
        graph_path,
        query_types=DEFAULT_QUERY_TYPES,
        premise_types=DEFAULT_PREMISE_TYPES,
        positive_edge_types=DEFAULT_POSITIVE_EDGE_TYPES,
    )
    query_id_set = set(int(qid) for qid in graph.query_ids)
    split_cache_path, split_cache = load_or_create_mathlib_split(
        graph_path,
        split_cache_json or DEFAULT_SPLIT_CACHE_JSON,
        query_ids=graph.query_ids,
        source_paths=graph.source_paths,
    )
    train_ids = [qid for qid in split_cache["train"] if qid in query_id_set]
    val_ids = [qid for qid in split_cache["val"] if qid in query_id_set]
    test_ids = [qid for qid in split_cache["test"] if qid in query_id_set]
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(f"Split JSON is incompatible with the graph: {split_cache_path}")
    print(
        f"[split] using {split_cache_path} (train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})",
        flush=True,
    )

    positive_sets = {k: set(v) for k, v in graph.positives.items()}

    train_ds = MathlibDataset(
        graph=graph,
        state_ids=train_ids,
        num_negatives_per_state=num_negatives_per_state,
        seed=seed,
        include_lean4_code=include_lean4_code,
        include_statement=include_statement,
        lean4_code_max_chars=lean4_code_max_chars,
        train=True,
    )
    val_ds = MathlibDataset(
        graph=graph,
        state_ids=val_ids,
        num_negatives_per_state=0,
        seed=seed,
        include_lean4_code=include_lean4_code,
        include_statement=include_statement,
        lean4_code_max_chars=lean4_code_max_chars,
        train=False,
    )
    test_ds = MathlibDataset(
        graph=graph,
        state_ids=test_ids,
        num_negatives_per_state=0,
        seed=seed,
        include_lean4_code=include_lean4_code,
        include_statement=include_statement,
        lean4_code_max_chars=lean4_code_max_chars,
        train=False,
    )
    train_ds.split_cache_json = str(split_cache_path)
    return train_ds, val_ds, test_ds, positive_sets


# -----------------------------------------------------------------------------
# Model-specific configuration
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Alibaba-NLP/gte-modernbert-base": {
        "trust_remote_code": True,
        "notes": "ModernBERT architecture, optimized for code retrieval",
    },
}


def make_run_name(
    model_name: str,
    lr: float,
    batch_size: int,
    num_negatives: int,
    retrieval_mask_mode: str,
    num_epochs: int,
) -> str:
    """Generate run name for output directory."""
    base = model_name.replace("/", "-")
    base = f"{base}-lr{lr}-bs{batch_size}-nneg{num_negatives}"
    if retrieval_mask_mode != "none":
        base += "-ml"
        if retrieval_mask_mode != "no_positive":
            base += retrieval_mask_mode
    if num_epochs != 1:
        base += f"-ne{num_epochs}"
    return base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train text encoders for Mathlib4 premise selection."
    )
    ap.add_argument(
        "--graph-json",
        type=str,
        default=str(DEFAULT_GRAPH_JSON),
    )
    ap.add_argument("--output-dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-cache-json",
        type=str,
        default=str(DEFAULT_SPLIT_CACHE_JSON),
        help="Preferred split JSON; if missing, load or create a shared seed-42 split JSON in the same directory.",
    )

    # Model selection
    ap.add_argument(
        "--model-name",
        type=str,
        default="Alibaba-NLP/gte-modernbert-base",
        help="Pre-trained model name or path",
    )

    # Training hyperparameters
    ap.add_argument("--mini-batch-size", type=int, default=32,
                    help="Mini-batch size for CachedMNRL loss")
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.04)
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--num-negatives-per-state", type=int, default=3)
    ap.add_argument(
        "--retrieval-mask-mode",
        choices=["none", "no_positive", "only_negative"],
        default="no_positive",
    )

    # Text processing
    ap.add_argument("--query-max-length", type=int, default=1024)
    ap.add_argument("--premise-max-length", type=int, default=1024)
    ap.add_argument("--include-lean4-code", action="store_true")
    ap.add_argument("--no-include-lean4-code", action="store_false", dest="include_lean4_code")
    ap.add_argument("--include-statement", action="store_true", default=False)
    ap.add_argument("--lean4-code-max-chars", type=int, default=2000)

    # Training options
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-bf16", action="store_false", dest="bf16")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-fp16", action="store_false", dest="fp16")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--dataloader-num-workers", type=int, default=4)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    ap.add_argument("--resume-from-checkpoint", type=str, default="")
    ap.add_argument("--report-to", type=str, nargs="*", default=["none"])
    ap.add_argument("--trust-remote-code", action="store_true", default=True)

    ap.set_defaults(include_lean4_code=True, bf16=True, fp16=False, gradient_checkpointing=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    # Build datasets
    train_ds, val_ds, test_ds, positive_sets = build_datasets(
        args.graph_json,
        seed=args.seed,
        num_negatives_per_state=args.num_negatives_per_state,
        include_lean4_code=args.include_lean4_code,
        include_statement=args.include_statement,
        lean4_code_max_chars=args.lean4_code_max_chars,
        split_cache_json=args.split_cache_json,
    )
    args.split_cache_json = train_ds.split_cache_json

    # Generate run name and output directory
    run_name = make_run_name(
        model_name=args.model_name,
        lr=args.lr,
        batch_size=args.train_batch_size,
        num_negatives=args.num_negatives_per_state,
        retrieval_mask_mode=args.retrieval_mask_mode,
        num_epochs=args.num_epochs,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)} | Test examples: {len(test_ds)}")
    print(f"Using model: {args.model_name}")
    print(f"Output dir: {output_dir}")
    sample_state_ids = list(train_ds.state_ids[:3])
    sample_payload = []
    for sid in sample_state_ids:
        positives = sorted(train_ds.positive_set(sid))
        sample_payload.append(
            {
                "state_id": sid,
                "query_text": train_ds.get_query_text(sid),
                "positive_premise_ids": positives[:3],
                "positive_premise_texts": [train_ds.get_premise_text(pid) for pid in positives[:3]],
            }
        )
    sample_path = output_dir / "sample_pairs.json"
    sample_path.write_text(json.dumps(sample_payload, indent=2))
    print(f"Wrote samples: {sample_path}")

    # Check model-specific config
    model_config = MODEL_CONFIGS.get(args.model_name, {})
    trust_remote = model_config.get("trust_remote_code", args.trust_remote_code)
    if "notes" in model_config:
        print(f"Model notes: {model_config['notes']}")

    # Load model
    model = SentenceTransformer(
        model_name_or_path=args.model_name,
        trust_remote_code=trust_remote,
        model_card_data=SentenceTransformerModelCardData(
            train_datasets=[{"name": "mathlib_graph", "revision": "mathlib4"}]
        ),
    )

    # Expose the underlying encoder configuration.
    try:
        model.config = model._first_module().auto_model.config
    except Exception:
        pass

    # Force max sequence length to avoid tokenizers with no predefined max length.
    max_seq_len = max(args.query_max_length, args.premise_max_length)
    try:
        model.max_seq_length = max_seq_len
    except Exception:
        pass
    try:
        tokenizer = model._first_module().tokenizer
        model_max = getattr(tokenizer, "model_max_length", None)
        if model_max is None or model_max > 100000:
            tokenizer.model_max_length = max_seq_len
    except Exception:
        pass
    print(f"Max sequence length set to {max_seq_len}")

    # Convert to HuggingFace datasets
    train_hf = train_ds.to_hf_dataset()
    val_hf = val_ds.to_hf_dataset()

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        batch_sampler="batch_sampler",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        # Enable manually to control use_reentrant for DDP safety.
        gradient_checkpointing=False,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy="steps",
        eval_steps=0.01,
        save_strategy="steps",
        save_steps=0.01,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=run_name,
    )

    # Enable non-reentrant gradient checkpointing.
    if args.gradient_checkpointing:
        gc_enabled = False
        gc_kwargs = {"use_reentrant": False}
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)  # type: ignore[arg-type]
                except TypeError:
                    model.gradient_checkpointing_enable()  # type: ignore[call-arg]
                gc_enabled = True
        except Exception:
            gc_enabled = False

        if not gc_enabled:
            try:
                auto_model = model._first_module().auto_model
                if hasattr(auto_model, "gradient_checkpointing_enable"):
                    try:
                        auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)  # type: ignore[arg-type]
                    except TypeError:
                        auto_model.gradient_checkpointing_enable()  # type: ignore[call-arg]
                    gc_enabled = True
            except Exception:
                gc_enabled = False

        print(f"Gradient checkpointing enabled: {gc_enabled} (use_reentrant=False)")

    # Loss function
    loss = MaskedCachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

    # Data collator
    collator = TextEncoderCollator(
        model=model,
        train_dataset=train_ds,
        all_positive_sets=positive_sets,
        num_negatives_per_state=args.num_negatives_per_state,
        retrieval_mask_mode=args.retrieval_mask_mode,
        query_max_length=args.query_max_length,
        premise_max_length=args.premise_max_length,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        data_collator=collator,
        loss=loss,
        callbacks=[],
    )

    # Train (resume if requested)
    resume_path = args.resume_from_checkpoint or None
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_state()

    # Save final model
    model_save_path = output_dir / "final"
    model.save_pretrained(str(model_save_path))

    # Save training metadata
    meta = {
        "graph_json": args.graph_json,
        "seed": args.seed,
        "model_name": args.model_name,
        "split_cache_json": args.split_cache_json,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_epochs": args.num_epochs,
        "num_negatives_per_state": args.num_negatives_per_state,
        "retrieval_mask_mode": args.retrieval_mask_mode,
        "query_max_length": args.query_max_length,
        "premise_max_length": args.premise_max_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "include_lean4_code": args.include_lean4_code,
        "include_statement": args.include_statement,
        "lean4_code_max_chars": args.lean4_code_max_chars,
        "model_config": model_config,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Training complete. Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
