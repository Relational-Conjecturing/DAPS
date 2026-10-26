from __future__ import annotations

import os
# Ensure transformers never tries to import TensorFlow/Flax in this process.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add phase2 code so premise_selector can be imported.
ROOT = Path(__file__).resolve().parent
PHASE2_DIR = (ROOT / "../phase2_train_with_subgraph").resolve()
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from premise_selector.embedding_models import EMBEDDING_MODELS, EmbeddingModelConfig
from premise_selector.model import PremiseSelectorModel, RelGTPremiseEncoder
from premise_selector.serialization_utils import load_torch_file
from premise_selector.text_encoder import LeanTextEncoder, TextEncoderConfig

from blueprint_data import BlueprintGraph, BlueprintQueryDataset, ensure_splits
from config import TrainConfig
from utils import (
    BlueprintFeatureCache,
    build_premise_inputs,
    set_seed,
)


PHASE2_GRAPH_SPECIFIC_KEYS = {
    "premise_encoder.conv.c_idx": "Blueprint has a new node ID space and centroid assignments.",
    "premise_encoder.type_encoder.embedding.weight": "Blueprint node types differ from Mathlib node types.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase3 Blueprint train-and-eval pipeline")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--splits-path", type=str, default=None)
    parser.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Phase2 checkpoint for transfer; Blueprint node/type state is initialized afresh.",
    )
    parser.add_argument("--text-encoder-path", type=str, default=None)
    parser.add_argument("--text-encoder-key", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--num-repo-negatives", type=int, default=None)
    parser.add_argument("--num-global-negatives", type=int, default=None)
    parser.add_argument("--max-subgraph-nodes", type=int, default=None)
    parser.add_argument("--subgraph-select-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to model_last.pt for resuming training")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow random initialization when checkpoint is missing. Disabled by default to avoid silent fallback.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def register_custom_embedding_model(key: str, path: str, max_length: int) -> None:
    if not key:
        return
    if not path:
        return
    cfg = EmbeddingModelConfig(
        name=key,
        model_path=path,
        max_length=max_length,
        use_t5=False,
        pooling_method="mean",
        normalize=True,
        padding_side="right",
        use_flash_attention=False,
    )
    EMBEDDING_MODELS[key] = cfg


def _validated_checkpoint_state(
    model: PremiseSelectorModel,
    data: object,
    checkpoint_path: str,
    *,
    transfer_from_phase2: bool = False,
) -> Dict[str, torch.Tensor]:
    state_dict = None
    for key in ("model", "model_state", "state_dict", "model_state_dict"):
        if isinstance(data, dict) and key in data:
            state_dict = data[key]
            break
    if state_dict is None and isinstance(data, dict):
        state_dict = data
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")
    target_state = model.state_dict()
    reset_keys = set(PHASE2_GRAPH_SPECIFIC_KEYS) if transfer_from_phase2 else set()
    required_keys = set(target_state) - reset_keys
    matched_keys = required_keys.intersection(state_dict)
    if not matched_keys:
        raise ValueError(
            f"Checkpoint has no parameter keys matching this model: {checkpoint_path}. "
            "Expected a model state dictionary or a supported checkpoint wrapper."
        )
    missing = sorted(required_keys - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target_state))
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint keys do not match this model: {checkpoint_path}; no parameters were loaded. "
            f"Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}"
        )
    invalid_keys = [key for key in sorted(matched_keys) if not torch.is_tensor(state_dict[key])]
    if invalid_keys:
        raise ValueError(f"Checkpoint has non-tensor model entries: {invalid_keys[:8]}")
    mismatches = [
        f"{key}: checkpoint={tuple(state_dict[key].shape)} model={tuple(target_state[key].shape)}"
        for key in sorted(matched_keys)
        if state_dict[key].shape != target_state[key].shape
    ]
    if mismatches:
        raise RuntimeError(
            f"Checkpoint shape mismatch in {checkpoint_path}; no parameters were loaded. "
            "Only Phase2 node-index and node-type state may be reset during transfer. "
            + "; ".join(mismatches[:8])
        )
    return {key: state_dict[key] for key in target_state if key not in reset_keys}


def load_checkpoint(
    model: PremiseSelectorModel,
    checkpoint_path: str,
    device: torch.device,
    *,
    allow_random_init: bool = False,
) -> None:
    if not checkpoint_path:
        if allow_random_init:
            print("[Checkpoint] No checkpoint path provided; using random init (explicitly allowed).")
            return
        raise FileNotFoundError(
            "No checkpoint path provided. Pass --checkpoint-path, or add --allow-random-init to run from scratch."
        )
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        if allow_random_init:
            print(f"[Checkpoint] Path not found: {checkpoint_path}; using random init (explicitly allowed).")
            return
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Provide --checkpoint-path, or add --allow-random-init to run from scratch."
        )
    print(f"[Checkpoint] Loading {checkpoint_path}")
    data = load_torch_file(ckpt_path, map_location="cpu")
    state_dict = _validated_checkpoint_state(model, data, checkpoint_path, transfer_from_phase2=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = set(PHASE2_GRAPH_SPECIFIC_KEYS).intersection(model.state_dict())
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(f"Unexpected transfer result: missing={missing}, unexpected={unexpected}")
    target_state = model.state_dict()
    reinitialized = {
        key: {"shape": list(target_state[key].shape), "reason": PHASE2_GRAPH_SPECIFIC_KEYS[key]}
        for key in sorted(expected_missing)
    }
    print(
        f"[Checkpoint] Transferred {len(state_dict)} tensor entries from Phase2; "
        f"kept {len(reinitialized)} Blueprint-specific entries at their fresh initialization."
    )
    for key, entry in reinitialized.items():
        print(f"[Checkpoint] Reinitialized {key}, shape={entry['shape']}: {entry['reason']}")


def load_resume_checkpoint(
    model: PremiseSelectorModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    resume_path: str,
    device: torch.device,
) -> tuple[int, float]:
    ckpt_path = Path(resume_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    print(f"[Resume] Loading {resume_path}")
    data = load_torch_file(ckpt_path, map_location="cpu")
    state_dict = _validated_checkpoint_state(model, data, resume_path)
    model.load_state_dict(state_dict, strict=True)
    opt_state = data.get("optimizer")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    sched_state = data.get("scheduler")
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
    start_epoch = int(data.get("epoch", 0)) + 1
    best_val = float(data.get("best_val", -1.0))
    return start_epoch, best_val


def multi_positive_infonce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    numerator = (exp_logits * positive_mask).sum(dim=1)
    denominator = exp_logits.sum(dim=1).clamp_min(1e-9)
    prob_sum = numerator / denominator
    loss = -torch.log(prob_sum + 1e-9)
    return loss.mean()


def precompute_premise_embeddings(
    model: PremiseSelectorModel,
    graph: BlueprintGraph,
    feature_cache: BlueprintFeatureCache,
    device: torch.device,
    *,
    max_nodes: int,
    chunk_size: int,
    seed_base: int,
) -> torch.Tensor:
    model.eval()
    all_ids = list(range(len(graph.nodes)))
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(all_ids), chunk_size), desc="Encoding premises", leave=False):
            chunk_ids = all_ids[start : start + chunk_size]
            inputs = build_premise_inputs(
                chunk_ids,
                graph,
                feature_cache,
                device=device,
                max_nodes=max_nodes,
                seed_base=seed_base,
            )
            emb = model.encode_premises(inputs).detach()
            chunks.append(emb)
    if not chunks:
        return torch.empty((0, model.query_head.net[-1].out_features), device=device)
    return torch.cat(chunks, dim=0)


def evaluate(
    model: PremiseSelectorModel,
    graph: BlueprintGraph,
    feature_cache: BlueprintFeatureCache,
    query_ids: Sequence[int],
    device: torch.device,
    *,
    ks: Sequence[int],
    max_nodes: int,
    chunk_size: int,
    save_embeddings_path: Path | None = None,
    embedding_meta: Dict[str, object] | None = None,
) -> Dict[str, float]:
    model.eval()
    total_nodes = len(graph.nodes)
    all_ids = list(range(total_nodes))
    all_embs = precompute_premise_embeddings(
        model,
        graph,
        feature_cache,
        device,
        max_nodes=max_nodes,
        chunk_size=chunk_size,
        seed_base=0,
    )
    if save_embeddings_path is not None:
        payload = {
            "premise_ids": all_ids,
            "embeddings": all_embs.detach().cpu(),
            "meta": embedding_meta or {},
        }
        save_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_embeddings_path)
        print(f"[Eval] Saved premise embeddings to {save_embeddings_path}")

    metrics_global: Dict[str, List[float]] = {f"global/recall@{k}": [] for k in ks}
    status_values = ("with_lean", "nlp_only")
    metrics_global_status: Dict[str, List[float]] = {
        f"global/recall@{k}_{status}": [] for status in status_values for k in ks
    }

    total_pos = 0
    total_queries = 0
    lean_statuses = {"all_formalized", "statement_formalized"}
    strict_query_counts = {"with_lean": 0, "nlp_only": 0, "mixed": 0}
    strict_pos_counts = {"with_lean": 0, "nlp_only": 0, "mixed": 0}

    with torch.no_grad():
        for qid in tqdm(query_ids, desc="Evaluating", leave=False):
            query = graph.get_node(qid)
            positive_ids = list(query.outgoing)
            if not positive_ids:
                continue
            total_queries += 1
            total_pos += len(positive_ids)
            statuses = [graph.get_node(pid).proof_status for pid in positive_ids]
            all_with_lean = all(status in lean_statuses for status in statuses)
            all_nlp_only = all(status not in lean_statuses for status in statuses)
            if all_with_lean:
                strict_group = "with_lean"
            elif all_nlp_only:
                strict_group = "nlp_only"
            else:
                strict_group = "mixed"
            strict_query_counts[strict_group] += 1
            strict_pos_counts[strict_group] += len(positive_ids)

            query_emb = feature_cache.get([qid], mode="query", device=device)
            q_proj = model.encode_queries(query_emb)

            # Global candidates (all nodes)
            if total_nodes > 0:
                scores_global = (q_proj @ all_embs.T).squeeze(0)
                if qid < scores_global.shape[0]:
                    scores_global[qid] = -1.0e9
                ranked_indices = torch.argsort(scores_global, descending=True).tolist()
                positive_set = set(positive_ids)
                for k in ks:
                    topk_ids = ranked_indices[:k]
                    hits = len(positive_set.intersection(topk_ids))
                    metrics_global[f"global/recall@{k}"].append(hits / len(positive_set))
                    if strict_group in ("with_lean", "nlp_only"):
                        metrics_global_status[f"global/recall@{k}_{strict_group}"].append(
                            hits / len(positive_set)
                        )

    if total_queries > 0:
        avg_total = total_pos / total_queries
        avg_with_lean = strict_pos_counts["with_lean"] / max(strict_query_counts["with_lean"], 1)
        avg_nlp_only = strict_pos_counts["nlp_only"] / max(strict_query_counts["nlp_only"], 1)
        avg_mixed = strict_pos_counts["mixed"] / max(strict_query_counts["mixed"], 1)
        print(
            "[Eval] Avg positives per query: "
            f"total={avg_total:.2f}, "
            f"with_lean={avg_with_lean:.2f}, "
            f"nlp_only={avg_nlp_only:.2f}, "
            f"mixed={avg_mixed:.2f} "
            f"(queries={total_queries}, with_lean_queries={strict_query_counts['with_lean']}, "
            f"nlp_only_queries={strict_query_counts['nlp_only']}, mixed_queries={strict_query_counts['mixed']})"
        )

    summary: Dict[str, float] = {}
    for key, vals in metrics_global.items():
        summary[key] = sum(vals) / len(vals) if vals else 0.0
    for key, vals in metrics_global_status.items():
        summary[key] = sum(vals) / len(vals) if vals else 0.0
    return summary


def train_one_epoch(
    model: PremiseSelectorModel,
    graph: BlueprintGraph,
    feature_cache: BlueprintFeatureCache,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    *,
    num_repo_negatives: int,
    num_global_negatives: int,
    max_nodes: int,
    log_every: int,
    epoch_seed: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    rng = random.Random(epoch_seed)

    for step, batch in enumerate(loader, start=1):
        batch_ids = batch.tolist() if torch.is_tensor(batch) else list(batch)
        if not batch_ids:
            continue

        optimizer.zero_grad(set_to_none=True)

        query_embs = feature_cache.get(batch_ids, mode="query", device=device)

        # Step 1: sample per-query candidate sets.
        batch_candidates_list = []  # [(qid, positives, negatives), ...]
        for qid in batch_ids:
            query = graph.get_node(qid)
            positives = list(query.outgoing)
            if not positives:
                batch_candidates_list.append((qid, [], []))
                continue
            negatives = graph.sample_negatives(
                qid,
                num_repo_negatives,
                num_global_negatives,
                rng=rng,
            )
            batch_candidates_list.append((qid, positives, negatives))

        # Step 2: collect all candidates in the batch (for in-batch negatives).
        all_batch_candidates = set()
        for qid, positives, negatives in batch_candidates_list:
            all_batch_candidates.update(positives)
            all_batch_candidates.update(negatives)
        all_batch_candidates = list(all_batch_candidates)

        if not all_batch_candidates:
            continue

        # Step 3: encode all batch candidates once.
        premise_inputs = build_premise_inputs(
            all_batch_candidates,
            graph,
            feature_cache,
            device=device,
            max_nodes=max_nodes,
            rng=rng,
        )
        all_premises_proj = model.encode_premises(premise_inputs)  # [num_candidates, proj_dim]

        # Step 4: compute per-query loss using all batch candidates as the negative pool.
        batch_loss = 0.0
        valid_count = 0

        for idx, (qid, positives, negatives) in enumerate(batch_candidates_list):
            if not positives:
                continue

            # Positive set for this query.
            positive_set = set(positives)

            # Build positive_mask over all batch candidates.
            positive_mask = torch.zeros((1, len(all_batch_candidates)), device=device)
            for i, cand_id in enumerate(all_batch_candidates):
                if cand_id in positive_set:
                    positive_mask[0, i] = 1.0

            # Skip if no positive is marked (should not happen in normal data).
            if positive_mask.sum() == 0:
                continue

            # Compute logits of this query against all batch candidates.
            q_proj = model.encode_queries(query_embs[idx : idx + 1])
            logits = model.compute_logits(q_proj, all_premises_proj)

            loss = multi_positive_infonce(logits, positive_mask)
            batch_loss += loss
            valid_count += 1

        if valid_count == 0:
            continue

        batch_loss = batch_loss / valid_count
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += batch_loss.item()
        total_batches += 1

        if log_every and step % log_every == 0:
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            print(
                f"[Train] step={step} loss={batch_loss.item():.4f} lr={lr:.6f}"
            )

    return total_loss / max(total_batches, 1)


def _hash_full_id(full_id: str) -> str:
    import hashlib
    return hashlib.sha256(full_id.encode("utf-8")).hexdigest()[:8]


def save_sample_case(graph: BlueprintGraph, query_id: int, output_path: Path) -> None:
    node = graph.get_node(query_id)
    payload = {
        "query_full_id": node.full_id,
        "repo": node.repo,
        "nlp_text_preview": (node.nlp_text or "")[:400],
        "num_dependencies": len(node.outgoing),
        "dependency_ids_preview": [
            {
                "raw_id": graph.get_node(nid).raw_id,
                "full_id_hash": _hash_full_id(graph.get_node(nid).full_id),
            }
            for nid in node.outgoing[:10]
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Debug] Saved sample case to {output_path}")


def save_filtered_summary_csv(
    graph: BlueprintGraph,
    summary_path: str,
    output_dir: Path,
) -> None:
    src_path = Path(summary_path)
    if not src_path.exists():
        print(f"[Summary] summary.csv not found: {summary_path}")
        return

    repo_counts: Dict[str, Dict[str, int]] = {}
    for repo, node_ids in graph.repo_nodes.items():
        counts = {"all_formalized": 0, "statement_formalized": 0, "nlp_only": 0}
        for nid in node_ids:
            status = graph.get_node(nid).proof_status
            if status in counts:
                counts[status] += 1
        repo_counts[repo] = counts

    out_path = output_dir / "summary_filtered.csv"
    with src_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"[Summary] Empty summary.csv: {summary_path}")
            return
        fieldnames = list(reader.fieldnames)
        for extra in ("with_lean", "nlp_only"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        with out_path.open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            kept = 0
            for row in reader:
                jsonl_path = row.get("jsonl_path", "")
                repo_key = Path(jsonl_path).stem if jsonl_path else ""
                if repo_key not in graph.repo_nodes:
                    continue
                counts = repo_counts.get(repo_key, {})
                with_lean = counts.get("all_formalized", 0) + counts.get("statement_formalized", 0)
                row["with_lean"] = str(with_lean)
                row["nlp_only"] = str(counts.get("nlp_only", 0))
                writer.writerow(row)
                kept += 1
    print(f"[Summary] Saved filtered summary to {out_path} (rows={kept})")


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_root:
        cfg.output_root = args.output_root
    if args.splits_path:
        cfg.splits_path = args.splits_path
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.text_encoder_path:
        cfg.text_encoder_path = args.text_encoder_path
    if args.text_encoder_key:
        cfg.text_encoder_key = args.text_encoder_key
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.min_lr is not None:
        cfg.min_lr = args.min_lr
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.early_stop_min_delta = args.early_stop_min_delta
    if args.num_repo_negatives is not None:
        cfg.num_repo_negatives = args.num_repo_negatives
    if args.num_global_negatives is not None:
        cfg.num_global_negatives = args.num_global_negatives
    if args.num_negatives is not None and args.num_repo_negatives is None and args.num_global_negatives is None:
        repo_n = max(0, int(args.num_negatives) // 2)
        global_n = max(0, int(args.num_negatives) - repo_n)
        cfg.num_repo_negatives = repo_n
        cfg.num_global_negatives = global_n
    if args.max_subgraph_nodes:
        cfg.max_subgraph_nodes = args.max_subgraph_nodes
    if args.subgraph_select_k is not None:
        cfg.subgraph_select_k = args.subgraph_select_k
    if args.device:
        cfg.device = args.device

    device = resolve_device(cfg.device)
    print(f"[Config] device={device}")

    set_seed(cfg.seed)

    resume_path = args.resume
    if resume_path:
        output_dir = Path(resume_path).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Resume] Using output dir {output_dir}")
    else:
        output_dir = cfg.make_output_dir()
        config_path = output_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
            print(f"[Config] Saved to {config_path}")

    graph = BlueprintGraph(cfg.data_dir, shuffle_seed=cfg.seed)
    train_ids, val_ids, test_ids = ensure_splits(
        graph,
        splits_path=cfg.splits_path,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    if train_ids:
        save_sample_case(graph, train_ids[0], output_dir / "sample_query.json")
    if getattr(cfg, "summary_csv_path", None):
        save_filtered_summary_csv(graph, cfg.summary_csv_path, output_dir)

    register_custom_embedding_model(cfg.text_encoder_key, cfg.text_encoder_path, cfg.text_encoder_max_length)
    text_cfg = TextEncoderConfig(
        embedding_model=cfg.text_encoder_key,
        max_length=cfg.text_encoder_max_length,
        trainable=not cfg.freeze_text_encoder,
    )
    text_encoder = LeanTextEncoder(text_cfg, device=device)

    feature_cache = BlueprintFeatureCache(
        graph,
        text_encoder,
        cache_device="cpu",
        chunk_size=16,
        max_length=cfg.text_encoder_max_length,
    )

    encoder = RelGTPremiseEncoder(
        num_nodes=len(graph.nodes),
        max_neighbor_hop=cfg.subgraph_hop,
        node_type_map=graph.type_to_idx,
        feature_dim=feature_cache.embedding_dim,
        channels=cfg.hidden_dim,
        local_num_layers=cfg.local_layers,
        global_dim=None,
        heads=cfg.heads,
        ff_dropout=cfg.dropout,
        attn_dropout=cfg.dropout,
        conv_type="full",
        num_centroids=4096,
        sample_node_len=cfg.max_subgraph_nodes,
        select_top_k=cfg.subgraph_select_k,
        selector_dropout=cfg.dropout,
    )

    model = PremiseSelectorModel(
        encoder=encoder,
        query_dim=feature_cache.embedding_dim,
        proj_dim=cfg.proj_dim,
        temperature=cfg.temperature,
    ).to(device)

    if not resume_path:
        load_checkpoint(
            model, cfg.checkpoint_path, device, allow_random_init=args.allow_random_init,
        )

    train_dataset = BlueprintQueryDataset(train_ids)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    total_steps = max(len(train_loader) * cfg.epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    min_lr = max(0.0, float(getattr(cfg, "min_lr", 0.0)))
    min_lr_ratio = min_lr / cfg.lr if cfg.lr > 0 else 0.0
    min_lr_ratio = min(min_lr_ratio, 1.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ks = [int(x) for x in cfg.eval_ks.split(",") if x.strip()]


    best_val = -1.0
    no_improve = 0
    best_path = output_dir / "model_best.pt"
    last_path = output_dir / "model_last.pt"
    metrics_csv_path = output_dir / "metrics.csv"
    wrote_header = metrics_csv_path.exists()

    start_epoch = 1
    if resume_path:
        start_epoch, best_val = load_resume_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )
        print(f"[Resume] start_epoch={start_epoch} best_val={best_val:.4f}")

    if args.eval_only:
        print("[Eval] eval-only mode; skip training and run val/test evaluation.")
        val_metrics = evaluate(
            model,
            graph,
            feature_cache,
            val_ids,
            device,
            ks=ks,
            max_nodes=cfg.max_subgraph_nodes,
            chunk_size=cfg.eval_chunk_size,
        )
        print(f"[Eval-only] Val: {val_metrics}")

        test_metrics = evaluate(
            model,
            graph,
            feature_cache,
            test_ids,
            device,
            ks=ks,
            max_nodes=cfg.max_subgraph_nodes,
            chunk_size=cfg.eval_chunk_size,
            save_embeddings_path=output_dir / "premise_embeddings_test.pt",
            embedding_meta={
                "checkpoint": str(last_path) if resume_path else cfg.checkpoint_path,
                "text_encoder": cfg.text_encoder_path,
                "embedding_model": cfg.text_encoder_key,
                "max_length": cfg.text_encoder_max_length,
                "max_nodes": cfg.max_subgraph_nodes,
                "seed_base": 0,
                "tag": "test_eval_only",
            },
        )
        test_path = output_dir / "test_metrics.json"
        test_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(f"[Eval-only] Test metrics saved to {test_path}")
        print(f"[Eval-only] Test: {test_metrics}")
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(
            model,
            graph,
            feature_cache,
            train_loader,
            optimizer,
            scheduler,
            device,
            num_repo_negatives=cfg.num_repo_negatives,
            num_global_negatives=cfg.num_global_negatives,
            max_nodes=cfg.max_subgraph_nodes,
            log_every=cfg.log_every,
            epoch_seed=cfg.seed + epoch,
        )

        val_metrics = evaluate(
            model,
            graph,
            feature_cache,
            val_ids,
            device,
            ks=ks,
            max_nodes=cfg.max_subgraph_nodes,
            chunk_size=cfg.eval_chunk_size,
        )
        elapsed = time.time() - start
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val={val_metrics} time={elapsed:.1f}s")

        val_score = val_metrics.get("global/recall@32", 0.0)
        if val_score > best_val + cfg.early_stop_min_delta:
            best_val = val_score
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
            print(f"[Checkpoint] New best model saved to {best_path}")
        else:
            no_improve += 1

        metrics_path = output_dir / f"metrics_epoch_{epoch}.json"
        metrics_path.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

        row = {"epoch": epoch, "loss": avg_loss, "time_sec": elapsed, "val_score": val_score}
        row.update(val_metrics)
        with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not wrote_header:
                writer.writeheader()
                wrote_header = True
            writer.writerow(row)

        # Always save last checkpoint for resume
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            },
            last_path,
        )

        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            print(
                f"[EarlyStop] No improvement in {no_improve} epochs. "
                f"Stopping at epoch {epoch}."
            )
            break

    if best_path.exists():
        print("[Eval] Loading best checkpoint for final test...")
        data = load_torch_file(best_path, map_location="cpu")
        model.load_state_dict(_validated_checkpoint_state(model, data, str(best_path)), strict=True)

    test_metrics = evaluate(
        model,
        graph,
        feature_cache,
        test_ids,
        device,
        ks=ks,
        max_nodes=cfg.max_subgraph_nodes,
        chunk_size=cfg.eval_chunk_size,
        save_embeddings_path=output_dir / "premise_embeddings_test.pt",
        embedding_meta={
            "checkpoint": str(best_path) if best_path.exists() else cfg.checkpoint_path,
            "text_encoder": cfg.text_encoder_path,
            "embedding_model": cfg.text_encoder_key,
            "max_length": cfg.text_encoder_max_length,
            "max_nodes": cfg.max_subgraph_nodes,
            "seed_base": 0,
            "tag": "test",
        },
    )
    test_path = output_dir / "test_metrics.json"
    test_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"[Eval] Test metrics saved to {test_path}")
    print(f"[Eval] Test: {test_metrics}")

    # Append test metrics to CSV for easy tracking
    test_row = {"epoch": "test", "loss": "", "time_sec": "", "val_score": ""}
    test_row.update(test_metrics)
    with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_row.keys()))
        if not wrote_header:
            writer.writeheader()
            wrote_header = True
        writer.writerow(test_row)


if __name__ == "__main__":
    main()
