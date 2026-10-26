#!/usr/bin/env python3
"""Evaluate phase2 selector on Mathlib + DAPS-Collection test splits.

This script is self-contained for the phase2 submission package. It can optionally
build full premise embeddings if they are missing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

# Offline-friendly defaults for transformers
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

from premise_selector.config import MathlibGraphConfig, TYPE_PRIORITY  # noqa: E402
from premise_selector.data import MathlibGraph, DEFAULT_FIXED_SPLIT_CACHE, load_or_create_mathlib_split
from premise_selector.embedding_models import EMBEDDING_MODELS  # noqa: E402
from premise_selector.features import NodeFeatureCache  # noqa: E402
from premise_selector.subgraph import SubgraphConfig, SubgraphBuilder  # noqa: E402
from premise_selector.subgraph_cache import SubgraphCache  # noqa: E402
from premise_selector.text_encoder import LeanTextEncoder, TextEncoderConfig  # noqa: E402
from local_select.query_topk import build_model_from_checkpoint, torch_load  # noqa: E402

DEFAULT_GRAPH_PATH = PROJECT_ROOT / (
    "build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/"
    "lake_graph_with_code_mathlib4-4.22.0-full.json"
)
DEFAULT_SAMPLE_GRAPH_PATH = PROJECT_ROOT / "graph-demo/example_graph.json"
DEFAULT_FILTER_CONFIG = SCRIPT_DIR / "metadata/premise_filter_config.json"
DEFAULT_SPLIT_CACHE_JSON = DEFAULT_FIXED_SPLIT_CACHE
DEFAULT_DAPS_COLLECTION_TEST_JSONL = PROJECT_ROOT / "datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl"
DEFAULT_OUT_CSV = SCRIPT_DIR / "outputs/selector_recall_metrics_phase2.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs/phase2_subgraph"

KS_DEFAULT = [1, 2, 4, 8, 16, 32, 64, 128]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc
            if isinstance(obj, dict):
                yield obj


def load_premise_filter_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[filter] premise filter config not found: {path} (skip)")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[filter] premise filter config read failed: {path} ({exc})")
        return {}


def build_premise_filter(config: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    allowed_types = set(config.get("allowed_subcategories", []) or [])
    blacklist = set(config.get("class_blacklist", []) or [])
    blacklist |= set(config.get("instance_blacklist", []) or [])
    blacklist |= set(config.get("extra_blacklist", []) or [])
    return allowed_types, blacklist


def _priority_rank(subcategory: str) -> int:
    try:
        return TYPE_PRIORITY.index(subcategory)
    except ValueError:
        return len(TYPE_PRIORITY)


def truncate_positive_ids(
    graph: MathlibGraph,
    positive_ids: List[int],
    *,
    qid: Optional[int] = None,
    allowed_ids: Optional[Set[int]] = None,
) -> List[int]:
    max_pos = graph.cfg.negative_sampling.max_positives
    uniq = [int(pid) for pid in set(positive_ids)]
    if allowed_ids is not None:
        uniq = [pid for pid in uniq if pid in allowed_ids]
    if qid is not None:
        uniq = [pid for pid in uniq if pid != int(qid)]
    uniq.sort(key=lambda nid: (_priority_rank(graph.nodes[nid].subcategory), graph.nodes[nid].name))
    return uniq[:max_pos]


def collect_mathlib_positives(
    query_ids: List[int],
    graph: MathlibGraph,
    allowed_ids: Set[int],
) -> List[List[int]]:
    positives_by_query: List[List[int]] = []
    for qid in query_ids:
        raw = list(graph.positives.get(int(qid), set()))
        positives_by_query.append(truncate_positive_ids(graph, raw, qid=int(qid), allowed_ids=allowed_ids))
    return positives_by_query


def build_name_to_ids(graph: MathlibGraph) -> Dict[str, List[int]]:
    name_to_ids: Dict[str, List[int]] = {}
    for node in graph.nodes:
        name_to_ids.setdefault(node.name, []).append(node.node_id)
    return name_to_ids


def resolve_positive_names(
    names: Sequence[Any],
    *,
    graph: MathlibGraph,
    name_to_ids: Dict[str, List[int]],
) -> List[int]:
    resolved: List[int] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        nid = graph.id_by_name.get(name)
        if nid is not None:
            resolved.append(int(nid))
            continue
        cands = name_to_ids.get(name, [])
        if cands:
            resolved.append(int(cands[0]))
    return resolved


def compute_topk(
    query_embs: torch.Tensor,
    premise_embs: torch.Tensor,
    *,
    k: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if query_embs.numel() == 0 or premise_embs.numel() == 0:
        return torch.empty((0, k), dtype=torch.int64)
    query_embs = query_embs.to(device)
    premise_embs = premise_embs.to(device)
    all_topk: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, query_embs.shape[0], batch_size), desc="Top-k"):
            batch = query_embs[start:start + batch_size]
            sims = batch @ premise_embs.T
            k_eff = min(k, sims.shape[1])
            _, top_idx = torch.topk(sims, k=k_eff, dim=1)
            if k_eff < k:
                pad = torch.full((top_idx.shape[0], k - k_eff), -1, dtype=top_idx.dtype, device=top_idx.device)
                top_idx = torch.cat([top_idx, pad], dim=1)
            all_topk.append(top_idx.cpu())
    return torch.cat(all_topk, dim=0) if all_topk else torch.empty((0, k), dtype=torch.int64)


def compute_metrics(
    top_indices: torch.Tensor,
    positives_by_query: List[List[int]],
    premise_ids: List[int],
    *,
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    hit_counts = {k: 0 for k in ks}
    rank_sums = {k: 0.0 for k in ks}
    accuracy_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    recall_sums = {k: 0.0 for k in ks}
    mrr_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    map_sums = {k: 0.0 for k in ks}
    fullrecall_sums = {k: 0.0 for k in ks}
    valid = 0
    pos_counts: List[int] = []

    for i, positives in enumerate(positives_by_query):
        if not positives:
            continue
        valid += 1
        pos_set = set(int(pid) for pid in positives)
        pos_counts.append(len(pos_set))
        pred_idx = top_indices[i].tolist()
        pred_ids = [int(premise_ids[idx]) for idx in pred_idx if idx >= 0]

        for k in ks:
            k_eff = min(k, len(pred_ids))
            if k_eff == 0:
                continue
            topk = pred_ids[:k_eff]
            hits = [pid for pid in topk if pid in pos_set]
            num_hits = len(hits)
            accuracy_sums[k] += 1.0 if num_hits > 0 else 0.0
            precision_sums[k] += num_hits / k_eff
            recall_sums[k] += num_hits / max(len(pos_set), 1)
            if num_hits > 0:
                hit_counts[k] += 1
                first_rank = next((r for r, pid in enumerate(topk, start=1) if pid in pos_set), None)
                if first_rank is not None:
                    mrr_sums[k] += 1.0 / first_rank
                    rank_sums[k] += first_rank

            rel = [1 if pid in pos_set else 0 for pid in topk]
            dcg = sum(rel[j] / math.log2(j + 2.0) for j in range(len(rel))) if rel else 0.0
            ideal_len = min(len(pos_set), k_eff)
            if ideal_len:
                idcg = sum(1.0 / math.log2(j + 2.0) for j in range(ideal_len))
                ndcg_sums[k] += (dcg / idcg) if idcg > 0 else 0.0

            num_correct_so_far = 0
            sum_precisions = 0.0
            for j, pid in enumerate(topk, start=1):
                if pid in pos_set:
                    num_correct_so_far += 1
                    sum_precisions += num_correct_so_far / j
            denom = min(k_eff, len(pos_set))
            if denom:
                map_sums[k] += sum_precisions / denom

            if num_hits == len(pos_set):
                fullrecall_sums[k] += 1.0

    metrics["valid_queries"] = float(valid)
    if pos_counts:
        metrics["avg_positives"] = sum(pos_counts) / len(pos_counts)

    if valid > 0:
        for k in ks:
            metrics[f"accuracy@{k}"] = accuracy_sums[k] / valid
            metrics[f"precision@{k}"] = precision_sums[k] / valid
            metrics[f"recall@{k}"] = recall_sums[k] / valid
            metrics[f"mrr@{k}"] = mrr_sums[k] / valid
            metrics[f"ndcg@{k}"] = ndcg_sums[k] / valid
            metrics[f"map@{k}"] = map_sums[k] / valid
            metrics[f"fullrecall@{k}"] = fullrecall_sums[k] / valid
            metrics[f"hit_rate@{k}"] = hit_counts[k] / valid
            metrics[f"avg_rank@{k}"] = (rank_sums[k] / hit_counts[k]) if hit_counts[k] else 0.0
    return metrics


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({k for row in rows for k in row.keys() if k not in ("date", "selector")})
    fieldnames = ["date", "selector"] + metric_keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_query_ids(path: Path) -> List[int]:
    data = json.loads(path.read_text())
    qids = data.get("query_ids", [])
    return [int(x) for x in qids]


def load_split_cache(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    splits = data.get("splits")
    if not isinstance(splits, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for key, vals in splits.items():
        if isinstance(vals, list):
            out[key] = [int(x) for x in vals]
    return out


def load_query_ids_required(path: Path, split_name: str) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Required split file for {split_name} not found: {path}")
    qids = load_query_ids(path)
    if not qids:
        raise ValueError(f"Split file for {split_name} is empty: {path}")
    return [int(x) for x in qids]


def resolve_graph_path(path: Path) -> Path:
    if path.exists():
        return path
    if DEFAULT_SAMPLE_GRAPH_PATH.exists():
        print(
            "[graph][WARNING] Full graph not found at "
            f"{path}. Falling back to the demo graph at {DEFAULT_SAMPLE_GRAPH_PATH}. "
        )
        return DEFAULT_SAMPLE_GRAPH_PATH
    raise FileNotFoundError(
        f"Graph path not found: {path}. Provide --graph-path explicitly or run build_mathlib_dataset pipeline."
    )


def resolve_mathlib_eval_splits(
    *,
    split_cache_json: Path,
    split_all_file: Optional[Path],
    split_holdout_file: Optional[Path],
    split_random_file: Optional[Path],
) -> Tuple[List[int], List[int], List[int]]:
    if split_cache_json.exists():
        split_cache = load_split_cache(split_cache_json)
        split_all = split_cache.get("test", [])
        split_holdout = split_cache.get("test_holdout", [])
        split_random = split_cache.get("test_random", [])
        if split_all and split_holdout and split_random:
            print(
                "[split] loaded from split cache: "
                f"{split_cache_json} (all={len(split_all)} holdout={len(split_holdout)} random={len(split_random)})"
            )
            return [int(x) for x in split_all], [int(x) for x in split_holdout], [int(x) for x in split_random]
        print(f"[split] split cache found but incomplete, fallback to explicit files: {split_cache_json}")

    if split_all_file is None or split_holdout_file is None or split_random_file is None:
        raise ValueError(
            "split-cache-json is required in this package. "
            "If you must use explicit split files, provide all of: "
            "--mathlib-test-all, --mathlib-test-holdout, --mathlib-test-random."
        )

    split_all = load_query_ids_required(split_all_file, "mathlib_test_all")
    split_holdout = load_query_ids_required(split_holdout_file, "mathlib_test_holdout")
    split_random = load_query_ids_required(split_random_file, "mathlib_test_random")

    all_set = set(split_all)
    missing_holdout = [qid for qid in split_holdout if qid not in all_set]
    missing_random = [qid for qid in split_random if qid not in all_set]
    if missing_holdout:
        raise ValueError("mathlib_test_holdout must be a subset of mathlib_test_all")
    if missing_random:
        raise ValueError("mathlib_test_random must be a subset of mathlib_test_all")

    print(
        "[split] loaded explicit split files: "
        f"all={len(split_all)} holdout={len(split_holdout)} random={len(split_random)}"
    )
    return split_all, split_holdout, split_random


def build_selector_candidate_pool(
    candidate_ids: List[int],
    selector_premise_ids: List[int],
    selector_premise_embs: torch.Tensor,
) -> Tuple[List[int], torch.Tensor]:
    selector_id_to_idx = {int(pid): idx for idx, pid in enumerate(selector_premise_ids)}
    keep_indices: List[int] = []
    kept_ids: List[int] = []
    for pid in candidate_ids:
        idx = selector_id_to_idx.get(int(pid))
        if idx is None:
            continue
        keep_indices.append(idx)
        kept_ids.append(int(pid))
    if not keep_indices:
        return [], selector_premise_embs[:0]
    index_tensor = torch.tensor(keep_indices, device=selector_premise_embs.device)
    filtered_embs = selector_premise_embs.index_select(0, index_tensor)
    return kept_ids, filtered_embs


def encode_phase2_queries(
    queries: List[str],
    *,
    model,
    text_encoder: LeanTextEncoder,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(queries), batch_size), desc="Encode queries"):
            batch = queries[start:start + batch_size]
            stmt_emb = text_encoder(batch, max_length=max_length).to(device)
            emb = model.encode_queries(stmt_emb)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))


def encode_premises_batch(
    model,
    builder: SubgraphBuilder,
    premise_ids: List[int],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(premise_ids), batch_size), desc="Encode premises"):
            batch_ids = premise_ids[start:start + batch_size]
            batch_inputs = builder.build_batch(batch_ids, device=device, seed_text_mode="premise")
            emb = model.encode_premises(batch_inputs)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phase2 selector on packaged test sets.")
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--filter-config", type=Path, default=DEFAULT_FILTER_CONFIG)
    parser.add_argument("--mathlib-test-all", type=Path, default=None)
    parser.add_argument("--mathlib-test-holdout", type=Path, default=None)
    parser.add_argument("--mathlib-test-random", type=Path, default=None)
    parser.add_argument(
        "--split-cache-json",
        type=Path,
        default=DEFAULT_SPLIT_CACHE_JSON,
        help="Preferred split cache JSON. If valid, overrides --mathlib-test-* files.",
    )
    parser.add_argument(
        "--daps-collection-test-jsonl",
        "--combined-test-jsonl",
        dest="daps_collection_test_jsonl",
        type=Path,
        default=DEFAULT_DAPS_COLLECTION_TEST_JSONL,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--premise-embeddings", type=Path, default=None)
    parser.add_argument("--embeddings-out", type=Path, default=None)
    parser.add_argument("--text-encoder-path", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--premise-batch", type=int, default=64)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--subgraph-cache-dir", type=Path, default=DEFAULT_OUT_DIR / "subgraph_cache")
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_OUT_DIR / "feature_cache")
    parser.add_argument("--ks", type=str, default=",".join(str(k) for k in KS_DEFAULT))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    args.graph_path = resolve_graph_path(args.graph_path)
    if args.split_cache_json is not None:
        args.split_cache_json, _ = load_or_create_mathlib_split(args.graph_path, args.split_cache_json)
    print(f"[env] device={device}")
    print(f"[graph] loading {args.graph_path}")
    graph = MathlibGraph(MathlibGraphConfig(
        graph_path=str(args.graph_path),
        split_cache_json=str(args.split_cache_json) if args.split_cache_json is not None else None,
    ))

    filter_cfg = load_premise_filter_config(args.filter_config)
    allowed_types, blacklist = build_premise_filter(filter_cfg)
    if not allowed_types:
        allowed_types = set(graph.cfg.premise_types)
    print(f"[filter] allowed_types={len(allowed_types)} blacklist={len(blacklist)}")

    base_premise_ids = [node.node_id for node in graph.nodes if node.subcategory in graph.cfg.premise_types]
    filtered_premise_ids = [
        pid for pid in base_premise_ids
        if graph.nodes[pid].subcategory in allowed_types and graph.nodes[pid].name not in blacklist
    ]
    print(f"[candidate] {len(base_premise_ids)} -> {len(filtered_premise_ids)}")

    # Load eval splits from explicit files only (no fallback to full query set).
    math_test_all, math_test_holdout, math_test_random = resolve_mathlib_eval_splits(
        split_cache_json=args.split_cache_json,
        split_all_file=args.mathlib_test_all,
        split_holdout_file=args.mathlib_test_holdout,
        split_random_file=args.mathlib_test_random,
    )

    # DAPS model + text encoder
    ckpt = torch_load(args.checkpoint, map_location=device)
    ckpt_embedding = ckpt.get("embedding_model") or ckpt.get("args", {}).get("embedding", "alibaba-gte-modernbert-ft")

    if args.text_encoder_path and ckpt_embedding in EMBEDDING_MODELS:
        EMBEDDING_MODELS[ckpt_embedding].model_path = str(args.text_encoder_path)

    max_length = ckpt.get("args", {}).get("max_length", TextEncoderConfig().max_length)
    text_encoder = LeanTextEncoder(
        TextEncoderConfig(trainable=False, embedding_model=ckpt_embedding, max_length=max_length),
        device=device,
    )
    model = build_model_from_checkpoint(ckpt, graph, device=device)

    # Premise embeddings (load or build)
    premise_emb_path = args.premise_embeddings
    if premise_emb_path is not None and premise_emb_path.exists():
        print(f"[premise] loading embeddings from {premise_emb_path}")
        premise_data = torch_load(premise_emb_path, map_location=device)
        premise_embs = premise_data["embeddings"].to(device)
        premise_ids = list(premise_data["premise_ids"])
    else:
        print("[premise] embeddings not found; encoding full library...")
        subgraph_mode = ckpt.get("args", {}).get("subgraph_mode", 0)
        max_nodes = ckpt.get("args", {}).get("max_nodes", 64)
        sub_cfg = SubgraphConfig(max_nodes=max_nodes, subgraph_mode=subgraph_mode)
        subgraph_cache = SubgraphCache(graph, sub_cfg, cache_device="cpu")
        args.subgraph_cache_dir.mkdir(parents=True, exist_ok=True)
        subgraph_cache.ensure_cache(args.subgraph_cache_dir)

        feature_cache = NodeFeatureCache(
            graph,
            text_encoder,
            cache_device="cpu",
            sub_node_max_length=ckpt.get("args", {}).get("sub_node_max_length", 512),
            full_node_max_length=ckpt.get("args", {}).get("max_length", 512),
            pattern_mode="none",
        )
        builder = SubgraphBuilder(
            graph,
            feature_cache,
            sub_cfg,
            subgraph_cache=subgraph_cache,
        )
        premise_ids = list(base_premise_ids)
        premise_embs = encode_premises_batch(
            model,
            builder,
            premise_ids,
            device=device,
            batch_size=args.premise_batch,
        ).to(device)

        if args.embeddings_out is not None:
            args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "premise_ids": premise_ids,
                "embeddings": premise_embs.cpu(),
                "embedding_model": ckpt_embedding,
                "max_length": max_length,
            }
            torch.save(payload, args.embeddings_out)
            print(f"[premise] saved embeddings to {args.embeddings_out}")

    selector_ids, selector_embs = build_selector_candidate_pool(
        filtered_premise_ids, premise_ids, premise_embs
    )
    if not selector_ids:
        raise RuntimeError("No candidates after filtering; check filter config / embeddings.")

    # Mathlib queries
    math_queries = [graph.nodes[qid].statement or "" for qid in math_test_all]
    math_query_embs = encode_phase2_queries(
        math_queries,
        model=model,
        text_encoder=text_encoder,
        device=device,
        max_length=max_length,
        batch_size=args.query_batch,
    )
    topk_all = compute_topk(
        math_query_embs,
        selector_embs,
        k=max(ks),
        batch_size=args.query_batch,
        device=device,
    )

    id_to_row = {qid: i for i, qid in enumerate(math_test_all)}
    holdout_rows = [id_to_row[qid] for qid in math_test_holdout if qid in id_to_row]
    random_rows = [id_to_row[qid] for qid in math_test_random if qid in id_to_row]

    positives_all = collect_mathlib_positives(math_test_all, graph, set(selector_ids))
    positives_holdout = [positives_all[i] for i in holdout_rows]
    positives_random = [positives_all[i] for i in random_rows]

    metrics_holdout = compute_metrics(topk_all[holdout_rows], positives_holdout, selector_ids, ks=ks)
    metrics_random = compute_metrics(topk_all[random_rows], positives_random, selector_ids, ks=ks)

    # DAPS-Collection test subset
    daps_collection_records = (
        list(iter_jsonl(args.daps_collection_test_jsonl))
        if args.daps_collection_test_jsonl.exists()
        else []
    )
    name_to_ids = build_name_to_ids(graph)
    combined_queries = [r.get("query_text", "") for r in daps_collection_records]
    combined_pos_ids: List[List[int]] = []
    empty_positive_rows: List[str] = []
    for r in daps_collection_records:
        pos: List[int] = []
        if isinstance(r.get("gold"), list):
            pos = [int(pid) for pid in r.get("gold", [])]
        elif isinstance(r.get("positive_ids"), list):
            pos = [int(pid) for pid in r.get("positive_ids", [])]
        elif isinstance(r.get("positives"), list) or isinstance(r.get("positive_names"), list):
            pos_names = r.get("positives") or r.get("positive_names") or []
            pos = resolve_positive_names(pos_names, graph=graph, name_to_ids=name_to_ids)
        pos = truncate_positive_ids(graph, pos, allowed_ids=set(selector_ids))
        if not pos:
            row_id = str(r.get("query_key", f"row_{len(combined_pos_ids)}"))
            empty_positive_rows.append(row_id)
        combined_pos_ids.append(pos)

    if empty_positive_rows:
        preview = ", ".join(empty_positive_rows[:10])
        raise RuntimeError(
            "DAPS-Collection contains queries with empty resolved positives after filtering. "
            f"First rows: {preview}. "
            "Check whether the JSONL uses positive names and whether they can be resolved in the current graph."
        )

    combined_query_embs = encode_phase2_queries(
        combined_queries,
        model=model,
        text_encoder=text_encoder,
        device=device,
        max_length=max_length,
        batch_size=args.query_batch,
    )
    combined_topk = compute_topk(
        combined_query_embs,
        selector_embs,
        k=max(ks),
        batch_size=args.query_batch,
        device=device,
    )
    metrics_combined = compute_metrics(combined_topk, combined_pos_ids, selector_ids, ks=ks)

    row: Dict[str, Any] = {
        "date": dt.date.today().isoformat(),
        "selector": "DAPS",
    }
    row.update(flatten_metrics("Mathlib4-H", metrics_holdout))
    row.update(flatten_metrics("Mathlib4-R", metrics_random))
    row.update(flatten_metrics("DAPS_collection", metrics_combined))

    write_metrics_csv(args.out_csv, [row])
    print(f"[done] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
