#!/usr/bin/env python3
"""
Evaluate the phase2 premise selector on aligned Mathlib / DAPS-Collection splits.

Ground-truth and candidate filtering follow the packaged phase2 filter configuration:
  - candidate pool = graph.cfg.premise_types
  - apply premise_filter_config.json (allowed_subcategories + blacklist)
  - DAPS-Collection positives rebuilt from source JSONL

Metrics include accuracy/precision/recall/mrr/ndcg/map/fullrecall/hit_rate/avg_rank.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

# Keep transformers fully offline (models are expected to be cached already).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
PROJECT_ROOT = REPO_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from premise_selector.config import MathlibGraphConfig, TYPE_PRIORITY  # noqa: E402
from premise_selector.data import MathlibGraph, DEFAULT_FIXED_SPLIT_CACHE, load_or_create_mathlib_split
from premise_selector.text_encoder import LeanTextEncoder, TextEncoderConfig  # noqa: E402
from local_select.query_topk import build_model_from_checkpoint, torch_load  # noqa: E402
from premise_selector.embedding_models import EMBEDDING_MODELS  # noqa: E402


DEFAULT_GRAPH_PATH = PROJECT_ROOT / (
    "build_mathlib_dataset/generated_graphs/mathlib4-4.22.0-full/"
    "lake_graph_with_code_mathlib4-4.22.0-full.json"
)
DEFAULT_SPLIT_CACHE_JSON = DEFAULT_FIXED_SPLIT_CACHE
DEFAULT_MATHLIB_TEST_ALL: Optional[Path] = None
DEFAULT_MATHLIB_TEST_HOLDOUT: Optional[Path] = None
DEFAULT_MATHLIB_TEST_RANDOM: Optional[Path] = None
DEFAULT_FILTER_CONFIG = REPO_ROOT / "metadata/premise_filter_config.json"
DEFAULT_COMBINED_JSONL = REPO_ROOT / "metadata/combined_leanabell_goedel_numina.jsonl"
DEFAULT_DAPS_COLLECTION_TEST_JSONL = PROJECT_ROOT / "datasets/DAPS-Collection/10-examples/DAPS_collection_test_10.jsonl"
DEFAULT_COMBINED_SEED = 42
DEFAULT_COMBINED_VAL_COUNT = 100
DEFAULT_COMBINED_TEST_COUNT = 100

DEFAULT_PHASE2_CKPT = REPO_ROOT / "runs/phase2_subgraph/checkpoint_epoch_80.pt"
DEFAULT_PHASE2_PREMISE_EMB = REPO_ROOT / (
    "runs/phase2_subgraph/embeddings_new/"
    "premise_embeddings_epoch_80_alibaba-gte-modernbert-ft_len1024.pt"
)
DEFAULT_PHASE2_TEXT_ENCODER = (
    REPO_ROOT
    / "../phase1_train_text_encoder/runs/Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc/final"
).resolve()



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


def filter_premise_ids_with_indices(
    premise_ids: List[int],
    graph: MathlibGraph,
    allowed_types: Set[str],
    blacklist: Set[str],
) -> Tuple[List[int], List[int]]:
    filtered_ids: List[int] = []
    filtered_indices: List[int] = []
    for idx, pid in enumerate(premise_ids):
        node = graph.nodes[int(pid)]
        if allowed_types and node.subcategory not in allowed_types:
            continue
        if blacklist and node.name in blacklist:
            continue
        filtered_indices.append(idx)
        filtered_ids.append(int(pid))
    return filtered_ids, filtered_indices


def build_fixed_holdout_splits(
    query_ids: List[int],
    graph: MathlibGraph,
    *,
    seed: int,
    val_count: int = 2000,
    test_count: int = 2000,
    holdout_per_split: int = 1000,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    if holdout_per_split > val_count or holdout_per_split > test_count:
        raise ValueError(
            "holdout_per_split must be <= val_count/test_count "
            f"(holdout={holdout_per_split}, val={val_count}, test={test_count})"
        )

    by_file: Dict[str, List[int]] = {}
    for qid in query_ids:
        spath = graph.nodes[qid].source_path or "<unknown>"
        by_file.setdefault(spath, []).append(qid)

    files = sorted(by_file.keys())
    rng.shuffle(files)

    def _pick_holdout_files(available_files: List[str]) -> List[str]:
        chosen: List[str] = []
        total = 0
        for fpath in available_files:
            chosen.append(fpath)
            total += len(by_file[fpath])
            if total >= holdout_per_split:
                break
        if total < holdout_per_split:
            raise ValueError(
                f"Not enough queries to build holdout split from files: need {holdout_per_split}, have {total}"
            )
        return chosen

    val_holdout_files = _pick_holdout_files(files)
    remaining_files = [f for f in files if f not in set(val_holdout_files)]
    test_holdout_files = _pick_holdout_files(remaining_files)
    holdout_files = set(val_holdout_files) | set(test_holdout_files)

    val_pool: List[int] = []
    for fpath in val_holdout_files:
        val_pool.extend(by_file[fpath])
    test_pool: List[int] = []
    for fpath in test_holdout_files:
        test_pool.extend(by_file[fpath])

    rng.shuffle(val_pool)
    rng.shuffle(test_pool)
    val_hold = val_pool[:holdout_per_split]
    test_hold = test_pool[:holdout_per_split]

    remaining = [
        qid
        for qid in query_ids
        if (graph.nodes[qid].source_path or "<unknown>") not in holdout_files
    ]
    rng.shuffle(remaining)

    val_extra_needed = max(0, val_count - len(val_hold))
    test_extra_needed = max(0, test_count - len(test_hold))
    if len(remaining) < val_extra_needed + test_extra_needed:
        raise ValueError(
            f"Not enough remaining queries for val/test splits: needed {val_extra_needed + test_extra_needed}, have {len(remaining)}"
        )

    val_extra = remaining[:val_extra_needed]
    test_extra = remaining[val_extra_needed : val_extra_needed + test_extra_needed]
    train_rest = remaining[val_extra_needed + test_extra_needed :]

    return {
        "train": train_rest,
        "val": val_hold + val_extra,
        "test": test_hold + test_extra,
        "val_holdout": val_hold,
        "val_random": val_extra,
        "test_holdout": test_hold,
        "test_random": test_extra,
    }


def load_query_ids(path: Path) -> Optional[List[int]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    qids = data.get("query_ids")
    if not isinstance(qids, list):
        return None
    return [int(x) for x in qids]


def load_query_ids_required(path: Path, split_name: str) -> List[int]:
    qids = load_query_ids(path)
    if qids is None:
        raise FileNotFoundError(f"Required split file for {split_name} not found or invalid: {path}")
    if not qids:
        raise ValueError(f"Split file for {split_name} is empty: {path}")
    return [int(x) for x in qids]


def load_split_cache(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
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


def resolve_mathlib_splits(
    *,
    split_cache_json: Path,
    split_all_file: Optional[Path],
    split_holdout_file: Optional[Path],
    split_random_file: Optional[Path],
) -> Tuple[List[int], List[int], List[int]]:
    if split_cache_json.exists():
        split_cache = load_split_cache(split_cache_json)
        test_all = split_cache.get("test", [])
        test_holdout = split_cache.get("test_holdout", [])
        test_random = split_cache.get("test_random", [])
        if test_all and test_holdout and test_random:
            print(
                "[split] loaded from split cache: "
                f"{split_cache_json} (all={len(test_all)} holdout={len(test_holdout)} random={len(test_random)})"
            )
            return [int(x) for x in test_all], [int(x) for x in test_holdout], [int(x) for x in test_random]
        print(f"[split] split cache found but incomplete, fallback to explicit files: {split_cache_json}")

    if split_all_file is None or split_holdout_file is None or split_random_file is None:
        raise ValueError(
            "split-cache-json is required in this package. "
            "If you must use explicit split files, provide all of: "
            "--mathlib-test-all, --mathlib-test-holdout, --mathlib-test-random."
        )

    math_test_all = load_query_ids_required(split_all_file, "mathlib_test_all")
    math_test_holdout = load_query_ids_required(split_holdout_file, "mathlib_test_holdout")
    math_test_random = load_query_ids_required(split_random_file, "mathlib_test_random")
    print(
        "[split] loaded explicit split files: "
        f"all={len(math_test_all)} holdout={len(math_test_holdout)} random={len(math_test_random)}"
    )
    return math_test_all, math_test_holdout, math_test_random


def strip_prefixes_for_phase2(prompt: str) -> str:
    if not prompt:
        return prompt
    text = prompt.lstrip()
    long_prefix = (
        "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n"
        "```lean4\n"
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 0\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )
    mid_prefix = (
        "import Mathlib\n"
        "import Aesop\n\n"
        "set_option maxHeartbeats 0\n\n"
        "open BigOperators Real Nat Topology Rat\n\n"
    )
    short_prefix = "import Mathlib\n\n"
    for prefix in (long_prefix, mid_prefix, short_prefix):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("```lean4\n", "")
    text = text.replace("```", "")
    return text.lstrip()


def clean_prompt_for_selection(prompt: str) -> str:
    text = prompt or ""
    if "```lean4" in text:
        text = text.split("```lean4", 1)[1]
    lines = text.splitlines()
    cleaned: List[str] = []
    skipping = True
    for line in lines:
        stripped = line.strip()
        if skipping:
            if not stripped:
                continue
            if stripped.startswith("import "):
                continue
            skipping = False
        cleaned.append(line)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


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


def encode_phase2_queries(
    queries: List[str],
    *,
    model,
    text_encoder,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(queries), batch_size), desc="encode_phase2_queries"):
            batch = queries[start:start + batch_size]
            stmt_emb = text_encoder(batch, max_length=max_length).to(device)
            emb = model.encode_queries(stmt_emb)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))


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


def collect_mathlib_positives(
    query_ids: List[int],
    graph: MathlibGraph,
    candidate_id_set: Set[int],
) -> List[List[int]]:
    positives_by_query: List[List[int]] = []
    for qid in query_ids:
        positives = _get_truncated_positives(graph, qid, allowed_ids=candidate_id_set)
        positives_by_query.append([int(pid) for pid in positives])
    return positives_by_query


def build_name_to_ids(graph: MathlibGraph) -> Dict[str, List[int]]:
    name_to_ids: Dict[str, List[int]] = defaultdict(list)
    for node in graph.nodes:
        name_to_ids[node.name].append(node.node_id)
    return name_to_ids


def resolve_premise_id(
    premise: Dict[str, Any],
    *,
    graph: MathlibGraph,
    name_to_ids: Dict[str, List[int]],
) -> Optional[int]:
    name = premise.get("name")
    if not name:
        return None
    nid = graph.id_by_name.get(name)
    if nid is not None:
        return nid
    src_path = premise.get("source_path")
    if src_path:
        for cand in name_to_ids.get(name, []):
            node = graph.nodes[cand]
            if node.source_path and node.source_path.endswith(src_path):
                return node.node_id
    return None


def _priority_rank(subcategory: str) -> int:
    try:
        return TYPE_PRIORITY.index(subcategory)
    except ValueError:
        return len(TYPE_PRIORITY)


def _get_truncated_positives(
    graph: MathlibGraph,
    qid: int,
    *,
    allowed_ids: Optional[Set[int]] = None,
) -> List[int]:
    max_pos = graph.cfg.negative_sampling.max_positives
    raw = list(graph.positives.get(qid, set()))
    if allowed_ids is not None:
        raw = [pid for pid in raw if int(pid) in allowed_ids]
    raw = [pid for pid in raw if int(pid) != int(qid)]
    raw.sort(key=lambda nid: (_priority_rank(graph.nodes[nid].subcategory), graph.nodes[nid].name))
    if len(raw) <= max_pos:
        return raw
    return raw[:max_pos]


def _truncate_positive_ids_like_graph(
    graph: MathlibGraph,
    positive_ids: List[int],
    *,
    allowed_ids: Optional[Set[int]] = None,
) -> List[int]:
    max_pos = graph.cfg.negative_sampling.max_positives
    uniq = [int(pid) for pid in set(positive_ids)]
    if allowed_ids is not None:
        uniq = [pid for pid in uniq if pid in allowed_ids]
    uniq.sort(key=lambda nid: (_priority_rank(graph.nodes[nid].subcategory), graph.nodes[nid].name))
    if len(uniq) <= max_pos:
        return uniq
    return uniq[:max_pos]


def build_combined_test(
    combined_path: Path,
    *,
    graph: MathlibGraph,
    candidate_id_set: Set[int],
    candidate_name_set: Set[str],
    allowed_types: Set[str],
    blacklist: Set[str],
    seed: int,
    val_count: int,
    test_count: int,
    filter_positives: bool = True,
) -> List[Dict[str, Any]]:
    name_to_ids = build_name_to_ids(graph)
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(iter_jsonl(combined_path), start=1):
        prompt = str(record.get("prompt", ""))
        cleaned = clean_prompt_for_selection(prompt)
        if not cleaned:
            continue

        premise_list = record.get("premise_list") or []
        positives = 0
        for prem in premise_list:
            if not isinstance(prem, dict):
                continue
            name = prem.get("name")
            if name and name in candidate_name_set:
                positives += 1
        if positives == 0:
            continue

        resolved: List[int] = []
        for prem in premise_list:
            if not isinstance(prem, dict):
                continue
            pid = resolve_premise_id(prem, graph=graph, name_to_ids=name_to_ids)
            if pid is None:
                continue
            node = graph.nodes[int(pid)]
            if allowed_types and node.subcategory not in allowed_types:
                continue
            if filter_positives and blacklist and node.name in blacklist:
                continue
            if int(pid) in candidate_id_set:
                resolved.append(int(pid))

        if not resolved:
            continue
        resolved = _truncate_positive_ids_like_graph(
            graph,
            resolved,
            allowed_ids=candidate_id_set,
        )
        if not resolved:
            continue

        dataset_index = str(record.get("dataset_index", idx))
        examples.append(
            {
                "line_idx": idx,
                "dataset_index": dataset_index,
                "prompt": prompt,
                "query_text": cleaned,
                "positives": list(resolved),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    test = examples[val_count: val_count + test_count]
    return test


def load_combined_test_jsonl(
    combined_test_path: Path,
    *,
    graph: MathlibGraph,
    candidate_id_set: Set[int],
    allowed_types: Set[str],
    blacklist: Set[str],
    filter_positives: bool = True,
) -> List[Dict[str, Any]]:
    """Load a prebuilt DAPS-Collection test JSONL with positive premise *names*."""
    name_to_ids = build_name_to_ids(graph)
    examples: List[Dict[str, Any]] = []
    for record in iter_jsonl(combined_test_path):
        query_text = str(record.get("query_text", ""))
        if not query_text.strip():
            continue
        pos_names = record.get("positives") or record.get("positive_names") or []
        resolved: List[int] = []
        for name in pos_names:
            if not name:
                continue
            pid = graph.id_by_name.get(name)
            if pid is None:
                ids = name_to_ids.get(name, [])
                if ids:
                    pid = ids[0]
            if pid is None:
                continue
            node = graph.nodes[int(pid)]
            if allowed_types and node.subcategory not in allowed_types:
                continue
            if filter_positives and blacklist and node.name in blacklist:
                continue
            if int(pid) in candidate_id_set:
                resolved.append(int(pid))
        if not resolved:
            continue
        resolved = _truncate_positive_ids_like_graph(graph, resolved, allowed_ids=candidate_id_set)
        if not resolved:
            continue
        examples.append(
            {
                "query_key": record.get("query_key"),
                "query_text": query_text,
                "positives": list(resolved),
            }
        )
    return examples



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phase2 selector on aligned Mathlib/DAPS-Collection splits")
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--filter-config", type=Path, default=DEFAULT_FILTER_CONFIG)
    parser.add_argument("--combined-jsonl", type=Path, default=DEFAULT_COMBINED_JSONL)
    parser.add_argument(
        "--daps-collection-test-jsonl",
        "--combined-test-jsonl",
        dest="daps_collection_test_jsonl",
        type=Path,
        default=DEFAULT_DAPS_COLLECTION_TEST_JSONL,
        help="Optional prebuilt DAPS-Collection test JSONL (query_text + positives names).",
    )
    parser.add_argument("--mathlib-test-all", type=Path, default=DEFAULT_MATHLIB_TEST_ALL)
    parser.add_argument("--mathlib-test-holdout", type=Path, default=DEFAULT_MATHLIB_TEST_HOLDOUT)
    parser.add_argument("--mathlib-test-random", type=Path, default=DEFAULT_MATHLIB_TEST_RANDOM)
    parser.add_argument(
        "--split-cache-json",
        type=Path,
        default=DEFAULT_SPLIT_CACHE_JSON,
        help="Preferred split cache JSON. If valid, overrides --mathlib-test-* files.",
    )
    parser.add_argument("--combined-seed", type=int, default=DEFAULT_COMBINED_SEED)
    parser.add_argument("--combined-val-count", type=int, default=DEFAULT_COMBINED_VAL_COUNT)
    parser.add_argument("--combined-test-count", type=int, default=DEFAULT_COMBINED_TEST_COUNT)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--metrics-csv", type=Path, default=SCRIPT_DIR / "outputs" / "selector_recall_metrics.csv")

    parser.add_argument("--phase2-checkpoint", type=Path, default=DEFAULT_PHASE2_CKPT)
    parser.add_argument("--phase2-embeddings", type=Path, default=DEFAULT_PHASE2_PREMISE_EMB)
    parser.add_argument("--phase2-text-encoder", type=Path, default=DEFAULT_PHASE2_TEXT_ENCODER)

    args = parser.parse_args()

    selector_name = "DAPS"
    ks = (1, 2, 4, 8, 16, 32, 64, 128)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[env] device={device}")
    if args.split_cache_json is not None:
        args.split_cache_json, _ = load_or_create_mathlib_split(args.graph_path, args.split_cache_json)
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

    full_premise_ids = [node.node_id for node in graph.nodes if node.subcategory in graph.cfg.premise_types]
    filtered_premise_ids, _ = filter_premise_ids_with_indices(full_premise_ids, graph, allowed_types, blacklist)
    candidate_ids = list(filtered_premise_ids)
    candidate_id_set = set(candidate_ids)
    name_to_id = {graph.nodes[pid].name: pid for pid in candidate_ids}
    print(f"[candidate] {len(full_premise_ids)} -> {len(candidate_ids)}")

    # Use filtered positives/candidates only.
    filtered_id_set = candidate_id_set
    print(f"[positives] filtered={len(filtered_id_set)}")

    math_test_all, math_test_holdout, math_test_random = resolve_mathlib_splits(
        split_cache_json=args.split_cache_json,
        split_all_file=args.mathlib_test_all,
        split_holdout_file=args.mathlib_test_holdout,
        split_random_file=args.mathlib_test_random,
    )
    all_set = set(math_test_all)
    if any(qid not in all_set for qid in math_test_holdout):
        raise ValueError("mathlib_test_holdout must be a subset of mathlib_test_all")
    if any(qid not in all_set for qid in math_test_random):
        raise ValueError("mathlib_test_random must be a subset of mathlib_test_all")
    print(f"[mathlib] test={len(math_test_all)} holdout={len(math_test_holdout)} random={len(math_test_random)}")

    id_to_idx = {qid: i for i, qid in enumerate(math_test_all)}
    holdout_rows = [id_to_idx[qid] for qid in math_test_holdout if qid in id_to_idx]
    random_rows = [id_to_idx[qid] for qid in math_test_random if qid in id_to_idx]

    positives_data: Dict[str, Any] = {}

    combined_test_jsonl_path = None
    if args.daps_collection_test_jsonl and args.daps_collection_test_jsonl.exists():
        combined_test_jsonl_path = args.daps_collection_test_jsonl
        print(f"[DAPS-Collection] using prebuilt test jsonl: {combined_test_jsonl_path}")
    elif args.daps_collection_test_jsonl:
        print(f"[DAPS-Collection] prebuilt test jsonl not found: {args.daps_collection_test_jsonl} (fall back to full JSONL)")

    positives_all = collect_mathlib_positives(math_test_all, graph, filtered_id_set)
    positives_holdout = [positives_all[i] for i in holdout_rows]
    positives_random = [positives_all[i] for i in random_rows]
    filtered_name_set = set(name_to_id.keys())
    if combined_test_jsonl_path is not None:
        combined_test = load_combined_test_jsonl(
            combined_test_jsonl_path,
            graph=graph,
            candidate_id_set=filtered_id_set,
            allowed_types=allowed_types,
            blacklist=blacklist,
            filter_positives=True,
        )
    else:
        combined_test = build_combined_test(
            args.combined_jsonl,
            graph=graph,
            candidate_id_set=filtered_id_set,
            candidate_name_set=filtered_name_set,
            allowed_types=allowed_types,
            blacklist=blacklist,
            seed=args.combined_seed,
            val_count=args.combined_val_count,
            test_count=args.combined_test_count,
            filter_positives=True,
        )
    combined_positives = [item["positives"] for item in combined_test]
    positives_data = {
        "holdout": positives_holdout,
        "random": positives_random,
        "combined_test": combined_test,
        "combined_positives": combined_positives,
    }
    print(f"[DAPS-Collection filtered] test={len(combined_test)}")

    rows: List[Dict[str, Any]] = []

    if True:
        if not args.phase2_checkpoint.exists() or not args.phase2_embeddings.exists():
            print("[phase2] checkpoint or embeddings missing, skip")
        else:
            print("[phase2] loading model...")
            ckpt = torch_load(args.phase2_checkpoint, map_location=device)
            premise_data = torch_load(args.phase2_embeddings, map_location=device)
            premise_embs = premise_data["embeddings"].to(device)
            premise_ids = list(premise_data["premise_ids"])

            max_length = (
                premise_data.get("max_length")
                or ckpt.get("args", {}).get("max_length")
                or TextEncoderConfig().max_length
            )
            ckpt_embedding = ckpt.get("embedding_model") or ckpt.get("args", {}).get("embedding", "reprover")
            if args.phase2_text_encoder and ckpt_embedding in EMBEDDING_MODELS:
                EMBEDDING_MODELS[ckpt_embedding].model_path = str(args.phase2_text_encoder)
            text_encoder = LeanTextEncoder(
                TextEncoderConfig(trainable=False, embedding_model=ckpt_embedding, max_length=max_length),
                device=device,
            )
            model = build_model_from_checkpoint(ckpt, graph, device=device)

            selector_ids, selector_embs = build_selector_candidate_pool(candidate_ids, premise_ids, premise_embs)
            print(f"[phase2] candidate embeddings: {len(selector_ids)}")
            if not selector_ids:
                print("[phase2] no candidates after filtering, skip")
            else:
                math_queries = [graph.nodes[qid].statement or "" for qid in math_test_all]
                math_query_embs = encode_phase2_queries(
                    math_queries,
                    model=model,
                    text_encoder=text_encoder,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                )
                math_top_idx = compute_topk(
                    math_query_embs,
                    selector_embs,
                    k=args.topk,
                    batch_size=args.batch_size,
                    device=device,
                )

                row: Dict[str, Any] = {"date": dt.date.today().isoformat(), "selector": selector_name}
                metrics_holdout = compute_metrics(
                    math_top_idx[holdout_rows],
                    positives_data["holdout"],
                    selector_ids,
                    ks=ks,
                )
                metrics_random = compute_metrics(
                    math_top_idx[random_rows],
                    positives_data["random"],
                    selector_ids,
                    ks=ks,
                )
                row.update(flatten_metrics("Mathlib4-H", metrics_holdout))
                row.update(flatten_metrics("Mathlib4-R", metrics_random))

                combined_queries = [item["query_text"] for item in positives_data["combined_test"]]
                combined_query_embs = encode_phase2_queries(
                    combined_queries,
                    model=model,
                    text_encoder=text_encoder,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                )
                combined_top_idx = compute_topk(
                    combined_query_embs,
                    selector_embs,
                    k=args.topk,
                    batch_size=args.batch_size,
                    device=device,
                )
                metrics_combined = compute_metrics(
                    combined_top_idx,
                    positives_data["combined_positives"],
                    selector_ids,
                    ks=ks,
                )
                row.update(flatten_metrics("DAPS_collection", metrics_combined))
                rows.append(row)

    # Write output CSVs
    if rows:
        write_metrics_csv(args.metrics_csv, rows)
        print(f"[metrics] wrote {args.metrics_csv}")


if __name__ == "__main__":
    main()
