from __future__ import annotations

import hashlib
import json
import os
import tempfile
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ijson

from .constants import DEFAULT_POSITIVE_EDGE_TYPES, DEFAULT_QUERY_TYPES


DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_CACHE_JSON = (
    Path(__file__).resolve().parents[1]
    / "metadata/splits/splits_6687f6e53f5af7b1.json"
)


@dataclass(frozen=True)
class FixedHoldoutSplitConfig:
    seed: int = 42
    val_count: int = 2000
    test_count: int = 2000
    holdout_per_split: int = 1000


def load_split_cache(path: Path) -> Dict[str, List[int]]:
    """Load split cache JSON with schema {"splits": {...}}."""
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


def build_fixed_holdout_splits(
    query_ids: Sequence[int],
    *,
    source_paths: Sequence[Optional[str]],
    cfg: FixedHoldoutSplitConfig,
) -> Dict[str, List[int]]:
    """File-level holdout split for train/val/test.

    1) Pick val_holdout_files first until >= holdout_per_split queries.
    2) Pick test_holdout_files from remaining files until >= holdout_per_split queries.
    3) Train excludes all queries from holdout files.
    """
    rng = random.Random(cfg.seed)
    by_file: Dict[str, List[int]] = defaultdict(list)
    for qid in query_ids:
        sp = source_paths[qid] if 0 <= qid < len(source_paths) else None
        sp = sp or "<unknown>"
        by_file[sp].append(int(qid))

    files = sorted(by_file.keys())
    rng.shuffle(files)

    def _pick_holdout_files(available_files: List[str]) -> List[str]:
        chosen: List[str] = []
        total = 0
        for fpath in available_files:
            chosen.append(fpath)
            total += len(by_file[fpath])
            if total >= cfg.holdout_per_split:
                break
        if total < cfg.holdout_per_split:
            raise ValueError(
                f"Not enough queries to build holdout split from files: need {cfg.holdout_per_split}, have {total}"
            )
        return chosen

    val_holdout_files = _pick_holdout_files(files)
    remaining_files = [f for f in files if f not in set(val_holdout_files)]
    test_holdout_files = _pick_holdout_files(remaining_files)

    holdout_files = set(val_holdout_files) | set(test_holdout_files)

    val_pool: List[int] = []
    for f in val_holdout_files:
        val_pool.extend(by_file[f])
    test_pool: List[int] = []
    for f in test_holdout_files:
        test_pool.extend(by_file[f])
    rng.shuffle(val_pool)
    rng.shuffle(test_pool)

    val_holdout = val_pool[: cfg.holdout_per_split]
    test_holdout = test_pool[: cfg.holdout_per_split]

    remaining = [
        int(qid)
        for qid in query_ids
        if (source_paths[qid] or "<unknown>") not in holdout_files
    ]
    rng.shuffle(remaining)

    val_extra_needed = max(0, cfg.val_count - len(val_holdout))
    test_extra_needed = max(0, cfg.test_count - len(test_holdout))
    if len(remaining) < val_extra_needed + test_extra_needed:
        raise ValueError(
            f"Not enough remaining queries for val/test splits: "
            f"needed {val_extra_needed + test_extra_needed}, have {len(remaining)}"
        )

    val_random = remaining[:val_extra_needed]
    test_random = remaining[val_extra_needed : val_extra_needed + test_extra_needed]
    train = remaining[val_extra_needed + test_extra_needed :]

    return {
        "train": train,
        "val": val_holdout + val_random,
        "test": test_holdout + test_random,
        "val_holdout": val_holdout,
        "val_random": val_random,
        "test_holdout": test_holdout,
        "test_random": test_random,
        "val_holdout_files": val_holdout_files,
        "test_holdout_files": test_holdout_files,
    }


def _canonical_query_inputs(graph_path: Path) -> Tuple[List[int], List[Optional[str]]]:
    name_to_id: Dict[str, int] = {}
    source_paths: List[Optional[str]] = []
    eligible: List[bool] = []
    with graph_path.open("rb") as handle:
        for node in ijson.items(handle, "nodes.item", use_float=True):
            if bool(node.get("in_blacklist", False)) or node.get("subcategory") == "file":
                continue
            node_id = len(source_paths)
            name = node.get("name") or ""
            if name:
                name_to_id[name] = node_id
            source_paths.append(node.get("source_path"))
            eligible.append(node.get("subcategory") in DEFAULT_QUERY_TYPES and bool((node.get("statement") or "").strip()))
    positive_queries: set[int] = set()
    with graph_path.open("rb") as handle:
        for edge in ijson.items(handle, "edges.item", use_float=True):
            if edge.get("type") not in DEFAULT_POSITIVE_EDGE_TYPES:
                continue
            source = name_to_id.get(edge.get("source_name"))
            target = name_to_id.get(edge.get("target_name"))
            if source is not None and target is not None:
                positive_queries.add(source)
    return [node_id for node_id, keep in enumerate(eligible) if keep and node_id in positive_queries], source_paths


def load_or_create_mathlib_split(
    graph_path: str | Path,
    preferred_path: str | Path = DEFAULT_SPLIT_CACHE_JSON,
    *,
    query_ids: Optional[Sequence[int]] = None,
    source_paths: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[Path, Dict[str, List[int]]]:
    preferred = Path(preferred_path)
    if preferred.is_file():
        splits = load_split_cache(preferred)
        if not all(splits.get(key) for key in ("train", "val", "test")):
            raise ValueError(f"Invalid split JSON: {preferred}")
        return preferred, splits
    graph = Path(graph_path)
    stat = graph.stat()
    metadata = {
        "split_impl_version": 3,
        "graph_name": graph.name,
        "graph_bytes": stat.st_size,
        "graph_mtime_ns": stat.st_mtime_ns,
        "seed": DEFAULT_SPLIT_SEED,
        "split_mode": "fixed_holdout",
        "val_count": 2000,
        "test_count": 2000,
        "holdout_per_split": 1000,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = preferred.parent / f"splits_seed{DEFAULT_SPLIT_SEED}_{digest}.json"
    if not path.exists():
        if query_ids is None or source_paths is None:
            query_ids, source_paths = _canonical_query_inputs(graph)
        built = build_fixed_holdout_splits(
            query_ids, source_paths=source_paths, cfg=FixedHoldoutSplitConfig(seed=DEFAULT_SPLIT_SEED),
        )
        split_info = {
            "mode": "fixed_holdout",
            "seed": DEFAULT_SPLIT_SEED,
            "val_count": 2000,
            "test_count": 2000,
            "holdout_per_split": 1000,
            "val_holdout_files": built.pop("val_holdout_files"),
            "test_holdout_files": built.pop("test_holdout_files"),
        }
        payload = {"meta": metadata, "splits": built, "split_info": split_info}
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[split] generated {path} (seed={DEFAULT_SPLIT_SEED})", flush=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("meta") != metadata:
        raise ValueError(f"Split metadata mismatch: {path}")
    splits = load_split_cache(path)
    if not all(splits.get(key) for key in ("train", "val", "test")):
        raise ValueError(f"Invalid generated split JSON: {path}")
    print(f"[split] loaded {path} (seed={DEFAULT_SPLIT_SEED})", flush=True)
    return path, splits
