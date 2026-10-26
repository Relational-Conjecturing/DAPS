from __future__ import annotations

import json
import random
import time
import hashlib
from collections import defaultdict
from array import array
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .config import (
    MathlibGraphConfig,
    NegativeSamplingConfig,
    TYPE_PRIORITY,
)
from .serialization_utils import load_torch_file

TRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
from phase1_train_text_encoder.common.splits import DEFAULT_SPLIT_CACHE_JSON as DEFAULT_FIXED_SPLIT_CACHE, load_or_create_mathlib_split

@dataclass
class NodeRecord:
    node_id: int
    name: str
    subcategory: str
    category: str
    module: Optional[str]
    source_path: Optional[str]
    line: Optional[int]
    column: Optional[int]
    isHumanTheorem: bool
    isProp: bool
    signature: Optional[str]
    statement: Optional[str]
    proof: Optional[str]
    lean4_code: Optional[str]
    doc: Optional[str]
    identified_proof_pattern: Optional[str]
    proof_pattern_explanation: Optional[str]
    why_smart: Optional[str]

    outgoing: List[int]
    incoming: List[int]

    def premise_text(self) -> str:
        """Textual view for premise candidates (rich context)."""
        parts: List[str] = []
        parts.append(f"name: {self.name}")
        parts.append(f"type: {self.subcategory}")
        if self.signature:
            parts.append(f"signature: {self.signature}")
        if self.statement:
            parts.append(f"statement: {self.statement}")
        if self.doc:
            parts.append(f"doc: {self.doc}")
        if self.lean4_code:
            parts.append(f"code:\n{self.lean4_code}")
        return "\n\n".join(parts)

    def statement_text(self) -> str:
        """Query text: only the guessed formal statement (no signature/doc/code)."""
        return self.statement or ""

    def neighbor_text(self) -> str:
        """Compact premise text used for subgraph neighbors."""
        parts = [f"name: {self.name}"]
        if self.signature:
            parts.append(f"signature: {self.signature}")
        if self.lean4_code:
            parts.append(f"lean4:\n{self.lean4_code}")
        return "\n\n".join(parts)

    def proof_pattern_text(self, *, include_explanation: bool = False, include_why: bool = False) -> str:
        """Compact text for proof pattern fields (used by optional pattern encoder)."""
        parts: List[str] = []
        if self.identified_proof_pattern:
            parts.append(f"pattern: {self.identified_proof_pattern}")
        if include_explanation and self.proof_pattern_explanation:
            parts.append(f"explanation: {self.proof_pattern_explanation}")
        if include_why and self.why_smart:
            parts.append(f"why_smart: {self.why_smart}")
        return "\n".join(parts)

    def as_text(self) -> str:
        """Return the premise text representation."""
        return self.premise_text()


def _priority_rank(subcategory: str) -> int:
    try:
        return TYPE_PRIORITY.index(subcategory)
    except ValueError:
        return len(TYPE_PRIORITY)


def _load_premise_filter_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        print(f"[premise_filter] Config not found: {path} (filter disabled)", flush=True)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[premise_filter] Failed to read config: {path} ({exc})", flush=True)
        return {}


def _build_premise_filter(config: Dict[str, object]) -> Tuple[set[str], set[str]]:
    allowed_types = set(config.get("allowed_subcategories", []) or [])
    blacklist = set(config.get("class_blacklist", []) or [])
    blacklist |= set(config.get("instance_blacklist", []) or [])
    blacklist |= set(config.get("extra_blacklist", []) or [])
    return allowed_types, blacklist


def _hash_premise_filter_config(config: Dict[str, object]) -> str:
    if not config:
        return ""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class MathlibGraph:
    """In-memory representation of the mathlib graph tailored for premise selection."""

    def __init__(self, cfg: MathlibGraphConfig):
        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] MathlibGraph.__init__ started", flush=True)
        cfg.validate()
        self.cfg = cfg
        self.nodes: List[NodeRecord] = []
        self.id_by_name: Dict[str, int] = {}
        self.module_to_node_ids: Dict[str, List[int]] = defaultdict(list)
        self.file_nodes: Dict[str, Tuple[str, Optional[str]]] = {}
        self.module_imports: Dict[str, set[str]] = defaultdict(set)
        self.positives: Dict[int, set[int]] = defaultdict(set)
        self.memory_bank_candidates: Dict[int, array] = {}
        self._premise_filter_enabled = False
        self._premise_filter_allowed_types: set[str] = set()
        self._premise_filter_blacklist: set[str] = set()
        self._premise_filter_hash = ""
        self._premise_filter_source: Optional[str] = None
        self._premise_filter_stats: Dict[str, int] = {}

        filter_path = Path(cfg.premise_filter_config) if cfg.premise_filter_config else None
        filter_cfg = _load_premise_filter_config(filter_path)
        allowed, blacklist = _build_premise_filter(filter_cfg)
        if allowed or blacklist:
            self._premise_filter_enabled = True
            self._premise_filter_allowed_types = allowed
            self._premise_filter_blacklist = blacklist
            self._premise_filter_hash = _hash_premise_filter_config(filter_cfg)
            self._premise_filter_source = str(filter_path) if filter_path else None
            print(
                f"[premise_filter] Enabled: allowed_types={len(allowed)} blacklist={len(blacklist)} "
                f"hash={self._premise_filter_hash}",
                flush=True,
            )
        elif cfg.premise_filter_config:
            print("[premise_filter] Config loaded but empty; filter disabled.", flush=True)

        print(f"[Rank {rank}] Calling _load_graph...", flush=True)
        self._load_graph(Path(cfg.graph_path))
        if self._premise_filter_enabled:
            stats = self._premise_filter_stats
            print(
                "[premise_filter] Premises kept "
                f"{stats.get('premise_kept', 0)}/{stats.get('premise_total', 0)} "
                f"(type_filtered={stats.get('premise_filtered_type', 0)} "
                f"blacklist_filtered={stats.get('premise_filtered_blacklist', 0)} "
                f"positives_filtered={stats.get('positives_filtered', 0)})",
                flush=True,
            )
        print(f"[Rank {rank}] _load_graph completed, processing types...", flush=True)
        uniq_types = sorted({node.subcategory for node in self.nodes})
        self.pad_type = "__pad__"
        uniq_types.append(self.pad_type)
        self.type_to_idx = {subcategory: idx for idx, subcategory in enumerate(uniq_types)}
        print(f"[Rank {rank}] Calling _build_module_imports...", flush=True)
        self._build_module_imports()
        print(f"[Rank {rank}] Building query_ids list...", flush=True)
        self.query_ids: List[int] = []
        max_pos = 0
        for node in self.nodes:
            if node.subcategory in self.cfg.query_types and node.node_id in self.positives:
                statement = (node.statement_text() or "").strip()
                if not statement:
                    continue
                self.positives[node.node_id].add(node.node_id)
                self.query_ids.append(node.node_id)
                max_pos = max(max_pos, len(self.positives[node.node_id]))
        self.max_positives_per_query = max_pos

        if self._load_external_split_cache():
            print(f"[Rank {rank}] Loaded external fixed split cache", flush=True)
        else:
            raise ValueError("No compatible shared split JSON was loaded")

        print(f"[Rank {rank}] Building global_premise_ids list...", flush=True)
        self.global_premise_ids: List[int] = [
            node.node_id
            for node in self.nodes
            if node.subcategory in self.cfg.premise_types and self._premise_filter_keep(node)
        ]
        print(f"[Rank {rank}] MathlibGraph.__init__ completed", flush=True)

    def add_external_query(self, *, name: str, statement: str, source_tag: Optional[str] = None) -> int:
        """Append a synthetic query node (no edges) for text-only datasets."""
        node_id = len(self.nodes)
        record = NodeRecord(
            node_id=node_id,
            name=name,
            subcategory="external_query",
            category="external_query",
            module=None,
            source_path=source_tag,
            line=None,
            column=None,
            isHumanTheorem=True,
            isProp=True,
            signature=None,
            statement=statement,
            proof=None,
            lean4_code=None,
            doc=None,
            identified_proof_pattern=None,
            proof_pattern_explanation=None,
            why_smart=None,
            outgoing=[],
            incoming=[],
        )
        self.nodes.append(record)
        self.id_by_name[name] = node_id
        # Ensure type_to_idx has this new type
        if record.subcategory not in self.type_to_idx:
            self.type_to_idx[record.subcategory] = len(self.type_to_idx)
        return node_id

    def _premise_filter_decision(self, node: NodeRecord) -> Tuple[bool, Optional[str]]:
        if not self._premise_filter_enabled:
            return True, None
        if self._premise_filter_allowed_types and node.subcategory not in self._premise_filter_allowed_types:
            return False, "type"
        if self._premise_filter_blacklist and node.name in self._premise_filter_blacklist:
            return False, "blacklist"
        return True, None

    def _premise_filter_keep(self, node: NodeRecord) -> bool:
        keep, _ = self._premise_filter_decision(node)
        return keep

    # -------------------------------------------------------------------------
    # Loading & preprocessing
    # -------------------------------------------------------------------------
    def _load_graph(self, path: Path) -> None:
        with path.open("r") as fp:
            raw = json.load(fp)

        filter_stats = {
            "premise_total": 0,
            "premise_kept": 0,
            "premise_filtered_type": 0,
            "premise_filtered_blacklist": 0,
            "positives_filtered": 0,
        }

        next_id = 0
        for node in raw["nodes"]:
            if bool(node.get("in_blacklist", False)):
                continue

            name = node["name"]
            subcategory = str(node.get("subcategory", "") or "")
            category = str(node.get("category", "") or "")

            if subcategory == "file":
                module = node.get("module") or name.replace("file_", "", 1)
                self.file_nodes[name] = (module, node.get("source_path"))
                continue

            record = NodeRecord(
                node_id=next_id,
                name=name,
                subcategory=subcategory,
                category=category,
                module=node.get("module"),
                source_path=node.get("source_path"),
                line=node.get("line"),
                column=node.get("column"),
                isHumanTheorem=bool(node.get("isHumanTheorem", False)),
                isProp=bool(node.get("isProp", False)),
                signature=node.get("signature"),
                statement=node.get("statement"),
                proof=node.get("proof"),
                lean4_code=node.get("lean4_code"),
                doc=node.get("doc"),
                identified_proof_pattern=node.get("identified_proof_pattern"),
                proof_pattern_explanation=node.get("proof_pattern_explanation"),
                why_smart=node.get("why_smart"),
                outgoing=[],
                incoming=[],
            )

            self.nodes.append(record)
            self.id_by_name[name] = next_id
            next_id += 1

            if record.subcategory in self.cfg.premise_types:
                filter_stats["premise_total"] += 1
                keep, reason = self._premise_filter_decision(record)
                if keep:
                    filter_stats["premise_kept"] += 1
                    if record.module:
                        self.module_to_node_ids[record.module].append(record.node_id)
                else:
                    if reason == "type":
                        filter_stats["premise_filtered_type"] += 1
                    elif reason == "blacklist":
                        filter_stats["premise_filtered_blacklist"] += 1

        for edge in raw["edges"]:
            if bool(edge.get("in_blacklist", False)):
                continue

            etype = edge["type"]
            src = self.id_by_name.get(edge["source_name"])
            dst = self.id_by_name.get(edge["target_name"])

            if src is None or dst is None:
                continue

            use_edge_filter = self.cfg.edge_type_filter
            is_positive_edge = (not use_edge_filter) or (etype in self.cfg.positive_edge_types)
            if is_positive_edge:
                self.nodes[src].outgoing.append(dst)
                self.nodes[dst].incoming.append(src)
                if self.nodes[src].subcategory in self.cfg.query_types:
                    if self.nodes[dst].subcategory in self.cfg.premise_types:
                        if self._premise_filter_keep(self.nodes[dst]):
                            self.positives[src].add(dst)
                        else:
                            filter_stats["positives_filtered"] += 1

            # Keep structural neighbors for later even when edge type is not positive.
            else:
                self.nodes[src].outgoing.append(dst)
                self.nodes[dst].incoming.append(src)

            if etype == "IMPORTS":
                src_module = self._module_from_file(edge["source_name"])
                dst_module = self._module_from_file(edge["target_name"])
                if src_module and dst_module:
                    self.module_imports[src_module].add(dst_module)

        self._premise_filter_stats = filter_stats

    def _module_from_file(self, file_node_name: str) -> Optional[str]:
        entry = self.file_nodes.get(file_node_name)
        if entry:
            return entry[0]
        return None

    # ---------------------- Split caching helpers ----------------------


    def _external_split_cache_path(self) -> Optional[Path]:
        env_path = os.environ.get("DAPS_SPLIT_CACHE_PATH", "").strip()
        configured = self.cfg.split_cache_json or env_path
        preferred = Path(configured) if configured else DEFAULT_FIXED_SPLIT_CACHE
        path, _ = load_or_create_mathlib_split(self.cfg.graph_path, preferred)
        return path

    def _load_external_split_cache(self) -> bool:
        path = self._external_split_cache_path()
        if path is None or not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            splits = data.get("splits")
            if not isinstance(splits, dict):
                return False
            query_id_set = set(int(qid) for qid in self.query_ids)
            loaded: Dict[str, List[int]] = {}
            for key, vals in splits.items():
                if isinstance(vals, list):
                    loaded[key] = [int(v) for v in vals if int(v) in query_id_set]
            if not loaded.get("train") or not loaded.get("val") or not loaded.get("test"):
                return False
            self.splits = loaded
            self.split_info = {
                "mode": "external_split_cache",
                "path": str(path),
            }
            print(
                f"[Rank {int(os.environ.get('RANK', '0'))}] Loaded external split cache from {path} "
                f"(train={len(self.splits.get('train', []))} "
                f"val={len(self.splits.get('val', []))} "
                f"test={len(self.splits.get('test', []))})"
            )
            return True
        except Exception as exc:
            raise ValueError(f"Failed to load shared split cache {path}: {exc}") from exc


    def _build_module_imports(self) -> None:
        # Ensure every module we have nodes for appears in the adjacency list.
        all_modules = set(self.module_to_node_ids.keys()) | set(self.module_imports.keys())
        for module in all_modules:
            self.module_imports.setdefault(module, set())

        closure_cache: Dict[str, set[str]] = {}
        visiting: set[str] = set()

        def _closure(module: str) -> set[str]:
            if module in closure_cache:
                return closure_cache[module]
            if module in visiting:
                # Cycle detected; at minimum include the module itself.
                return {module}
            visiting.add(module)
            reachable = {module}
            for dep in self.module_imports.get(module, ()):
                reachable |= _closure(dep)
            visiting.remove(module)
            closure_cache[module] = reachable
            return reachable

        for module in all_modules:
            self.module_imports[module] = _closure(module)


    # -------------------------------------------------------------------------
    # Sampling helpers
    # -------------------------------------------------------------------------
    def get_node(self, node_id: int) -> NodeRecord:
        return self.nodes[node_id]

    def iter_split(self, split: str) -> Iterable[int]:
        return iter(self.splits[split])

    def _module_candidates(self, module: Optional[str]) -> List[int]:
        if not module:
            return []
        candidate_modules = self.module_imports.get(module, {module})
        agg: List[int] = []
        for mod in candidate_modules:
            agg.extend(self.module_to_node_ids.get(mod, []))
        return agg

    def _random_hard_candidates(self, node: NodeRecord) -> List[int]:
        """Import-closure nodes plus earlier nodes in the same file."""
        pool: set[int] = set()
        modules = self.module_imports.get(node.module, {node.module} if node.module else set())
        for mod in modules:
            pool.update(self.module_to_node_ids.get(mod, []))

        if node.source_path and node.line is not None:
            current_line = node.line
            current_col = node.column or 0
            for nid in self.module_to_node_ids.get(node.module, []):
                other = self.nodes[nid]
                if other.source_path != node.source_path or other.node_id == node.node_id:
                    continue
                if other.line is None:
                    continue
                other_col = other.column or 0
                if other.line < current_line or (other.line == current_line and other_col < current_col):
                    pool.add(nid)
        return list(pool)

    def sample_positives(self, node_id: int, max_positives: int) -> List[int]:
        raw = list(self.positives.get(node_id, []))
        if not raw:
            return []
        raw.sort(key=lambda nid: (
            _priority_rank(self.nodes[nid].subcategory),
            self.nodes[nid].name,
        ))
        if len(raw) <= max_positives:
            return raw
        return raw[:max_positives]

    def sample_negatives(
        self,
        node_id: int,
        cfg: NegativeSamplingConfig,
        *,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        rng = rng or random
        node = self.nodes[node_id]
        positives = self.positives.get(node_id, set())

        hard_negs: List[int] = []
        if cfg.include_same_file and node.source_path:
            same_file = [
                nid
                for nid in self.module_to_node_ids.get(node.module, [])
                if nid != node_id
                and nid not in positives
                and self.nodes[nid].source_path == node.source_path
                and (
                    (not cfg.same_file_only_above_line)
                    or (
                        self.nodes[nid].line is not None
                        and node.line is not None
                        and self.nodes[nid].line < node.line
                    )
                )
            ]
            hard_negs = self._draw(same_file, cfg.hard_per_query, rng=rng)

        exclusion = set(positives)
        exclusion.add(node_id)
        exclusion.update(hard_negs)

        local_random: List[int] = []

        # random_hard: import-closure + same-file-above-line pool
        if cfg.random_hard_per_query > 0:
            rh_pool = [
                nid for nid in self._random_hard_candidates(node)
                if nid not in exclusion
            ]
            rh_pick = self._draw(rh_pool, cfg.random_hard_per_query, rng=rng)
            local_random.extend(rh_pick)
            exclusion.update(rh_pick)

        remaining_local = cfg.random_per_query
        if cfg.memory_bank_per_query:
            mb_quota = min(cfg.memory_bank_per_query, remaining_local)
            if mb_quota > 0:
                bank_candidates = self.memory_bank_candidates.get(node_id)
                if bank_candidates:
                    mb_pool = [nid for nid in bank_candidates if nid not in exclusion]
                    if mb_pool:
                        mb_pick = self._draw(mb_pool, min(mb_quota, len(mb_pool)), rng=rng)
                        local_random.extend(mb_pick)
                        exclusion.update(mb_pick)
                        remaining_local -= len(mb_pick)

        remaining_random = max(0, remaining_local)
        pool = [
            nid
            for nid in self._module_candidates(node.module)
            if nid not in exclusion
        ]
        more_local = self._draw(pool, remaining_random, rng=rng)
        local_random.extend(more_local)
        exclusion.update(local_random)
        remaining_local -= len(more_local)

        global_negs: List[int] = []
        if cfg.global_per_query > 0 and self.global_premise_ids:
            global_pool = [
                nid for nid in self.global_premise_ids
                if nid not in exclusion
            ]
            global_negs = self._draw(global_pool, cfg.global_per_query, rng=rng)
            exclusion.update(global_negs)

        return local_random, hard_negs, global_negs

    @staticmethod
    def _draw(candidates: Sequence[int], k: int, rng: Optional[random.Random] = None) -> List[int]:
        rng = rng or random
        if not candidates or k <= 0:
            return []
        if len(candidates) <= k:
            return list(candidates)
        return rng.sample(list(candidates), k)

    def sample_random_hard(self, node_id: int, k: int, rng: Optional[random.Random] = None) -> List[int]:
        """Sample from import-closure + same-file-earlier nodes (excludes positives/self)."""
        node = self.nodes[node_id]
        exclusion = set(self.positives.get(node_id, set()))
        exclusion.add(node_id)
        pool = [nid for nid in self._random_hard_candidates(node) if nid not in exclusion]
        return self._draw(pool, k, rng=rng)

    def _log_sampling_stats(self) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        cfg = self.cfg.negative_sampling
        rng = random.Random(self.cfg.split.seed)
        total_pos = total_rand = total_hard = total_global = 0
        num_queries = len(self.query_ids)
        if num_queries == 0:
            return
        for qid in self.query_ids:
            positives = self.sample_positives(qid, cfg.max_positives)
            rand_negs, hard_negs, global_negs = self.sample_negatives(qid, cfg, rng=rng)
            total_pos += len(positives)
            total_rand += len(rand_negs)
            total_hard += len(hard_negs)
            total_global += len(global_negs)
        avg_pos = total_pos / num_queries
        avg_rand = total_rand / num_queries
        avg_hard = total_hard / num_queries
        avg_global = total_global / num_queries
        print(
            "[MathlibGraph] Avg/query - "
            f"positives: {avg_pos:.2f}, hard negatives: {avg_hard:.2f}, "
            f"random negatives: {avg_rand:.2f}, global negatives: {avg_global:.2f}, "
            f"total negatives: {avg_rand + avg_hard + avg_global:.2f}"
        )

    def load_memory_bank(self, path: str | Path) -> None:
        data = load_torch_file(path, map_location="cpu")
        query_ids = data.get("query_ids")
        offsets = data.get("offsets")
        flat = data.get("candidates")
        if query_ids is None or offsets is None or flat is None:
            raise ValueError(f"Memory bank file {path} missing required tensors.")
        query_list = query_ids.tolist()
        offset_list = offsets.tolist()
        if len(offset_list) != len(query_list) + 1:
            raise ValueError(
                f"Memory bank offsets malformed: expected {len(query_list) + 1}, got {len(offset_list)}"
            )
        flat_list = flat.tolist()
        mapping: Dict[int, array] = {}
        for idx, qid in enumerate(query_list):
            start = int(offset_list[idx])
            end = int(offset_list[idx + 1])
            if end <= start:
                continue
            # Store candidates in compact array('I') form to minimize per-query overhead.
            segment = array("I", (int(flat_list[pos]) for pos in range(start, end)))
            if segment:
                mapping[int(qid)] = segment
        self.memory_bank_candidates = mapping


class PremiseSelectionDataset(Dataset):
    """PyTorch Dataset that returns (query, positives, negatives) tuples."""

    def __init__(
        self,
        graph: MathlibGraph,
        split: str,
        max_queries: Optional[int] = None,
        *,
        fixed_eval: bool = False,
        dynamic_fraction: float = 0.0,
        log_progress: bool = False,
        repeat_factor: int = 1,
    ):
        self.graph = graph
        if split not in graph.splits:
            raise KeyError(f"Unknown split '{split}'. Available: {list(graph.splits)}")
        query_ids = graph.splits[split]
        if max_queries is not None and max_queries > 0:
            query_ids = query_ids[:max_queries]
        repeat_factor = max(1, int(repeat_factor))
        if split == "train" and repeat_factor > 1:
            query_ids = query_ids * repeat_factor
        self.query_ids = query_ids
        self.neg_cfg = graph.cfg.negative_sampling
        self.rng = random.Random(graph.cfg.split.seed + hash(split))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.fixed_eval = fixed_eval
        self.dynamic_fraction = dynamic_fraction if fixed_eval else 0.0
        self.dynamic_query_ids: set[int] = set()
        self.fixed_negatives: Dict[int, Tuple[List[int], List[int], List[int]]] = {}
        if self.fixed_eval and self.query_ids:
            cache_path = self._fixed_negative_cache_path(split, max_queries)
            expected_meta = {
                "split": split,
                "seed": graph.cfg.split.seed,
                "train_ratio": graph.cfg.split.train_ratio,
                "val_ratio": graph.cfg.split.val_ratio,
                "random_per_query": self.neg_cfg.random_per_query,
                "random_hard_per_query": getattr(self.neg_cfg, "random_hard_per_query", 0),
                "hard_per_query": self.neg_cfg.hard_per_query,
                "global_per_query": self.neg_cfg.global_per_query,
                "memory_bank_per_query": self.neg_cfg.memory_bank_per_query,
                "include_same_file": self.neg_cfg.include_same_file,
                "same_file_only_above_line": self.neg_cfg.same_file_only_above_line,
                "dynamic_fraction": self.dynamic_fraction,
                "max_queries": max_queries or 0,
                "total_queries": len(self.query_ids),
                "graph_name": Path(graph.cfg.graph_path).name,
            }
            loaded_from_cache = False
            if cache_path.exists():
                try:
                    data = load_torch_file(cache_path, map_location="cpu")
                    meta = data.get("meta")
                    if meta == expected_meta:
                        dyn_ids = data.get("dynamic_query_ids", [])
                        fixed = data.get("fixed_negatives", {})
                        self.dynamic_query_ids = set(int(x) for x in dyn_ids)
                        for qid, triple in fixed.items():
                            rand_neg, hard_neg, global_neg = triple
                            self.fixed_negatives[int(qid)] = (
                                list(rand_neg),
                                list(hard_neg),
                                list(global_neg),
                            )
                        loaded_from_cache = True
                        if log_progress:
                            print(f"[Rank {self.rank}] Loaded cached negatives for {split} from {cache_path}")
                except Exception as exc:  # pragma: no cover
                    print(f"[Rank {self.rank}] Failed to load cached negatives ({cache_path}): {exc}")

            if not loaded_from_cache:
                if self.world_size > 1 and self.rank != 0:
                    wait_start = time.time()
                    while not cache_path.exists() and (time.time() - wait_start) < 300:
                        time.sleep(2)
                    if cache_path.exists():
                        try:
                            data = load_torch_file(cache_path, map_location="cpu")
                            meta = data.get("meta")
                            if meta == expected_meta:
                                dyn_ids = data.get("dynamic_query_ids", [])
                                fixed = data.get("fixed_negatives", {})
                                self.dynamic_query_ids = set(int(x) for x in dyn_ids)
                                for qid, triple in fixed.items():
                                    rand_neg, hard_neg, global_neg = triple
                                    self.fixed_negatives[int(qid)] = (
                                        list(rand_neg),
                                        list(hard_neg),
                                        list(global_neg),
                                    )
                                loaded_from_cache = True
                                if log_progress:
                                    print(f"[Rank {self.rank}] Loaded cached negatives for {split} after wait")
                        except Exception as exc:  # pragma: no cover
                            print(f"[Rank {self.rank}] Failed to load cached negatives after wait: {exc}")

            if not loaded_from_cache:
                total = len(self.query_ids)
                num_dynamic = 0
                if self.dynamic_fraction > 0:
                    num_dynamic = max(1, int(total * self.dynamic_fraction))
                if num_dynamic > 0:
                    idx_list = list(range(total))
                    dynamic_indices = set(self.rng.sample(idx_list, min(num_dynamic, total)))
                else:
                    dynamic_indices = set()
                iterator = enumerate(self.query_ids)
                if log_progress and self.rank == 0:
                    iterator = tqdm(
                        iterator,
                        total=len(self.query_ids),
                        desc=f"Building {split} fixed negatives",
                    )
                for idx, qid in iterator:
                    if idx in dynamic_indices:
                        self.dynamic_query_ids.add(qid)
                        continue
                    local_seed = graph.cfg.split.seed * 131071 + int(qid)
                    local_rng = random.Random(local_seed)
                    rand_neg, hard_neg, global_neg = self.graph.sample_negatives(
                        qid, self.neg_cfg, rng=local_rng
                    )
                    self.fixed_negatives[qid] = (rand_neg, hard_neg, global_neg)
                if self.rank == 0:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "meta": expected_meta,
                                "dynamic_query_ids": sorted(self.dynamic_query_ids),
                                "fixed_negatives": self.fixed_negatives,
                            },
                            cache_path,
                        )
                        if log_progress:
                            print(f"[Rank {self.rank}] Saved fixed negatives for {split} to {cache_path}")
                    except Exception as exc:  # pragma: no cover
                        print(f"[Rank {self.rank}] Failed to save negatives cache ({cache_path}): {exc}")
            self.dynamic_fraction_used = len(self.dynamic_query_ids) / max(len(self.query_ids), 1)
        else:
            self.dynamic_fraction_used = 0.0

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_id = self.query_ids[idx]
        positives = self.graph.sample_positives(query_id, self.neg_cfg.max_positives)
        if not positives:
            raise RuntimeError(f"Query {query_id} has no positives; dataset should have filtered this out.")

        if self.fixed_eval and query_id not in self.dynamic_query_ids:
            rand_neg, hard_neg, global_neg = self.fixed_negatives.get(query_id, ([], [], []))
        else:
            rand_neg, hard_neg, global_neg = self.graph.sample_negatives(query_id, self.neg_cfg, rng=self.rng)
        return {
            "query_id": torch.tensor(query_id, dtype=torch.long),
            "positive_ids": torch.tensor(positives, dtype=torch.long),
            "random_negative_ids": torch.tensor(rand_neg, dtype=torch.long) if rand_neg else torch.empty(0, dtype=torch.long),
            "hard_negative_ids": torch.tensor(hard_neg, dtype=torch.long) if hard_neg else torch.empty(0, dtype=torch.long),
            "global_negative_ids": torch.tensor(global_neg, dtype=torch.long) if global_neg else torch.empty(0, dtype=torch.long),
            "is_dynamic": torch.tensor(query_id in self.dynamic_query_ids, dtype=torch.bool),
        }

    def _fixed_negative_cache_path(self, split: str, max_queries: Optional[int]) -> Path:
        base = Path(self.graph.cfg.graph_path).parent / "splits"
        limit = f"_max{max_queries}" if max_queries else ""
        fname = (
            f"{split}_negatives_seed{self.graph.cfg.split.seed}"
            f"_r{self.neg_cfg.random_per_query}"
            f"_rh{getattr(self.neg_cfg, 'random_hard_per_query', 0)}"
            f"_h{self.neg_cfg.hard_per_query}"
            f"_g{self.neg_cfg.global_per_query}"
            f"_mb{self.neg_cfg.memory_bank_per_query}"
            f"_dyn{self.dynamic_fraction:.2f}"
            f"{limit}.pt"
        )
        return base / fname


# -----------------------------------------------------------------------------
# External JSONL dataset support (text-only queries, graph-backed premises)
# -----------------------------------------------------------------------------

def _clean_prompt_text(prompt: str) -> str:
    """Remove boilerplate (code fence header/import block) to keep a clean statement."""
    text = prompt
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
            # stop skipping at first non-import/non-empty line
            skipping = False
        cleaned.append(line)
    # drop leading/trailing blank lines
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _resolve_premise_id(premise: Dict, graph: MathlibGraph) -> Optional[int]:
    name = premise.get("name")
    if not name:
        return None
    node_id = graph.id_by_name.get(name)
    if node_id is not None:
        return node_id
    # fallback: try matching by source_path suffix if multiple names
    src_path = premise.get("source_path")
    if src_path:
        matches = [
            nid
            for nid, node in enumerate(graph.nodes)
            if node.name == name and node.source_path and node.source_path.endswith(src_path)
        ]
        if matches:
            return matches[0]
    return None


def _filter_premise_ids_by_type(premise_ids: List[int], graph: MathlibGraph) -> List[int]:
    allowed = graph.cfg.premise_types
    return [pid for pid in premise_ids if graph.nodes[pid].subcategory in allowed]


def load_external_jsonl_into_graph(
    path: str | Path,
    graph: MathlibGraph,
    *,
    seed: int = 42,
    val_count: int = 1000,
    test_count: int = 1000,
    source_tag: str = "external_jsonl",
) -> Dict[str, List[Tuple[int, List[int]]]]:
    rng = random.Random(seed)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    examples: List[Tuple[int, List[int]]] = []
    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            premises = obj.get("premise_list") or []
            resolved: List[int] = []
            for prem in premises:
                pid = _resolve_premise_id(prem, graph)
                if pid is not None:
                    resolved.append(pid)
            resolved = _filter_premise_ids_by_type(resolved, graph)
            if not resolved:
                continue
            prompt = obj.get("prompt", "")
            cleaned = _clean_prompt_text(prompt)
            if not cleaned:
                continue
            q_name = f"external_query_{source_tag}_{obj.get('dataset_index', len(examples))}"
            qid = graph.add_external_query(name=q_name, statement=cleaned, source_tag=source_tag)
            examples.append((qid, resolved))

    rng.shuffle(examples)
    val = examples[:val_count]
    test = examples[val_count : val_count + test_count]
    train = examples[val_count + test_count :]
    return {"train": train, "val": val, "test": test}


def load_external_jsonl(
    path: str | Path,
    graph: MathlibGraph,
    *,
    seed: int = 42,
    val_count: int = 1000,
    test_count: int = 1000,
) -> Dict[str, List[Tuple[int, str, List[int]]]]:
    """Load external JSONL without mutating the graph.

    Returns splits of (uid, query_text, positive_node_ids).
    uid is a stable negative integer derived from dataset_index when available.
    """
    rng = random.Random(seed)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    examples: List[Tuple[int, str, List[int]]] = []
    with path.open("r") as fp:
        for line_idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            premises = obj.get("premise_list") or []
            resolved: List[int] = []
            for prem in premises:
                pid = _resolve_premise_id(prem, graph)
                if pid is not None:
                    resolved.append(pid)
            resolved = _filter_premise_ids_by_type(resolved, graph)
            if not resolved:
                continue

            cleaned = _clean_prompt_text(obj.get("prompt", ""))
            if not cleaned:
                continue

            dataset_index = obj.get("dataset_index")
            try:
                idx_int = int(dataset_index)
            except Exception:
                idx_int = line_idx
            uid = -1 - idx_int
            examples.append((uid, cleaned, resolved))

    rng.shuffle(examples)
    val = examples[:val_count]
    test = examples[val_count : val_count + test_count]
    train = examples[val_count + test_count :]
    return {"train": train, "val": val, "test": test}


class ExternalTextPremiseDataset(Dataset):
    """Dataset for text-only queries with graph-backed premises (no graph node append)."""

    def __init__(self, examples: List[Tuple[int, str, List[int]]], *, query_embeddings: torch.Tensor | None = None):
        self.examples = examples
        self.query_embeddings = query_embeddings  # [N, D] on CPU if provided

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, query_text, positives = self.examples[idx]
        if not positives:
            raise RuntimeError(f"External example {uid} has no positives.")
        item: Dict[str, object] = {
            "query_id": torch.tensor(int(uid), dtype=torch.long),
            "positive_ids": torch.tensor(positives, dtype=torch.long),
            "random_negative_ids": torch.empty(0, dtype=torch.long),
            "hard_negative_ids": torch.empty(0, dtype=torch.long),
            "global_negative_ids": torch.empty(0, dtype=torch.long),
            "is_dynamic": torch.tensor(False, dtype=torch.bool),
            "query_text": query_text,
        }
        if self.query_embeddings is not None:
            item["query_embedding"] = self.query_embeddings[idx]
        # Type ignore: DataLoader collate handles objects/strings.
        return item  # type: ignore[return-value]


class ExternalPremiseDataset(Dataset):
    """Dataset for text-only queries with graph-backed premises."""

    def __init__(self, graph: MathlibGraph, examples: List[Tuple[int, List[int]]]):
        self.graph = graph
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_id, positives = self.examples[idx]
        if not positives:
            raise RuntimeError(f"External query {query_id} has no positives.")
        return {
            "query_id": torch.tensor(query_id, dtype=torch.long),
            "positive_ids": torch.tensor(positives, dtype=torch.long),
            "random_negative_ids": torch.empty(0, dtype=torch.long),
            "hard_negative_ids": torch.empty(0, dtype=torch.long),
            "global_negative_ids": torch.empty(0, dtype=torch.long),
            "is_dynamic": torch.tensor(False, dtype=torch.bool),
        }


class FixedNegativeStoreDataset(Dataset):
    """Wrap an existing dataset, overriding per-item negatives from a CandidateStore.

    This is intentionally opt-in and does not change default training behavior.
    """

    def __init__(self, base: Dataset, *, store, strict: bool = False):
        self.base = base
        self.store = store
        self.strict = strict

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        qid_raw = item.get("query_id")
        qid = int(qid_raw.item() if isinstance(qid_raw, torch.Tensor) else qid_raw)
        negs = self.store.get(qid)
        if self.strict and negs.numel() == 0:
            raise RuntimeError(f"FixedNegativeStoreDataset missing candidates for query_id={qid}")
        # Override all negatives with the fixed retrieval negatives.
        item["random_negative_ids"] = torch.empty(0, dtype=torch.long)
        item["hard_negative_ids"] = torch.empty(0, dtype=torch.long)
        item["global_negative_ids"] = negs.to(torch.long)
        item["is_dynamic"] = torch.tensor(False, dtype=torch.bool)
        return item
