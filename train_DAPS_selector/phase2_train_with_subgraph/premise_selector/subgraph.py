from __future__ import annotations

import random
from collections import deque
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Any

import torch

from .data import MathlibGraph
from .features import NodeFeatureCache

STRUCTURED_NODE_TYPES = ("theorem", "lemma", "definition")

_SUBGRAPH_MODE_ALIASES = {
    "0": "random_2hop",
    "default": "random_2hop",
    "random": "random_2hop",
    "random_2hop": "random_2hop",
    "1": "out_1hop",
    "out": "out_1hop",
    "outgoing": "out_1hop",
    "out_1hop": "out_1hop",
    "2": "in_1hop",
    "in": "in_1hop",
    "incoming": "in_1hop",
    "in_1hop": "in_1hop",
    "3": "in_out_1hop",
    "inout": "in_out_1hop",
    "bidirectional": "in_out_1hop",
    "in_out_1hop": "in_out_1hop",
    "4": "out_1hop_structured",
    "out_structured": "out_1hop_structured",
    "out_1hop_structured": "out_1hop_structured",
    "5": "in_1hop_structured",
    "in_structured": "in_1hop_structured",
    "in_1hop_structured": "in_1hop_structured",
    "6": "in_out_1hop_structured",
    "inout_structured": "in_out_1hop_structured",
    "in_out_1hop_structured": "in_out_1hop_structured",
    "7": "out_1hop_full",
    "out_full": "out_1hop_full",
    "out_1hop_full": "out_1hop_full",
    "8": "in_1hop_full",
    "in_full": "in_1hop_full",
    "in_1hop_full": "in_1hop_full",
    "9": "in_out_1hop_full",
    "inout_full": "in_out_1hop_full",
    "in_out_1hop_full": "in_out_1hop_full",
    "10": "out_1hop_structured_full",
    "out_structured_full": "out_1hop_structured_full",
    "out_1hop_structured_full": "out_1hop_structured_full",
    "11": "in_1hop_structured_full",
    "in_structured_full": "in_1hop_structured_full",
    "in_1hop_structured_full": "in_1hop_structured_full",
    "12": "in_out_1hop_structured_full",
    "inout_structured_full": "in_out_1hop_structured_full",
    "in_out_1hop_structured_full": "in_out_1hop_structured_full",
    "13": "global_random_dummy",
    "global_random": "global_random_dummy",
    "global_random_dummy": "global_random_dummy",
}

_SUBGRAPH_MODE_SETTINGS = {
    "random_2hop": {"strategy": "bfs", "disable_selector": False, "structured_only": False, "incoming": True, "outgoing": True},
    "out_1hop": {"strategy": "neighbors", "disable_selector": False, "structured_only": False, "incoming": False, "outgoing": True},
    "in_1hop": {"strategy": "neighbors", "disable_selector": False, "structured_only": False, "incoming": True, "outgoing": False},
    "in_out_1hop": {"strategy": "neighbors", "disable_selector": False, "structured_only": False, "incoming": True, "outgoing": True},
    "out_1hop_structured": {"strategy": "neighbors", "disable_selector": False, "structured_only": True, "incoming": False, "outgoing": True},
    "in_1hop_structured": {"strategy": "neighbors", "disable_selector": False, "structured_only": True, "incoming": True, "outgoing": False},
    "in_out_1hop_structured": {"strategy": "neighbors", "disable_selector": False, "structured_only": True, "incoming": True, "outgoing": True},
    "out_1hop_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": False, "incoming": False, "outgoing": True},
    "in_1hop_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": False, "incoming": True, "outgoing": False},
    "in_out_1hop_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": False, "incoming": True, "outgoing": True},
    "out_1hop_structured_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": True, "incoming": False, "outgoing": True},
    "in_1hop_structured_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": True, "incoming": True, "outgoing": False},
    "in_out_1hop_structured_full": {"strategy": "neighbors", "disable_selector": True, "structured_only": True, "incoming": True, "outgoing": True},
    "global_random_dummy": {
        "strategy": "global_random",
        "disable_selector": True,
        "structured_only": False,
        "incoming": False,
        "outgoing": False,
        "dummy_edges": True,
        "dummy_degrees": True,
        "dummy_hops": True,
    },
}


def canonicalize_subgraph_mode(mode: str | int) -> str:
    key = str(mode).lower()
    if key not in _SUBGRAPH_MODE_ALIASES:
        valid = ", ".join(sorted(set(_SUBGRAPH_MODE_ALIASES.values())))
        raise ValueError(f"Unknown subgraph mode '{mode}'. Supported modes: {valid}")
    return _SUBGRAPH_MODE_ALIASES[key]


def subgraph_mode_settings(mode: str | int) -> Dict[str, Any]:
    canonical = canonicalize_subgraph_mode(mode)
    return _SUBGRAPH_MODE_SETTINGS[canonical]


def subgraph_mode_disables_selector(mode: str | int) -> bool:
    return subgraph_mode_settings(mode)["disable_selector"]

# Per-process shared cache objects to avoid re-loading the subgraph cache in each
# DataLoader worker. Combined with mmap loading in SubgraphCache, this keeps the
# physical RAM footprint low even when multiple workers are spawned.
_SHARED_SUBGRAPH_CACHE: Dict[Path, Any] = {}


@dataclass
class SubgraphConfig:
    max_nodes: int = 64
    max_hop: int = 2
    include_incoming: bool = True
    include_outgoing: bool = True
    subgraph_mode: str = "random_2hop"
    hop1_budget: int = 0
    hop2_budget: int = 0
    dummy_edges: bool = False
    dummy_degrees: bool = False
    dummy_hops: bool = False

    def __post_init__(self) -> None:
        # Normalize aliases (numeric or string) to a canonical mode string.
        self.subgraph_mode = canonicalize_subgraph_mode(self.subgraph_mode)
        settings = subgraph_mode_settings(self.subgraph_mode)
        if "dummy_edges" in settings:
            self.dummy_edges = bool(settings["dummy_edges"])
        if "dummy_degrees" in settings:
            self.dummy_degrees = bool(settings["dummy_degrees"])
        if "dummy_hops" in settings:
            self.dummy_hops = bool(settings["dummy_hops"])
        if settings["strategy"] == "neighbors":
            # Neighbor-only modes are strictly 1-hop; keep max_hop in sync for encoding.
            self.max_hop = 1
        if self.hop1_budget < 0 or self.hop2_budget < 0:
            raise ValueError("Hop budgets must be >= 0")
        if self.max_hop < 2:
            self.hop2_budget = 0


@dataclass
class SubgraphSample:
    node_ids: List[int]
    hops: List[int]


def _sample_bfs(
    graph: MathlibGraph,
    cfg: SubgraphConfig,
    seed_id: int,
    rng: Optional[random.Random] = None,
    *,
    deterministic: bool = False,
) -> SubgraphSample:
    nodes = [seed_id]
    hops = [0]

    queue = deque([(seed_id, 0)])
    visited = {seed_id}
    hop_budgets = {}
    if cfg.hop1_budget > 0:
        hop_budgets[1] = cfg.hop1_budget
    if cfg.hop2_budget > 0:
        hop_budgets[2] = cfg.hop2_budget
    hop_counts = {1: 0, 2: 0}

    target = cfg.max_nodes
    while queue and len(nodes) < target:
        current, hop = queue.popleft()
        if hop >= cfg.max_hop:
            continue
        neighbors: List[int] = []
        if cfg.include_outgoing:
            neighbors.extend(graph.get_node(current).outgoing)
        if cfg.include_incoming:
            neighbors.extend(graph.get_node(current).incoming)
        if deterministic:
            neighbors = sorted(neighbors)
        elif rng is not None:
            rng.shuffle(neighbors)
        else:  # fallback to module-level randomness if no RNG provided
            random.shuffle(neighbors)
        for nbr in neighbors:
            if nbr in visited:
                continue
            next_hop = hop + 1
            budget = hop_budgets.get(next_hop)
            if budget is not None and hop_counts[next_hop] >= budget:
                continue
            visited.add(nbr)
            nodes.append(nbr)
            hops.append(next_hop)
            queue.append((nbr, next_hop))
            if next_hop in hop_counts:
                hop_counts[next_hop] += 1
            if len(nodes) >= target:
                break

    while len(nodes) < target:
        nodes.append(-1)
        hops.append(0)

    return SubgraphSample(nodes, hops)


def _sample_neighbors_1hop(
    graph: MathlibGraph,
    cfg: SubgraphConfig,
    seed_id: int,
    *,
    include_incoming: bool,
    include_outgoing: bool,
    structured_only: bool,
    deterministic: bool = False,
) -> SubgraphSample:
    nodes = [seed_id]
    hops = [0]
    seen = {seed_id}
    hop1_budget = cfg.hop1_budget if cfg.hop1_budget > 0 else None
    hop1_count = 0

    candidates: List[int] = []
    if include_outgoing:
        candidates.extend(graph.get_node(seed_id).outgoing)
    if include_incoming:
        candidates.extend(graph.get_node(seed_id).incoming)

    if deterministic:
        candidates = sorted(candidates)

    for nbr in candidates:
        if hop1_budget is not None and hop1_count >= hop1_budget:
            break
        if nbr in seen:
            continue
        if structured_only:
            subcategory = graph.get_node(nbr).subcategory or ""
            if subcategory.lower() not in STRUCTURED_NODE_TYPES:
                continue
        seen.add(nbr)
        nodes.append(nbr)
        hops.append(1)
        hop1_count += 1
        if len(nodes) >= cfg.max_nodes:
            break

    while len(nodes) < cfg.max_nodes:
        nodes.append(-1)
        hops.append(0)

    return SubgraphSample(nodes, hops)

def _sample_global_random(
    graph: MathlibGraph,
    cfg: SubgraphConfig,
    seed_id: int,
    rng: Optional[random.Random] = None,
    *,
    deterministic: bool = False,
) -> SubgraphSample:
    nodes = [seed_id]
    hops = [0]
    target = cfg.max_nodes
    remaining = max(0, target - 1)

    if remaining <= 0:
        return SubgraphSample(nodes, hops)

    if deterministic:
        rng_local: random.Random = random.Random(seed_id)
    elif rng is not None:
        rng_local = rng
    else:
        rng_local = random

    # Match default subgraph sampling universe: all graph nodes (non-file nodes).
    pool = list(range(len(graph.nodes)))
    pool_len = len(pool)
    seen = {seed_id}
    sampled: List[int] = []
    if pool_len > 0:
        attempts = 0
        max_attempts = remaining * 10 + 100
        while len(sampled) < remaining and attempts < max_attempts:
            attempts += 1
            idx = rng_local.randrange(pool_len)
            nid = pool[idx]
            if nid in seen:
                continue
            seen.add(nid)
            sampled.append(nid)
        if len(sampled) < remaining:
            for nid in pool:
                if nid in seen:
                    continue
                seen.add(nid)
                sampled.append(nid)
                if len(sampled) >= remaining:
                    break

    nodes.extend(sampled[:remaining])
    while len(nodes) < target:
        nodes.append(-1)
        hops.append(0)

    if len(hops) < len(nodes):
        hops.extend([0] * (len(nodes) - len(hops)))

    return SubgraphSample(nodes, hops)


def sample_subgraph_nodes(
    graph: MathlibGraph,
    cfg: SubgraphConfig,
    seed_id: int,
    rng: Optional[random.Random] = None,
    *,
    deterministic: bool = False,
) -> SubgraphSample:
    """Shared sampling helper used by SubgraphBuilder and SubgraphCache."""
    settings = subgraph_mode_settings(cfg.subgraph_mode)
    strategy = settings["strategy"]

    if strategy == "bfs":
        return _sample_bfs(
            graph,
            cfg,
            seed_id,
            rng=rng,
            deterministic=deterministic,
        )

    if strategy == "neighbors":
        return _sample_neighbors_1hop(
            graph,
            cfg,
            seed_id,
            include_incoming=settings["incoming"],
            include_outgoing=settings["outgoing"],
            structured_only=settings["structured_only"],
            deterministic=deterministic,
        )

    if strategy == "global_random":
        return _sample_global_random(
            graph,
            cfg,
            seed_id,
            rng=rng,
            deterministic=deterministic,
        )

    raise ValueError(f"Unsupported subgraph strategy '{strategy}' for mode {cfg.subgraph_mode}")


class SubgraphBuilder:
    def __init__(
        self,
        graph: MathlibGraph,
        feature_cache: NodeFeatureCache,
        cfg: SubgraphConfig,
        subgraph_cache: Optional[object] = None,  # SubgraphCache instance
        subgraph_cache_path: Optional[str] = None,  # Path to cache for lazy loading
    ):
        self.graph = graph
        self.feature_cache = feature_cache
        self.cfg = cfg
        self.subgraph_cache = subgraph_cache
        self.subgraph_cache_path = subgraph_cache_path
        # Per-process shared cache to avoid N copies when multiple workers are spawned.
        self._worker_cache = None  # type: Any
        self.pad_type_idx = graph.type_to_idx[graph.pad_type]
        self.pad_feature = torch.zeros(feature_cache.feature_dim, dtype=torch.float32)
        self.rng = random.Random(0)

    def _get_cache(self):
        """Get cache, loading it lazily in worker processes if needed."""
        # If we have a cache object already, use it
        if self.subgraph_cache is not None:
            return self.subgraph_cache

        # If we have a cache path and we're in a worker, load it once
        if self.subgraph_cache_path is not None:
            if self._worker_cache is None:
                # Reuse a per-process shared cache object if available.
                cache_path = Path(self.subgraph_cache_path)
                global _SHARED_SUBGRAPH_CACHE
                cached = _SHARED_SUBGRAPH_CACHE.get(cache_path)
                if cached is not None:
                    self._worker_cache = cached
                else:
                    from .subgraph_cache import SubgraphCache
                    cache = SubgraphCache(self.graph, self.cfg, cache_device="cpu")
                    worker_info = torch.utils.data.get_worker_info()
                    # Only one worker on rank 0 prints load logs to avoid noisy output
                    rank_str = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
                    verbose_load = (worker_info is None or worker_info.id == 0) and rank_str == "0"
                    cache._load_from_disk(cache_path, verbose=verbose_load)
                    _SHARED_SUBGRAPH_CACHE[cache_path] = cache
                    self._worker_cache = cache
            return self._worker_cache

        return None

    def build_batch(
        self,
        node_ids: Sequence[int],
        device: torch.device | str,
        seed_text_mode: str = "premise",
    ) -> Dict[str, torch.Tensor]:
        # Use cached subgraphs if available
        cache = self._get_cache()
        if cache is not None:
            return self._build_batch_cached(node_ids, device, seed_text_mode, cache)

        # Fallback to dynamic sampling
        samples = [self._sample_subgraph(node_id) for node_id in node_ids]

        seed_ids: List[int] = [sample.node_ids[0] for sample in samples if sample.node_ids and sample.node_ids[0] >= 0]
        other_ids: List[int] = [
            node_id
            for sample in samples
            for idx, node_id in enumerate(sample.node_ids)
            if idx > 0 and node_id >= 0
        ]
        seed_mode = "statement" if seed_text_mode == "statement" else "premise"
        # full_node is for query/seed nodes that need full text; sub_node is for subgraph neighbors (name only).
        seed_mode = "full_node" if seed_text_mode == "statement" else "sub_node"
        seed_feats = self.feature_cache.get(seed_ids, device=device, mode=seed_mode)
        other_feats = self.feature_cache.get(other_ids, device=device, mode="sub_node")

        seed_ptr = 0
        other_ptr = 0
        pad_feat = self.pad_feature.to(device)
        neighbor_types = []
        neighbor_hops = []
        neighbor_features = []

        batch_edges: List[torch.Tensor] = []
        batch_vec = []
        node_offset = 0
        in_degrees = []
        out_degrees = []

        for batch_idx, sample in enumerate(samples):
            local_types = []
            local_hops = []
            local_feats = []

            index_map: Dict[int, int] = {}
            for pos, node_id in enumerate(sample.node_ids):
                if node_id >= 0:
                    index_map[node_id] = pos

            for pos, (node_id, hop) in enumerate(zip(sample.node_ids, sample.hops)):
                if node_id >= 0:
                    node = self.graph.get_node(node_id)
                    local_types.append(self.graph.type_to_idx.get(node.subcategory, self.pad_type_idx))
                    local_hops.append(0 if self.cfg.dummy_hops else hop)
                    if pos == 0:
                        local_feats.append(seed_feats[seed_ptr])
                        seed_ptr += 1
                    else:
                        local_feats.append(other_feats[other_ptr])
                        other_ptr += 1
                else:
                    local_types.append(self.pad_type_idx)
                    local_hops.append(0)
                    local_feats.append(pad_feat)

            neighbor_types.append(torch.tensor(local_types, dtype=torch.long, device=device))
            neighbor_hops.append(torch.tensor(local_hops, dtype=torch.long, device=device))
            neighbor_features.append(torch.stack(local_feats, dim=0))
            if self.cfg.dummy_degrees:
                in_degrees.append(torch.zeros(self.cfg.max_nodes, dtype=torch.long, device=device))
                out_degrees.append(torch.zeros(self.cfg.max_nodes, dtype=torch.long, device=device))
            else:
                in_degrees.append(torch.tensor([len(self.graph.get_node(nid).incoming) if nid >= 0 else 0 for nid in sample.node_ids], dtype=torch.long, device=device))
                out_degrees.append(torch.tensor([len(self.graph.get_node(nid).outgoing) if nid >= 0 else 0 for nid in sample.node_ids], dtype=torch.long, device=device))

            if not self.cfg.dummy_edges:
                edges = []
                for src_id, src_pos in index_map.items():
                    for dst_id in self.graph.get_node(src_id).outgoing:
                        dst_pos = index_map.get(dst_id)
                        if dst_pos is not None:
                            edges.append(
                                torch.tensor(
                                    [src_pos + node_offset, dst_pos + node_offset],
                                    dtype=torch.long,
                                    device=device,
                                )
                            )
                if edges:
                    batch_edges.append(torch.stack(edges, dim=1))
            node_offset += self.cfg.max_nodes
            batch_vec.append(torch.full((self.cfg.max_nodes,), batch_idx, dtype=torch.long, device=device))

        neighbor_types = torch.stack(neighbor_types, dim=0)
        neighbor_hops = torch.stack(neighbor_hops, dim=0)
        neighbor_features = torch.stack(neighbor_features, dim=0)
        in_degrees = torch.stack(in_degrees, dim=0)
        out_degrees = torch.stack(out_degrees, dim=0)

        edge_index = torch.cat(batch_edges, dim=1) if batch_edges and not self.cfg.dummy_edges else torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_tensor = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long, device=device)

        node_indices = torch.tensor(node_ids, dtype=torch.long, device=device)

        return {
            "neighbor_types": neighbor_types,
            "neighbor_hops": neighbor_hops,
            "neighbor_features": neighbor_features,
            "in_degrees": in_degrees,
            "out_degrees": out_degrees,
            "edge_index": edge_index,
            "batch": batch_tensor,
            "node_indices": node_indices,
        }

    def _build_batch_cached(
        self,
        node_ids: Sequence[int],
        device: torch.device | str,
        seed_text_mode: str = "premise",
        cache: object = None,
    ) -> Dict[str, torch.Tensor]:
        """Build batch using precomputed subgraph cache."""
        # Get all cached subgraph structures
        cached_data = [cache.get(nid) for nid in node_ids]

        # Extract seed and other node IDs for feature lookup
        seed_ids: List[int] = []
        other_ids: List[int] = []

        for subgraph_node_ids, _, _, _ in cached_data:
            nodes_list = subgraph_node_ids.tolist()
            if nodes_list and nodes_list[0] >= 0:
                seed_ids.append(nodes_list[0])
            for nid in nodes_list[1:]:
                if nid >= 0:
                    other_ids.append(nid)

        # Fetch features
        seed_mode = "full_node" if seed_text_mode == "statement" else "sub_node"
        seed_feats = self.feature_cache.get(seed_ids, device=device, mode=seed_mode)
        other_feats = self.feature_cache.get(other_ids, device=device, mode="sub_node")

        # Build batched tensors
        seed_ptr = 0
        other_ptr = 0
        pad_feat = self.pad_feature.to(device)

        neighbor_types = []
        neighbor_hops = []
        neighbor_features = []
        in_degrees = []
        out_degrees = []
        batch_edges: List[torch.Tensor] = []
        batch_vec = []
        node_offset = 0

        for batch_idx, (subgraph_node_ids, subgraph_hops, subgraph_types, local_edges) in enumerate(cached_data):
            # Move cached tensors to device
            subgraph_node_ids = subgraph_node_ids.to(device)
            subgraph_hops = subgraph_hops.to(device)
            subgraph_types = subgraph_types.to(device)
            local_edges = local_edges.to(device)

            expected_len = self.cfg.max_nodes
            if subgraph_node_ids.size(0) != expected_len:
                raise ValueError(
                    f"Cached subgraph size {subgraph_node_ids.size(0)} != configured max_nodes {expected_len}. "
                    "Please regenerate subgraph cache or match --max-nodes to the cache."
                )

            # Build features
            local_feats = []
            for pos, nid in enumerate(subgraph_node_ids.tolist()):
                if nid >= 0:
                    if pos == 0:
                        local_feats.append(seed_feats[seed_ptr])
                        seed_ptr += 1
                    else:
                        local_feats.append(other_feats[other_ptr])
                        other_ptr += 1
                else:
                    local_feats.append(pad_feat)

            neighbor_types.append(subgraph_types)
            if self.cfg.dummy_hops:
                neighbor_hops.append(torch.zeros_like(subgraph_hops))
            else:
                neighbor_hops.append(subgraph_hops)
            neighbor_features.append(torch.stack(local_feats, dim=0))
            if self.cfg.dummy_degrees:
                in_degrees.append(torch.zeros(self.cfg.max_nodes, dtype=torch.long, device=device))
                out_degrees.append(torch.zeros(self.cfg.max_nodes, dtype=torch.long, device=device))
            else:
                in_degrees.append(
                    torch.tensor(
                        [len(self.graph.get_node(int(nid)).incoming) if int(nid) >= 0 else 0 for nid in subgraph_node_ids.tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                )
                out_degrees.append(
                    torch.tensor(
                        [len(self.graph.get_node(int(nid)).outgoing) if int(nid) >= 0 else 0 for nid in subgraph_node_ids.tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                )

            # Offset edges for batching
            if not self.cfg.dummy_edges and local_edges.numel() > 0:
                batch_edges.append(local_edges + node_offset)

            node_offset += self.cfg.max_nodes
            batch_vec.append(torch.full((self.cfg.max_nodes,), batch_idx, dtype=torch.long, device=device))

        neighbor_types = torch.stack(neighbor_types, dim=0)
        neighbor_hops = torch.stack(neighbor_hops, dim=0)
        neighbor_features = torch.stack(neighbor_features, dim=0)
        in_degrees = torch.stack(in_degrees, dim=0)
        out_degrees = torch.stack(out_degrees, dim=0)

        edge_index = torch.cat(batch_edges, dim=1) if batch_edges and not self.cfg.dummy_edges else torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_tensor = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long, device=device)

        node_indices = torch.tensor(node_ids, dtype=torch.long, device=device)

        return {
            "neighbor_types": neighbor_types,
            "neighbor_hops": neighbor_hops,
            "neighbor_features": neighbor_features,
            "in_degrees": in_degrees,
            "out_degrees": out_degrees,
            "edge_index": edge_index,
            "batch": batch_tensor,
            "node_indices": node_indices,
        }

    def _sample_subgraph(self, seed_id: int) -> SubgraphSample:
        return sample_subgraph_nodes(
            self.graph,
            self.cfg,
            seed_id,
            rng=self.rng,
            deterministic=False,
        )
