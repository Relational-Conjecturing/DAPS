from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from .data import MathlibGraph
from .serialization_utils import load_torch_file
from .subgraph import (
    SubgraphConfig,
    SubgraphSample,
    sample_subgraph_nodes,
)


class SubgraphCache:
    """Precomputes and caches subgraph structures for all nodes."""

    def __init__(
        self,
        graph: MathlibGraph,
        cfg: SubgraphConfig,
        cache_device: torch.device | str = "cpu",
    ):
        self.graph = graph
        self.cfg = cfg
        self.cache_device = torch.device(cache_device)
        # Cache stores (node_ids, hops, types, local_edges) for each node
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.pad_type_idx = graph.type_to_idx[graph.pad_type]

    def ensure_cache(
        self,
        cache_dir: str | Path,
        *,
        force: bool = False,
    ) -> None:
        """Ensure subgraph cache exists on disk and load it."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = self._find_matching_cache(cache_path) or self._cache_file(cache_path)

        if force or not cache_file.exists():
            self._precompute_to_disk(cache_file)

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._load_from_disk(cache_file)
                return
            except Exception as exc:
                last_exc = exc
                wait = random.uniform(10, 20)
                print(
                    f"[SubgraphCache] Load failed for {cache_file} (attempt {attempt}/3): {exc}; "
                    f"retrying after {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)

        # Cache might be partially written or corrupted on shared FS; rebuild once.
        print(f"[SubgraphCache] Load failed after retries; rebuilding {cache_file}...", flush=True)
        backup = cache_file.with_suffix(cache_file.suffix + ".corrupt")
        try:
            cache_file.replace(backup)
            print(f"[SubgraphCache] Renamed corrupt cache to {backup}", flush=True)
        except Exception:
            try:
                cache_file.unlink()
            except Exception:
                pass
        self._precompute_to_disk(cache_file)
        try:
            self._load_from_disk(cache_file)
        except Exception:
            # Surface the last error if rebuild still fails.
            if last_exc is not None:
                raise last_exc
            raise

    def _cache_file(self, cache_dir: Path) -> Path:
        return cache_dir / f"subgraph_cache_{self._config_suffix()}.pt"

    def _config_suffix(self) -> str:
        suffix = f"n{self.cfg.max_nodes}_h{self.cfg.max_hop}_in{int(self.cfg.include_incoming)}_out{int(self.cfg.include_outgoing)}"
        if self.cfg.hop1_budget:
            suffix += f"_h1b{self.cfg.hop1_budget}"
        if self.cfg.hop2_budget:
            suffix += f"_h2b{self.cfg.hop2_budget}"
        if self.cfg.subgraph_mode != "random_2hop":
            suffix += f"_mode{self.cfg.subgraph_mode}"
        if getattr(self.cfg, "dummy_edges", False) or getattr(self.cfg, "dummy_degrees", False) or getattr(self.cfg, "dummy_hops", False):
            suffix += "_dummy"
        return suffix

    def _find_matching_cache(self, cache_dir: Path) -> Path | None:
        """Find an existing cache file matching current config."""
        exact = cache_dir / f"subgraph_cache_{self._config_suffix()}.pt"
        if exact.exists():
            return exact
        # Default-mode caches may omit the mode suffix.
        if self.cfg.subgraph_mode == "random_2hop" and not (self.cfg.hop1_budget or self.cfg.hop2_budget):
            legacy_pattern = f"subgraph_cache_n{self.cfg.max_nodes}_h*_in*_out*.pt"
            matches = sorted(cache_dir.glob(legacy_pattern))
            if matches:
                return matches[-1]
        return None

    def _sample_subgraph_optimized(self, seed_id: int) -> SubgraphSample:
        """Deterministic sampling for cache precomputation."""
        return sample_subgraph_nodes(
            self.graph,
            self.cfg,
            seed_id,
            rng=None,
            deterministic=True,
        )

    def _precompute_to_disk(self, cache_file: Path) -> None:
        """Precompute all subgraph structures and save to disk."""
        print(f"Precomputing subgraph cache to {cache_file}...")
        total_nodes = len(self.graph.nodes)

        # For each node, store: (node_ids, hops, types, local_edges)
        cache_data = {}

        for node_id in tqdm(range(total_nodes), desc="Precomputing subgraphs"):
            sample = self._sample_subgraph_optimized(node_id)

            # Extract node types
            types = []
            for nid in sample.node_ids:
                if nid >= 0:
                    node = self.graph.get_node(nid)
                    types.append(self.graph.type_to_idx.get(node.subcategory, self.pad_type_idx))
                else:
                    types.append(self.pad_type_idx)

            # Build local edge index (within this subgraph)
            local_edges = []
            if not getattr(self.cfg, "dummy_edges", False):
                index_map = {}
                for pos, nid in enumerate(sample.node_ids):
                    if nid >= 0:
                        index_map[nid] = pos

                for src_id, src_pos in index_map.items():
                    for dst_id in self.graph.get_node(src_id).outgoing:
                        dst_pos = index_map.get(dst_id)
                        if dst_pos is not None:
                            local_edges.append([src_pos, dst_pos])

            # Convert to tensors and store
            node_ids_tensor = torch.tensor(sample.node_ids, dtype=torch.long)
            hops_tensor = torch.tensor(sample.hops, dtype=torch.long)
            types_tensor = torch.tensor(types, dtype=torch.long)
            edges_tensor = (
                torch.tensor(local_edges, dtype=torch.long).t().contiguous()
                if local_edges
                else torch.zeros((2, 0), dtype=torch.long)
            )

            cache_data[node_id] = (
                node_ids_tensor,
                hops_tensor,
                types_tensor,
                edges_tensor,
            )

        # Save to disk atomically: write to tmp then rename, so other ranks never
        # observe a partially-written file. Force zipfile serialization so mmap
        # loads are supported.
        tmp_path = cache_file.with_suffix(f"{cache_file.suffix}.tmp.{os.getpid()}")
        torch.save(cache_data, tmp_path, _use_new_zipfile_serialization=True)
        if not tmp_path.exists():
            if cache_file.exists():
                print(
                    f"[SubgraphCache] Cache already present at {cache_file}; skipping tmp rename.",
                    flush=True,
                )
                return
            raise FileNotFoundError(f"Expected tmp cache at {tmp_path}, but it is missing.")
        try:
            tmp_path.replace(cache_file)
        except FileNotFoundError:
            if cache_file.exists():
                print(
                    f"[SubgraphCache] Cache already present at {cache_file}; skipping tmp rename.",
                    flush=True,
                )
                return
            raise
        print(f"Subgraph cache saved to {cache_file}")

    def _load_from_disk(self, cache_file: Path, *, verbose: bool = True) -> None:
        """Load precomputed subgraph cache from disk.

        Use mmap=True when available so multiple worker processes can share the
        underlying storage instead of each loading a full copy into RAM.
        """
        if verbose:
            print(f"Loading subgraph cache from {cache_file}...")
        cache_data = load_torch_file(
            cache_file,
            map_location=self.cache_device,
            allow_mmap=True,
            description=str(cache_file),
        )

        if not isinstance(cache_data, dict):
            raise RuntimeError(f"Invalid cache file format at {cache_file}")

        self.cache = cache_data
        if verbose:
            print(f"Loaded {len(self.cache)} precomputed subgraphs")

    def get(self, node_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cached subgraph structure for a node.

        Returns:
            (node_ids, hops, types, local_edges)
        """
        if node_id not in self.cache:
            raise KeyError(f"Node {node_id} not found in subgraph cache")
        return self.cache[node_id]
