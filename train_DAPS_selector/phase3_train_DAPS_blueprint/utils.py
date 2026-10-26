from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from blueprint_node import BlueprintNode


def build_premise_text(node: BlueprintNode) -> str:
    parts = [f"id: {node.raw_id}"]
    if node.nlp_text:
        parts.append(f"nlp: {node.nlp_text}")
    if node.lean_decl_name:
        parts.append(f"lean_decl_name: {node.lean_decl_name}")
    if node.lean4_code:
        parts.append(f"lean4:\n{node.lean4_code}")
    return "\n\n".join(parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_subgraph(
    graph: "BlueprintGraph",
    seed_id: int,
    max_nodes: int = 16,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], List[int]]:
    """Sample 2-hop neighbors and return (node_ids, hops) padded to max_nodes.

    The first entry is always the seed node with hop=0.
    """
    if rng is None:
        rng = random

    neighbors: Dict[int, int] = {}

    seed_node = graph.get_node(seed_id)
    hop1 = list(seed_node.outgoing)
    for nid in hop1:
        neighbors[nid] = 1

    for nid in hop1:
        node = graph.get_node(nid)
        for nid2 in node.outgoing:
            if nid2 == seed_id:
                continue
            if nid2 not in neighbors:
                neighbors[nid2] = 2

    neighbors.pop(seed_id, None)

    neighbor_ids = list(neighbors.keys())
    if len(neighbor_ids) > max_nodes - 1:
        neighbor_ids = rng.sample(neighbor_ids, max_nodes - 1)

    rng.shuffle(neighbor_ids)

    node_ids = [seed_id] + neighbor_ids
    hops = [0] + [neighbors[nid] for nid in neighbor_ids]

    while len(node_ids) < max_nodes:
        node_ids.append(-1)
        hops.append(0)

    return node_ids, hops


class BlueprintFeatureCache:
    def __init__(
        self,
        graph: "BlueprintGraph",
        text_encoder,
        *,
        cache_device: torch.device | str = "cpu",
        chunk_size: int = 16,
        max_length: int = 1024,
    ):
        self.graph = graph
        self.text_encoder = text_encoder
        self.cache_device = torch.device(cache_device)
        self.chunk_size = max(1, int(chunk_size))
        self.max_length = max(1, int(max_length))
        self.cache: Dict[str, Dict[int, torch.Tensor]] = {"premise": {}, "query": {}}
        self.trainable = bool(getattr(text_encoder, "cfg", None) and text_encoder.cfg.trainable)

    @property
    def embedding_dim(self) -> int:
        return self.text_encoder.embedding_dim

    def _texts(self, node_ids: Sequence[int], mode: str) -> List[str]:
        if mode == "query":
            return [self.graph.get_node(nid).statement_text() for nid in node_ids]
        return [build_premise_text(self.graph.get_node(nid)) for nid in node_ids]

    def get(
        self,
        node_ids: Sequence[int],
        *,
        mode: str = "premise",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if not node_ids:
            dim = self.embedding_dim
            return torch.empty((0, dim), dtype=torch.float32, device=device or self.cache_device)

        cache = self.cache[mode]
        missing = [nid for nid in node_ids if nid not in cache]
        if missing:
            texts = self._texts(missing, mode)
            enc_chunks: List[torch.Tensor] = []
            for start in range(0, len(texts), self.chunk_size):
                chunk_texts = texts[start : start + self.chunk_size]
                if not chunk_texts:
                    continue
                if self.trainable:
                    chunk_emb = self.text_encoder(chunk_texts, max_length=self.max_length)
                else:
                    with torch.no_grad():
                        chunk_emb = self.text_encoder(chunk_texts, max_length=self.max_length)
                enc_chunks.append(chunk_emb.detach().cpu())
            if enc_chunks:
                encodings = torch.cat(enc_chunks, dim=0)
                for nid, emb in zip(missing, encodings):
                    cache[nid] = emb.to(self.cache_device)

        stacked = torch.stack([cache[nid] for nid in node_ids], dim=0)
        if device is None:
            return stacked
        return stacked.to(device)


def build_premise_inputs(
    node_ids: Sequence[int],
    graph: "BlueprintGraph",
    feature_cache: BlueprintFeatureCache,
    *,
    device: torch.device | str,
    max_nodes: int = 16,
    rng: Optional[random.Random] = None,
    seed_base: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    samples: List[Tuple[List[int], List[int]]] = []
    for nid in node_ids:
        if seed_base is not None:
            local_rng = random.Random(seed_base + int(nid))
            sample_ids, hops = sample_subgraph(graph, nid, max_nodes=max_nodes, rng=local_rng)
        else:
            sample_ids, hops = sample_subgraph(graph, nid, max_nodes=max_nodes, rng=rng)
        samples.append((sample_ids, hops))

    seed_ids = [sample_ids[0] for sample_ids, _ in samples if sample_ids and sample_ids[0] >= 0]
    other_ids: List[int] = []
    for sample_ids, _ in samples:
        for nid in sample_ids[1:]:
            if nid >= 0:
                other_ids.append(nid)

    seed_feats = feature_cache.get(seed_ids, mode="premise", device=device)
    other_feats = feature_cache.get(other_ids, mode="premise", device=device)

    pad_type_idx = graph.type_to_idx[graph.pad_type]
    pad_feat = torch.zeros(feature_cache.embedding_dim, dtype=torch.float32, device=device)

    neighbor_types = []
    neighbor_hops = []
    neighbor_features = []
    in_degrees = []
    out_degrees = []
    batch_edges: List[torch.Tensor] = []
    batch_vec = []
    node_offset = 0

    seed_ptr = 0
    other_ptr = 0

    for batch_idx, (sample_ids, hops) in enumerate(samples):
        local_types = []
        local_hops = []
        local_feats = []

        index_map: Dict[int, int] = {}
        for pos, nid in enumerate(sample_ids):
            if nid >= 0:
                index_map[nid] = pos

        for pos, (nid, hop) in enumerate(zip(sample_ids, hops)):
            if nid >= 0:
                node = graph.get_node(nid)
                local_types.append(graph.type_to_idx.get(node.subcategory, pad_type_idx))
                local_hops.append(hop)
                if pos == 0:
                    local_feats.append(seed_feats[seed_ptr])
                    seed_ptr += 1
                else:
                    local_feats.append(other_feats[other_ptr])
                    other_ptr += 1
            else:
                local_types.append(pad_type_idx)
                local_hops.append(0)
                local_feats.append(pad_feat)

        neighbor_types.append(torch.tensor(local_types, dtype=torch.long, device=device))
        neighbor_hops.append(torch.tensor(local_hops, dtype=torch.long, device=device))
        neighbor_features.append(torch.stack(local_feats, dim=0))
        in_degrees.append(
            torch.tensor(
                [len(graph.get_node(nid).incoming) if nid >= 0 else 0 for nid in sample_ids],
                dtype=torch.long,
                device=device,
            )
        )
        out_degrees.append(
            torch.tensor(
                [len(graph.get_node(nid).outgoing) if nid >= 0 else 0 for nid in sample_ids],
                dtype=torch.long,
                device=device,
            )
        )

        edges = []
        for src_id, src_pos in index_map.items():
            for dst_id in graph.get_node(src_id).outgoing:
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
        node_offset += max_nodes
        batch_vec.append(torch.full((max_nodes,), batch_idx, dtype=torch.long, device=device))

    neighbor_types_t = torch.stack(neighbor_types, dim=0)
    neighbor_hops_t = torch.stack(neighbor_hops, dim=0)
    neighbor_features_t = torch.stack(neighbor_features, dim=0)
    in_degrees_t = torch.stack(in_degrees, dim=0)
    out_degrees_t = torch.stack(out_degrees, dim=0)

    edge_index = (
        torch.cat(batch_edges, dim=1)
        if batch_edges
        else torch.zeros((2, 0), dtype=torch.long, device=device)
    )
    batch_tensor = (
        torch.cat(batch_vec, dim=0)
        if batch_vec
        else torch.zeros((0,), dtype=torch.long, device=device)
    )

    node_indices = torch.tensor(node_ids, dtype=torch.long, device=device)

    return {
        "neighbor_types": neighbor_types_t,
        "neighbor_hops": neighbor_hops_t,
        "neighbor_features": neighbor_features_t,
        "in_degrees": in_degrees_t,
        "out_degrees": out_degrees_t,
        "edge_index": edge_index,
        "batch": batch_tensor,
        "node_indices": node_indices,
    }
