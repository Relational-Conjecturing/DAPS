from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .data import MathlibGraph
from .serialization_utils import load_torch_file


class EmbeddingLogger:
    """Collects per-batch embeddings and persists them to disk once per epoch."""

    def __init__(self, log_dir: Optional[str | Path], rank: int):
        self.log_dir = Path(log_dir) if log_dir else None
        self.rank = rank
        self.reset()

    def enabled(self) -> bool:
        return self.log_dir is not None

    def reset(self) -> None:
        self.query_ids: List[torch.Tensor] = []
        self.query_embs: List[torch.Tensor] = []
        self.premise_ids: List[torch.Tensor] = []
        self.premise_embs: List[torch.Tensor] = []

    def log_queries(self, ids: Sequence[int], embeddings: torch.Tensor) -> None:
        if not self.enabled() or not ids:
            return
        self.query_ids.append(torch.tensor(ids, dtype=torch.long))
        self.query_embs.append(embeddings.detach().cpu())

    def log_premises(self, ids: Sequence[int], embeddings: torch.Tensor) -> None:
        if not self.enabled() or not ids:
            return
        self.premise_ids.append(torch.tensor(ids, dtype=torch.long))
        self.premise_embs.append(embeddings.detach().cpu())

    def dump_epoch(self, epoch: int) -> None:
        if not self.enabled():
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payloads = (
            ("queries", self.query_ids, self.query_embs),
            ("premises", self.premise_ids, self.premise_embs),
        )
        for split_name, id_chunks, emb_chunks in payloads:
            if not id_chunks:
                continue
            ids = torch.cat(id_chunks, dim=0)
            embs = torch.cat(emb_chunks, dim=0)
            path = self.log_dir / f"{split_name}_rank{self.rank}_epoch{epoch}.pt"
            torch.save({"ids": ids, "embeddings": embs}, path)
        self.reset()


def _aggregate_embeddings(split_name: str, epoch: int, log_dir: Path, world_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    merged: Dict[int, torch.Tensor] = {}
    for rank in range(world_size):
        path = log_dir / f"{split_name}_rank{rank}_epoch{epoch}.pt"
        if not path.exists():
            continue
        payload = load_torch_file(path, map_location="cpu")
        ids = payload.get("ids")
        embs = payload.get("embeddings")
        if ids is None or embs is None:
            continue
        id_list = ids.tolist()
        for idx, node_id in enumerate(id_list):
            merged[int(node_id)] = embs[idx].detach().clone()
        path.unlink(missing_ok=True)
    if not merged:
        return None
    sorted_items = sorted(merged.items())
    node_ids = torch.tensor([item[0] for item in sorted_items], dtype=torch.long)
    embeddings = torch.stack([item[1] for item in sorted_items], dim=0)
    return node_ids, embeddings


def _serialize_memory_bank(
    candidates: Dict[int, List[int]],
    path: Path,
) -> None:
    sorted_ids = sorted(candidates.keys())
    offsets = [0]
    flat: List[int] = []
    for qid in sorted_ids:
        cand_list = candidates[qid]
        offsets.append(offsets[-1] + len(cand_list))
        flat.extend(cand_list)
    data = {
        "query_ids": torch.tensor(sorted_ids, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "candidates": torch.tensor(flat, dtype=torch.long) if flat else torch.empty((0,), dtype=torch.long),
    }
    torch.save(data, path)


def _build_candidate_lists(
    query_ids: torch.Tensor,
    query_embs: torch.Tensor,
    premise_ids: torch.Tensor,
    premise_embs: torch.Tensor,
    positives: Dict[int, set[int]],
    *,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> Dict[int, List[int]]:
    if top_k <= 0 or query_embs.numel() == 0 or premise_embs.numel() == 0:
        return {}
    query_embs = torch.nn.functional.normalize(query_embs, dim=1)
    premise_embs = torch.nn.functional.normalize(premise_embs, dim=1)
    query_embs = query_embs.to(device)
    premise_embs = premise_embs.to(device)
    candidate_map: Dict[int, List[int]] = {}
    num_premises = premise_embs.size(0)
    extra = min(num_premises, max(top_k * 2, top_k + 64))
    for start in range(0, query_embs.size(0), chunk_size):
        end = min(start + chunk_size, query_embs.size(0))
        q_chunk = query_embs[start:end]
        sims = q_chunk @ premise_embs.T
        top_idx = torch.topk(sims, k=extra, dim=1).indices.cpu()
        for row in range(q_chunk.size(0)):
            qid = int(query_ids[start + row])
            excluded = positives.get(qid, set())
            picked: List[int] = []
            seen = set()
            for col in top_idx[row].tolist():
                nid = int(premise_ids[col])
                if nid == qid or nid in excluded or nid in seen:
                    continue
                seen.add(nid)
                picked.append(nid)
                if len(picked) >= top_k:
                    break
            if picked:
                candidate_map[qid] = picked
    return candidate_map


class MemoryBankManager:
    """Coordinates embedding logging and ANN-based hard negative generation."""

    def __init__(
        self,
        graph: MathlibGraph,
        *,
        log_dir: Optional[str | Path],
        world_size: int,
        rank: int,
        top_k: int,
        chunk_size: int,
        build_device: Optional[str] = None,
        distributed: bool = False,
        negatives_per_query: int = 0,
    ):
        self.graph = graph
        self.log_dir = Path(log_dir) if log_dir else None
        self.world_size = world_size
        self.rank = rank
        self.distributed = distributed
        self.top_k = top_k
        self.chunk_size = max(1, chunk_size)
        self.build_device = torch.device(build_device) if build_device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = EmbeddingLogger(self.log_dir, rank) if self.log_dir else None
        self.negatives_per_query = negatives_per_query

    def enabled(self) -> bool:
        return self.logger is not None and self.negatives_per_query > 0

    def log_batch(
        self,
        query_ids: Sequence[int],
        query_embeddings: torch.Tensor,
        candidate_ids: Sequence[int],
        premise_embeddings: torch.Tensor,
    ) -> None:
        if not self.enabled():
            return
        self.logger.log_queries(query_ids, query_embeddings)
        self.logger.log_premises(candidate_ids, premise_embeddings)

    def finalize_epoch(self, epoch: int) -> Optional[Path]:
        if not self.enabled():
            return None
        assert self.logger is not None
        self.logger.dump_epoch(epoch)
        if self.distributed:
            dist.barrier()
        memory_path: Optional[Path] = None
        if self.rank == 0 and self.log_dir is not None:
            aggregated_queries = _aggregate_embeddings("queries", epoch, self.log_dir, self.world_size)
            aggregated_premises = _aggregate_embeddings("premises", epoch, self.log_dir, self.world_size)
            if aggregated_queries and aggregated_premises:
                query_ids, query_embs = aggregated_queries
                premise_ids, premise_embs = aggregated_premises
                candidate_map = _build_candidate_lists(
                    query_ids,
                    query_embs,
                    premise_ids,
                    premise_embs,
                    self.graph.positives,
                    top_k=self.top_k,
                    chunk_size=self.chunk_size,
                    device=self.build_device,
                )
                if candidate_map:
                    memory_path = self.log_dir / f"memory_bank_epoch{epoch}.pt"
                    _serialize_memory_bank(candidate_map, memory_path)
        if self.distributed:
            dist.barrier()
        if memory_path and memory_path.exists():
            self.graph.load_memory_bank(memory_path)
        elif self.distributed and self.rank != 0 and self.log_dir is not None:
            candidate_path = self.log_dir / f"memory_bank_epoch{epoch}.pt"
            if candidate_path.exists():
                self.graph.load_memory_bank(candidate_path)
        return memory_path
