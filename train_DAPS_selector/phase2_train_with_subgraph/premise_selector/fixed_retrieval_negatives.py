from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .serialization_utils import load_torch_file


def _torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> object:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class PremiseEmbeddingIndex:
    premise_ids: torch.Tensor  # [P]
    embeddings: torch.Tensor  # [P, D]
    embedding_model: str
    max_length: int

    @staticmethod
    def load(path: str | Path, *, map_location: str | torch.device = "cpu") -> "PremiseEmbeddingIndex":
        payload = _torch_load(path, map_location=map_location)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict in premise embeddings file: {path}")
        premise_ids = payload.get("premise_ids")
        embeddings = payload.get("embeddings")
        if not isinstance(premise_ids, (list, tuple, torch.Tensor)) or embeddings is None:
            raise ValueError(f"Invalid premise embeddings payload: {path}")
        if not isinstance(premise_ids, torch.Tensor):
            premise_ids = torch.tensor(list(premise_ids), dtype=torch.long)
        else:
            premise_ids = premise_ids.to(torch.long)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"premise embeddings must be a torch.Tensor: {path}")
        embeddings = embeddings.to(torch.float32)
        embedding_model = str(payload.get("embedding_model") or "")
        max_length = int(payload.get("max_length") or 0)
        if embeddings.dim() != 2:
            raise ValueError(f"Expected [P, D] embeddings, got shape={tuple(embeddings.shape)}: {path}")
        if premise_ids.numel() != embeddings.size(0):
            raise ValueError(
                f"premise_ids length mismatch: ids={premise_ids.numel()} embs={embeddings.size(0)}: {path}"
            )
        return PremiseEmbeddingIndex(
            premise_ids=premise_ids,
            embeddings=embeddings,
            embedding_model=embedding_model,
            max_length=max_length,
        )


class FrozenQueryProjector(nn.Module):
    """Query projection head matching PremiseSelectorModel.encode_queries()."""

    def __init__(self, *, input_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.query_encoder(x)
        x = self.query_head(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    @staticmethod
    def from_checkpoint(
        ckpt_path: str | Path,
        *,
        input_dim: int,
        device: torch.device | str,
    ) -> "FrozenQueryProjector":
        ckpt = _torch_load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise TypeError(f"Expected dict checkpoint: {ckpt_path}")
        args = ckpt.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        hidden_dim = int(args.get("hidden_dim", 512) or 512)
        proj_dim = int(args.get("proj_dim", 256) or 256)

        model_state = ckpt.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError(f"Checkpoint missing model_state: {ckpt_path}")

        def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not state:
                return {}
            keys = list(state.keys())
            if all(k.startswith("module.") for k in keys):
                return {k[len("module.") :]: v for k, v in state.items()}
            return dict(state)

        state = _strip_module_prefix(model_state)
        projector = FrozenQueryProjector(input_dim=input_dim, hidden_dim=hidden_dim, proj_dim=proj_dim).to(device)
        # Load only query params; keep strict=False so we don't require premise_encoder weights.
        missing, unexpected = projector.load_state_dict(state, strict=False)
        # Expect lots of unexpected keys (premise encoder); missing should be empty or benign.
        if missing:
            # The query weights must exist; otherwise retrieval is meaningless.
            required_prefixes = ("query_encoder.", "query_head.")
            missing_required = [k for k in missing if k.startswith(required_prefixes)]
            if missing_required:
                raise ValueError(f"Checkpoint missing query weights: {missing_required[:8]} ... ({ckpt_path})")
        if unexpected:
            pass
        projector.eval()
        return projector


def build_topk_nonpositive_candidates(
    query_ids: torch.Tensor,
    query_embs: torch.Tensor,
    premise_index: PremiseEmbeddingIndex,
    positives: Mapping[int, set[int]],
    *,
    top_k: int,
    chunk_size: int,
    device: torch.device,
) -> Dict[int, List[int]]:
    """Brute-force topk retrieval, filtering out positives and returning non-positive premise ids."""
    if top_k <= 0 or query_embs.numel() == 0 or premise_index.embeddings.numel() == 0:
        return {}
    query_ids = query_ids.to(torch.long)
    query_embs = torch.nn.functional.normalize(query_embs.to(torch.float32), dim=1)
    premise_embs = torch.nn.functional.normalize(premise_index.embeddings.to(torch.float32), dim=1)
    premise_ids = premise_index.premise_ids.to(torch.long)

    query_embs = query_embs.to(device)
    premise_embs = premise_embs.to(device)
    premise_ids = premise_ids.to("cpu")

    candidate_map: Dict[int, List[int]] = {}
    num_premises = int(premise_embs.size(0))
    extra = min(num_premises, max(top_k * 2, top_k + 64))

    for start in range(0, query_embs.size(0), chunk_size):
        end = min(start + chunk_size, query_embs.size(0))
        q_chunk = query_embs[start:end]
        sims = q_chunk @ premise_embs.T
        top_idx = torch.topk(sims, k=extra, dim=1).indices.to("cpu")
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
            candidate_map[qid] = picked
    return candidate_map


def serialize_candidate_map(
    candidates: Mapping[int, Sequence[int]],
    path: str | Path,
    *,
    meta: Dict[str, object],
) -> None:
    path = Path(path)
    sorted_ids = sorted(candidates.keys())
    offsets = [0]
    flat: List[int] = []
    for qid in sorted_ids:
        cand_list = list(candidates[qid])
        offsets.append(offsets[-1] + len(cand_list))
        flat.extend(cand_list)
    payload = {
        "meta": meta,
        "query_ids": torch.tensor(sorted_ids, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "candidates": torch.tensor(flat, dtype=torch.long) if flat else torch.empty((0,), dtype=torch.long),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp, _use_new_zipfile_serialization=True)
    tmp.replace(path)


def _hash_for_paths(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def default_cache_path(
    cache_dir: str | Path,
    *,
    tag: str,
    split_name: str,
    ckpt_path: str,
    premise_embeddings_path: str,
    top_k: int,
    chunk_size: int,
) -> Path:
    cache_dir = Path(cache_dir)
    digest = _hash_for_paths(tag, split_name, ckpt_path, premise_embeddings_path, str(top_k), str(chunk_size))
    return cache_dir / f"fixed_retrieval_negs_{tag}_{split_name}_k{top_k}_c{chunk_size}_{digest}.pt"


class CandidateStore:
    """Compact lookup for per-query fixed candidates stored as (query_ids, offsets, flat candidates)."""

    def __init__(self, *, query_ids: Sequence[int], offsets: Sequence[int], candidates: torch.Tensor):
        self.query_ids = list(int(x) for x in query_ids)
        self.offsets = list(int(x) for x in offsets)
        self.candidates = candidates.to(torch.long)
        if len(self.offsets) != len(self.query_ids) + 1:
            raise ValueError("Invalid offsets length.")

    @staticmethod
    def load(path: str | Path) -> Tuple["CandidateStore", Dict[str, object]]:
        payload = load_torch_file(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid candidate store payload: {path}")
        query_ids = payload.get("query_ids")
        offsets = payload.get("offsets")
        candidates = payload.get("candidates")
        meta = payload.get("meta") or {}
        if not isinstance(query_ids, torch.Tensor) or not isinstance(offsets, torch.Tensor) or not isinstance(candidates, torch.Tensor):
            raise TypeError(f"Invalid candidate store tensors: {path}")
        store = CandidateStore(
            query_ids=query_ids.tolist(),
            offsets=offsets.tolist(),
            candidates=candidates,
        )
        return store, dict(meta) if isinstance(meta, dict) else {}

    def get(self, query_id: int) -> torch.Tensor:
        idx = bisect.bisect_left(self.query_ids, int(query_id))
        if idx >= len(self.query_ids) or self.query_ids[idx] != int(query_id):
            return torch.empty((0,), dtype=torch.long)
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        if end <= start:
            return torch.empty((0,), dtype=torch.long)
        return self.candidates[start:end]


def build_or_load_candidate_store(
    *,
    tag: str,
    split_name: str,
    cache_dir: str | Path,
    ckpt_path: str | Path,
    premise_embeddings_path: str | Path,
    query_ids: Sequence[int],
    projected_query_embs: torch.Tensor,
    positives: Mapping[int, set[int]],
    top_k: int,
    chunk_size: int,
    build_device: torch.device,
    force_rebuild: bool,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(ckpt_path))
    premise_embeddings_path = str(Path(premise_embeddings_path))
    out_path = default_cache_path(
        cache_dir,
        tag=tag,
        split_name=split_name,
        ckpt_path=ckpt_path,
        premise_embeddings_path=premise_embeddings_path,
        top_k=top_k,
        chunk_size=chunk_size,
    )
    if out_path.exists() and not force_rebuild:
        return out_path

    premise_index = PremiseEmbeddingIndex.load(premise_embeddings_path, map_location="cpu")
    query_ids_tensor = torch.tensor(list(query_ids), dtype=torch.long)
    candidates = build_topk_nonpositive_candidates(
        query_ids_tensor,
        projected_query_embs,
        premise_index,
        positives,
        top_k=top_k,
        chunk_size=chunk_size,
        device=build_device,
    )
    meta = {
        "tag": tag,
        "split": split_name,
        "checkpoint": ckpt_path,
        "premise_embeddings": premise_embeddings_path,
        "top_k": int(top_k),
        "chunk_size": int(chunk_size),
    }
    serialize_candidate_map(candidates, out_path, meta=meta)
    return out_path
