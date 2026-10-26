from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .data import MathlibGraph, NodeRecord
from .serialization_utils import load_torch_file
from .text_encoder import LeanTextEncoder, TextEncoderConfig


# ...
class NodeFeatureCache:
    """Caches frozen text features for premises (rich) and queries (statements)."""

    DEFAULT_SUB_NODE_MAX_LENGTH = 512
    PREMISE_TEXT_SPEC = "name+signature+lean4_code"

    def __init__(
        self,
        graph: MathlibGraph,
        text_encoder: LeanTextEncoder,
        cache_device: torch.device | str = "cpu",
        chunk_size: int = 16,
        sub_node_max_length: int = DEFAULT_SUB_NODE_MAX_LENGTH,
        full_node_max_length: int | None = None,
        pattern_mode: str = "none",
        pattern_encoder: Optional[LeanTextEncoder] = None,
        pattern_max_length: int = 768,
        pattern_label_map: Optional[Dict[str, int]] = None,
    ):
        self.graph = graph
        self.text_encoder = text_encoder
        self.pattern_encoder = pattern_encoder or text_encoder
        self.pattern_mode = self._normalize_pattern_mode(pattern_mode)
        self.pattern_max_length = max(1, int(pattern_max_length))
        self.cache_device = torch.device(cache_device)
        self.cache: Dict[str, Dict[int, torch.Tensor]] = {}
        self.chunk_size = max(1, chunk_size)
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.sub_node_max_length = sub_node_max_length
        self.full_node_max_length = full_node_max_length
        self._mode_dims: Dict[str, int] = {}
        self.pattern_label_map = pattern_label_map or (
            self._build_pattern_label_map() if self.pattern_mode != "none" else {}
        )
        self._pattern_extra_dim = self._infer_pattern_dim()
        if self.pattern_mode != "none":
            print(
                f"[FeatureCache] proof pattern mode={self.pattern_mode} "
                f"labels={len(self.pattern_label_map)} extra_dim={self._pattern_extra_dim}",
                flush=True,
            )

    def _normalize_pattern_mode(self, mode: str) -> str:
        key = (mode or "none").strip().lower()
        if key in ("", "none", "off", "disable"):
            return "none"
        if key in ("id", "id_only", "pattern_id"):
            return "id_only"
        if key in ("id_text", "pattern_text", "id_expl", "id_and_expl"):
            return "id_text"
        if key in ("full_text", "id_expl_why", "id_text_why", "id_expl_why_smart", "all"):
            return "full_text"
        # Fallback to safe default
        return "none"

    def _infer_pattern_dim(self) -> int:
        if self.pattern_mode == "none":
            return 0
        if self.pattern_mode == "id_only":
            return 1
        return self.pattern_encoder.embedding_dim if self.pattern_encoder else 0

    def _build_pattern_label_map(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for node in self.graph.nodes:
            label = (node.identified_proof_pattern or "").strip()
            if not label:
                continue
            counter[label] += 1
        ordered: List[Tuple[str, int]] = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return {label: idx + 1 for idx, (label, _cnt) in enumerate(ordered)}

    def _pattern_id(self, node: NodeRecord) -> int:
        label = (node.identified_proof_pattern or "").strip()
        if not label:
            return 0
        return self.pattern_label_map.get(label, 0)

    def _canonical_mode(self, mode: str) -> str:
        key = mode.lower()
        if key in ("premise", "sub_node"):
            return "sub_node"
        if key in ("statement", "full_node"):
            return "full_node"
        return key

    def _register_dim(self, mode: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        self._mode_dims[mode] = tensor.size(-1)

    @property
    def feature_dim(self) -> int:
        cached = self._mode_dims.get("sub_node")
        if cached:
            return cached
        return self.text_encoder.embedding_dim + self._pattern_extra_dim

    @property
    def statement_dim(self) -> int:
        return self._mode_dims.get("full_node") or self.text_encoder.embedding_dim

    def _mode_max_length(self, mode: str) -> int:
        if mode == "full_node":
            return self.full_node_max_length or getattr(self.text_encoder.cfg, "max_length", 512)
        return self.sub_node_max_length or getattr(self.text_encoder.cfg, "max_length", 512)

    def _premise_text(self, node: NodeRecord) -> str:
        return node.neighbor_text()

    def _encode_pattern_batch(self, nodes: Sequence[NodeRecord]) -> torch.Tensor:
        if self.pattern_mode == "none":
            return torch.empty((len(nodes), 0), dtype=torch.float32)
        if self.pattern_mode == "id_only":
            ids = [self._pattern_id(node) for node in nodes]
            return torch.tensor(ids, dtype=torch.float32).unsqueeze(-1)

        include_why = self.pattern_mode == "full_text"
        texts: List[str] = []
        indices: List[int] = []
        for idx, node in enumerate(nodes):
            text = node.proof_pattern_text(include_explanation=True, include_why=include_why)
            if text.strip():
                texts.append(text)
                indices.append(idx)
        dim = self._pattern_extra_dim
        if not texts or self.pattern_encoder is None or dim == 0:
            return torch.zeros((len(nodes), dim), dtype=torch.float32)

        outputs = torch.zeros((len(nodes), dim), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(texts), self.chunk_size):
                chunk_texts = texts[start : start + self.chunk_size]
                chunk_indices = indices[start : start + self.chunk_size]
                chunk_emb = (
                    self.pattern_encoder(chunk_texts, max_length=self.pattern_max_length)
                    .detach()
                    .cpu()
                )
                outputs[torch.as_tensor(chunk_indices, dtype=torch.long)] = chunk_emb
        return outputs

    def _encode_batch(self, node_ids: Sequence[int], mode: str) -> torch.Tensor:
        mode = self._canonical_mode(mode)
        nodes = [self.graph.get_node(nid) for nid in node_ids]
        if mode == "full_node":
            texts = [node.statement_text() for node in nodes]
        else:  # "sub_node"
            texts = [self._premise_text(node) for node in nodes]
        max_len = self._mode_max_length(mode)

        text_chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.chunk_size):
                chunk = texts[start : start + self.chunk_size]
                if not chunk:
                    continue
                chunk_emb = self.text_encoder(chunk, max_length=max_len).detach().cpu()
                text_chunks.append(chunk_emb)
        text_embs = (
            torch.cat(text_chunks, dim=0)
            if text_chunks
            else torch.empty((0, self.text_encoder.embedding_dim), dtype=torch.float32)
        )
        if mode == "sub_node" and node_ids:
            pattern_feats = self._encode_pattern_batch(nodes)
            if pattern_feats.numel() > 0:
                text_embs = torch.cat([text_embs, pattern_feats], dim=1)
        self._register_dim(mode, text_embs)
        return text_embs

    def get(
        self,
        node_ids: Sequence[int],
        device: torch.device | str | None = None,
        mode: str = "sub_node",
    ) -> torch.Tensor:
        mode = self._canonical_mode(mode)
        if not node_ids:
            dim = self.statement_dim if mode == "full_node" else self.feature_dim
            return torch.empty((0, dim), dtype=torch.float32, device=device or self.cache_device)

        if mode in self.tensor_cache:
            tensor = self.tensor_cache[mode]
            max_id = max(node_ids)
            if max_id < tensor.size(0):
                idx = torch.as_tensor(node_ids, dtype=torch.long, device=tensor.device)
                selected = tensor.index_select(0, idx)
                if device is None or device == self.cache_device:
                    return selected
                return selected.to(device)
            # Fallback: tensor cache is stale (e.g., external queries appended); drop cache and encode on-the-fly.
            print(f"[NodeFeatureCache] tensor cache for mode '{mode}' is smaller than requested id {max_id}; switching to on-the-fly encoding.")
            self.tensor_cache.pop(mode, None)

        cache = self.cache.setdefault(mode, {})
        missing = [nid for nid in node_ids if nid not in cache]
        if missing:
            encodings = self._encode_batch(missing, mode=mode)
            for nid, emb in zip(missing, encodings):
                cache[nid] = emb.to(self.cache_device)
        stacked = torch.stack([cache[nid] for nid in node_ids], dim=0)
        if device is None:
            return stacked
        return stacked.to(device)

    def get_full_nodes(
        self, node_ids: Sequence[int], device: torch.device | str | None = None
    ) -> torch.Tensor:
        return self.get(node_ids, device=device, mode="full_node")

    # ------------------------------------------------------------------
    # Disk-backed caching helpers
    # ------------------------------------------------------------------
    def ensure_disk_cache(
        self,
        cache_dir: str | Path,
        *,
        batch_size: int = 512,
        modes: Sequence[str] = ("premise", "statement"),
        force: bool = False,
    ) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        pending: List[str] = []
        for mode in modes:
            path = self._cache_file(cache_path, mode)
            if force or not path.exists():
                pending.append(mode)

        rebuild_attempted = False
        retry_delay = 10.0
        while True:
            if pending:
                self._precompute_to_disk(cache_path, pending, batch_size=batch_size)
            try:
                self._load_tensor_cache(cache_path, modes)
                break
            except Exception as exc:
                if not rebuild_attempted:
                    # Wait once to avoid racing with another writer.
                    print(f"[FeatureCache] Load failed ({exc}); retrying after {retry_delay}s...", flush=True)
                    import time
                    time.sleep(retry_delay)
                    rebuild_attempted = True
                    continue
                if force:
                    raise
                print(f"[FeatureCache] Failed to load cache after retry ({exc}); rebuilding...", flush=True)
                for mode in modes:
                    path = self._cache_file(cache_path, mode)
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                pending = list(modes)
                rebuild_attempted = False

    def _cache_file(self, cache_dir: Path, mode: str) -> Path:
        canon = self._canonical_mode(mode)
        model_name = getattr(self.text_encoder.cfg, "embedding_model", "unknown")
        max_len = self._mode_max_length(canon)
        text_spec = self.PREMISE_TEXT_SPEC if canon == "sub_node" else "statement"
        pattern_suffix = ""
        if canon == "sub_node" and self.pattern_mode != "none":
            suffix_parts = [f"pattern-{self.pattern_mode}"]
            if self.pattern_mode == "id_only":
                suffix_parts.append(f"k{len(self.pattern_label_map)}")
            else:
                patt_model = getattr(
                    getattr(self.pattern_encoder, "cfg", None) or self.text_encoder.cfg,
                    "embedding_model",
                    "unknown",
                )
                suffix_parts.append(f"emb{patt_model}")
                suffix_parts.append(f"plen{self.pattern_max_length}")
            pattern_suffix = "_" + "_".join(suffix_parts)
        return cache_dir / f"{canon}_features_{model_name}_len{max_len}_{text_spec}{pattern_suffix}.pt"

    def _precompute_to_disk(
        self,
        cache_dir: Path,
        modes: Sequence[str],
        *,
        batch_size: int,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.graph.nodes)
        for mode in modes:
            canon = self._canonical_mode(mode)
            dim = self.statement_dim if canon == "full_node" else self.feature_dim
            tensor = torch.empty((total, dim), dtype=torch.float32)
            iterator = tqdm(
                range(0, total, batch_size),
                desc=f"precompute:{canon}",
            )
            for start in iterator:
                end = min(start + batch_size, total)
                batch_ids = list(range(start, end))
                tensor[start:end] = self._encode_batch(batch_ids, mode=canon)
            final_path = self._cache_file(cache_dir, canon)
            tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            # Atomic publish: avoid other ranks seeing a partially-written file.
            # Force zipfile serialization so mmap loads are supported.
            torch.save(tensor.cpu(), tmp_path, _use_new_zipfile_serialization=True)
            try:
                tmp_path.replace(final_path)
            except FileNotFoundError:
                # Another writer may have moved/cleaned the tmp file; retry once.
                import time
                time.sleep(10)
                if tmp_path.exists():
                    tmp_path.replace(final_path)
                else:
                    # If final exists, trust it; otherwise surface error.
                    if not final_path.exists():
                        raise

    def _load_tensor_cache(self, cache_dir: Path, modes: Sequence[str]) -> None:
        for mode in modes:
            canon = self._canonical_mode(mode)
            path = self._cache_file(cache_dir, canon)
            if not path.exists():
                continue
            # Memory-map cached tensors for shared access across processes.
            print(f"Loading {canon} features from {path}...")
            tensor = load_torch_file(
                path,
                map_location=self.cache_device,
                allow_mmap=True,
                description=str(path),
            )
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Cached features at {path} are not a tensor.")
            tensor = tensor.to(self.cache_device)
            self.tensor_cache[canon] = tensor
            self._register_dim(canon, tensor)
            print(f"Successfully loaded {canon} features, shape: {tensor.shape}")
