from __future__ import annotations

import argparse
import hashlib
import json
import random
import multiprocessing as mp
import os
import sys
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None

# Support direct script execution:
# python /abs/path/.../premise_selector/train.py
# In that mode __package__ is empty, so relative imports would fail without this bootstrap.
if __package__ in (None, ""):
    _PKG_DIR = Path(__file__).resolve().parent
    _PKG_PARENT = _PKG_DIR.parent
    if str(_PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(_PKG_PARENT))
    __package__ = "premise_selector"

from .config import MathlibGraphConfig
from .data import (
    MathlibGraph,
    PremiseSelectionDataset,
    ExternalTextPremiseDataset,
    FixedNegativeStoreDataset,
    load_external_jsonl,
)
from .features import NodeFeatureCache
from .memory_bank import MemoryBankManager
from .model import PremiseSelectorModel, RelGTPremiseEncoder
from .serialization_utils import load_torch_file
from .subgraph import (
    SubgraphBuilder,
    SubgraphConfig,
    canonicalize_subgraph_mode,
    subgraph_mode_disables_selector,
)
from .subgraph_cache import SubgraphCache
from .text_encoder import LeanTextEncoder, TextEncoderConfig
from .fixed_retrieval_negatives import (
    CandidateStore,
    FrozenQueryProjector,
    build_or_load_candidate_store,
    default_cache_path,
)

_CACHE_FILENAME_PATTERN = re.compile(r"subgraph_cache_n(\d+)(?:_[^.]+)*\.pt")
_METRIC_GROUP = None
_METRIC_GROUP_BACKEND = None
_BARRIER_GROUP = None
_BARRIER_GROUP_BACKEND = None


def _make_timeout(minutes: int | float | None) -> timedelta:
    minutes = minutes if minutes is not None else 10
    try:
        minutes_val = float(minutes)
    except (TypeError, ValueError):
        minutes_val = 10.0
    return timedelta(minutes=minutes_val)


def _get_metric_group(distributed: bool, timeout_minutes: int | float | None = None):
    """Return a process group for metric reductions, preferring CPU/Gloo to avoid NCCL hangs."""
    global _METRIC_GROUP, _METRIC_GROUP_BACKEND
    if not distributed or not dist.is_initialized():
        return None, None
    if _METRIC_GROUP is not None:
        return _METRIC_GROUP, _METRIC_GROUP_BACKEND
    use_gloo = os.environ.get("USE_GLOO_FOR_METRICS", "1") == "1"
    if use_gloo and dist.get_backend() == "nccl":
        try:
            _METRIC_GROUP = dist.new_group(backend="gloo", timeout=_make_timeout(timeout_minutes))
            _METRIC_GROUP_BACKEND = "gloo"
            if dist.get_rank() == 0:
                print("[Metrics] Using Gloo CPU group for metric reductions", flush=True)
            return _METRIC_GROUP, _METRIC_GROUP_BACKEND
        except Exception as exc:  # pragma: no cover - fallback path
            if dist.get_rank() == 0:
                print(f"[Metrics] Failed to create Gloo group, falling back to NCCL metrics: {exc}", flush=True)
    _METRIC_GROUP = dist.group.WORLD
    _METRIC_GROUP_BACKEND = dist.get_backend()
    return _METRIC_GROUP, _METRIC_GROUP_BACKEND


def _get_barrier_group(distributed: bool, timeout_minutes: int | float | None = None):
    """Barrier group that defaults to CPU/Gloo and longer timeout for cache prep phases."""
    global _BARRIER_GROUP, _BARRIER_GROUP_BACKEND
    if not distributed or not dist.is_initialized():
        return None, None
    if _BARRIER_GROUP is not None:
        return _BARRIER_GROUP, _BARRIER_GROUP_BACKEND
    prefer_gloo = os.environ.get("USE_GLOO_FOR_BARRIERS", "1") == "1"
    backend = "gloo" if prefer_gloo and dist.get_backend() == "nccl" else dist.get_backend()
    try:
        _BARRIER_GROUP = dist.new_group(backend=backend, timeout=_make_timeout(timeout_minutes))
        _BARRIER_GROUP_BACKEND = backend
        if dist.get_rank() == 0:
            print(f"[Barrier] Using {backend} group for sync (timeout={timeout_minutes} min)", flush=True)
    except Exception as exc:  # pragma: no cover
        _BARRIER_GROUP = dist.group.WORLD
        _BARRIER_GROUP_BACKEND = dist.get_backend()
        if dist.get_rank() == 0:
            print(f"[Barrier] Failed to create custom group; falling back to default: {exc}", flush=True)
    return _BARRIER_GROUP, _BARRIER_GROUP_BACKEND

def multi_positive_infonce(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    reduction: str = "query",
) -> torch.Tensor:
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    numerator = (exp_logits * positive_mask).sum(dim=1)
    denominator = exp_logits.sum(dim=1).clamp_min(1e-9)
    prob_sum = numerator / denominator

    if reduction == "query":
        prob = prob_sum
    elif reduction == "positive":
        pos_counts = positive_mask.sum(dim=1).clamp_min(1.0)
        prob = prob_sum / pos_counts
    else:
        raise ValueError(f"Unknown loss reduction: {reduction}")

    loss = -torch.log(prob + 1e-9)
    return loss.mean()


def batch_ranking_metrics(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    ks: Sequence[int] = (1, 5, 8, 10, 16, 32),
    ndcg_ks: Sequence[int] = (16, 32),
) -> tuple[Dict[str, float], int]:
    metrics: Dict[str, float] = {f"recall@{k}": 0.0 for k in ks}
    # Add coverage@k metrics.
    metrics.update({f"coverage@{k}": 0.0 for k in ks})
    for k in ndcg_ks:
        metrics[f"ndcg@{k}"] = 0.0
    metrics["mrr"] = 0.0

    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    valid_rows = 0

    discount_cache = {
        k: (1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=logits.device)))
        for k in ndcg_ks
    }

    for row_idx in range(sorted_indices.size(0)):
        pos_mask = positive_mask[row_idx].bool()
        total_pos = int(pos_mask.sum().item())
        if total_pos == 0:
            continue
        valid_rows += 1
        ranked = sorted_indices[row_idx]
        hits = pos_mask[ranked]

        for k in ks:
            topk_hits = hits[:k].float().sum().item()
            # Recall@k: whether at least one positive is found (binary metric).
            metrics[f"recall@{k}"] += 1.0 if topk_hits > 0 else 0.0
            # Coverage@k: fraction of positives captured in top-k.
            metrics[f"coverage@{k}"] += topk_hits / total_pos

        hit_indices = torch.nonzero(hits, as_tuple=False).flatten()
        if hit_indices.numel() > 0:
            rank = hit_indices[0].item()
            metrics["mrr"] += 1.0 / (rank + 1)

        for k in ndcg_ks:
            discounts = discount_cache[k]
            gains = hits[:k].float()
            if gains.numel() == 0:
                continue
            dcg = (gains * discounts[: gains.numel()]).sum().item()
            ideal_len = min(total_pos, k)
            ideal_dcg = discounts[:ideal_len].sum().item()
            if ideal_dcg > 0:
                metrics[f"ndcg@{k}"] += dcg / ideal_dcg

    return metrics, valid_rows


class CrossBatchMemory:
    """Stores premise embeddings from previous batches for extra negatives."""

    def __init__(self, capacity: int, sample_size: int, device: torch.device):
        self.capacity = max(0, capacity)
        self.sample_size = max(0, sample_size)
        self.device = device
        self.storage: deque[torch.Tensor] = deque()
        self.total = 0

    def add(self, embeddings: torch.Tensor) -> None:
        if self.capacity <= 0 or embeddings is None or embeddings.numel() == 0:
            return
        cpu_emb = embeddings.detach().cpu()
        self.storage.append(cpu_emb)
        self.total += cpu_emb.size(0)
        while self.total > self.capacity and self.storage:
            removed = self.storage.popleft()
            self.total -= removed.size(0)

    def sample(self) -> torch.Tensor | None:
        if self.sample_size <= 0 or self.total == 0:
            return None
        stacked = torch.cat(list(self.storage), dim=0)
        if stacked.size(0) <= self.sample_size:
            return stacked.to(self.device)
        perm = torch.randperm(stacked.size(0))[: self.sample_size]
        return stacked[perm].to(self.device)


class ResourceMonitor:
    """Periodically reports GPU/CPU utilization and logs to wandb if available."""

    def __init__(self, interval_s: float, device_ids: List[int]):
        self.interval = max(0.0, interval_s)
        self.device_ids = device_ids or []
        self.last_time = 0.0
        self.nvml_handles = []
        if self.interval > 0 and pynvml is not None and self.device_ids:
            try:
                pynvml.nvmlInit()
                for device_id in self.device_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
                    self.nvml_handles.append((device_id, handle))
            except Exception as exc:  # pragma: no cover
                print(f"[ResourceMonitor] NVML init failed: {exc}")
                self.nvml_handles = []
        else:
            self.nvml_handles = []

    def _collect_gpu_stats(self) -> List[Dict[str, float]]:
        stats = []
        if not self.nvml_handles:
            return stats
        for device_id, handle in self.nvml_handles:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # watts
                stats.append(
                    {
                        "id": device_id,
                        "util_percent": float(util.gpu),
                        "mem_used_gb": mem.used / 1e9,
                        "mem_total_gb": mem.total / 1e9,
                        "power_w": power,
                    }
                )
            except Exception as exc:  # pragma: no cover
                print(f"[ResourceMonitor] Failed to query GPU {device_id}: {exc}")
        return stats

    def _collect_cpu_stats(self) -> Dict[str, float]:
        data = {}
        if psutil is None:
            return data
        try:
            data["cpu_percent"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            data["cpu_mem_used_gb"] = vm.used / 1e9
            data["cpu_mem_total_gb"] = vm.total / 1e9
            data["cpu_count"] = psutil.cpu_count()
        except Exception as exc:  # pragma: no cover
            print(f"[ResourceMonitor] Failed to query CPU stats: {exc}")
        return data

    def maybe_log(self, phase: str, step: int, wandb_logger: "WandbLogger | None" = None):
        if self.interval <= 0:
            return
        now = time.time()
        if now - self.last_time < self.interval:
            return
        self.last_time = now
        gpu_stats = self._collect_gpu_stats()
        cpu_stats = self._collect_cpu_stats()
        log_data: Dict[str, float] = {}
        lines = []
        for stat in gpu_stats:
            gid = stat["id"]
            lines.append(
                f"GPU{gid}: util {stat['util_percent']:.1f}% | "
                f"mem {stat['mem_used_gb']:.2f}/{stat['mem_total_gb']:.2f} GB | "
                f"power {stat['power_w']:.1f} W"
            )
            log_data.update(
                {
                    f"resource/gpu{gid}_util_percent": stat["util_percent"],
                    f"resource/gpu{gid}_mem_used_gb": stat["mem_used_gb"],
                    f"resource/gpu{gid}_mem_total_gb": stat["mem_total_gb"],
                    f"resource/gpu{gid}_power_w": stat["power_w"],
                }
            )
        if cpu_stats:
            cpu_line = (
                f"CPU util {cpu_stats.get('cpu_percent', 0):.1f}% | "
                f"RAM {cpu_stats.get('cpu_mem_used_gb', 0):.2f}/"
                f"{cpu_stats.get('cpu_mem_total_gb', 0):.2f} GB"
            )
            lines.append(cpu_line)
            log_data.update({f"resource/{k}": v for k, v in cpu_stats.items()})
        if lines:
            print(f"[ResourceMonitor][{phase} step {step}] " + " | ".join(lines))
        if wandb_logger is not None and lines:
            log_payload = {**log_data, "resource/step": step, "resource/phase_label": phase}
            wandb_logger.log(log_payload)


class WandbLogger:
    """Thin wrapper to log metrics to Weights & Biases on rank 0 only."""

    def __init__(self, enabled: bool, project: str | None, run_name: str | None, entity: str | None, config: Dict):
        self.enabled = bool(enabled and wandb is not None and project)
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                config=config,
                reinit=True,
            )
        elif enabled and wandb is None:
            print("[WandbLogger] wandb is not installed; logging disabled.")

    def log(self, payload: Dict[str, float]):
        if self.enabled and self.run is not None:
            wandb.log(payload)

    def finish(self):
        if self.enabled and self.run is not None:
            wandb.finish()


def per_positive_infonce_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    mask = positive_mask > 0
    logits = logits - logits.max(dim=1, keepdim=True).values
    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom
    log_prob = log_prob.masked_fill(~mask, 0.0)
    counts = mask.sum(dim=1).clamp_min(1.0)
    loss = -(log_prob.sum(dim=1) / counts)
    return loss.mean()


def diversity_regularizer(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    mask = positive_mask > 0
    if not mask.any():
        return torch.zeros((), device=logits.device)
    logits_pos = logits.masked_fill(~mask, float("-inf")) / max(temperature, 1e-6)
    log_probs = torch.log_softmax(logits_pos, dim=1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).nan_to_num().sum(dim=1)
    valid = mask.sum(dim=1) > 0
    if not valid.any():
        return torch.zeros((), device=logits.device)
    return (-entropy[valid]).mean()


def single_positive_ce_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    retrieval_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy on exactly one positive per row (others masked away)."""
    logits = logits * scale
    if retrieval_mask is not None:
        logits = logits.masked_fill(~retrieval_mask.bool(), float("-inf"))
    if labels is None:
        labels = positive_mask.argmax(dim=1)
    return F.cross_entropy(logits, labels)


def build_loss_fn(
    loss_type: str,
    loss_reduction: str,
    *,
    diversity_weight: float = 0.0,
    diversity_temperature: float = 1.0,
    loss_scale: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor]:
    def compute_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        retrieval_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if loss_type == "query_infonce":
            base = multi_positive_infonce(logits, positive_mask, reduction=loss_reduction)
        elif loss_type == "positive_infonce":
            base = per_positive_infonce_loss(logits, positive_mask)
        elif loss_type == "positive_infonce_diversity":
            base = per_positive_infonce_loss(logits, positive_mask)
            if diversity_weight > 0:
                div = diversity_regularizer(
                    logits,
                    positive_mask,
                    temperature=diversity_temperature,
                )
                base = base + diversity_weight * div
        elif loss_type == "loss5":
            base = single_positive_ce_loss(
                logits,
                positive_mask,
                retrieval_mask=retrieval_mask,
                labels=labels,
                scale=loss_scale,
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return base

    return compute_loss


class CollateFunction:
    """Picklable collate function for DataLoader with multiprocessing."""

    def __init__(
        self,
        builder: SubgraphBuilder,
        feature_cache: NodeFeatureCache,
        seed: int = 42,
        max_candidates: int | None = None,
        expand_per_positive: bool = False,
        graph: MathlibGraph | None = None,
        loss5_neg_source: str | None = None,
        loss5_neg_count: int = 3,
        loss5_pair_batch_size: int | None = None,
    ):
        self.builder = builder
        self.feature_cache = feature_cache
        self.seed = int(seed)
        self.max_candidates = max_candidates
        self.rng = random.Random(self.seed)
        self.expand_per_positive = expand_per_positive
        self.graph = graph
        self.loss5_neg_source = (loss5_neg_source or "").strip().lower() or None
        self.loss5_neg_count = max(0, int(loss5_neg_count))
        self.loss5_pair_batch_size = loss5_pair_batch_size if loss5_pair_batch_size and loss5_pair_batch_size > 0 else None
        if self.loss5_neg_source and self.graph is None:
            raise ValueError("loss5 negative sampling requires a graph reference.")
        if self.loss5_neg_source and self.loss5_neg_source not in {"global", "random_hard"}:
            raise ValueError(f"Unknown loss5_neg_source: {loss5_neg_source}")

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng.seed(self.seed)

    def _loss5_sample_negatives(self, query_id: int, positives: set[int]) -> List[int]:
        """Hammer-style negatives: either global random or random_hard."""
        if not self.loss5_neg_source or self.loss5_neg_count <= 0 or self.graph is None:
            return []
        if self.loss5_neg_source == "global":
            pool = [
                nid
                for nid in self.graph.global_premise_ids
                if nid != query_id and nid not in positives
            ]
            if not pool:
                return []
            if len(pool) <= self.loss5_neg_count:
                return pool
            return self.rng.sample(pool, self.loss5_neg_count)
        if self.loss5_neg_source == "random_hard":
            return self.graph.sample_random_hard(query_id, self.loss5_neg_count, rng=self.rng)
        return []

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # batch may contain a mix of mathlib items (graph query_id) and external items (query_text/query_embedding).
        positives_per_query: List[List[int]] = []
        negs_per_query: List[List[int]] = []

        base_query_ids: List[int] = []
        base_is_graph_node: List[bool] = []
        base_query_embeddings: List[torch.Tensor | None] = []

        is_dynamic_flags = []
        for item in batch:
            pos_tensor = item["positive_ids"]
            pos = pos_tensor.tolist() if hasattr(pos_tensor, "tolist") else list(pos_tensor)
            positives_per_query.append(pos)
            if self.loss5_neg_source:
                qid = int(item.get("query_id", -1))
                negs = self._loss5_sample_negatives(qid, set(pos))
                negs_per_query.append(negs)
            else:
                neg_list: List[int] = []
                for neg_key in ("random_negative_ids", "hard_negative_ids", "global_negative_ids"):
                    neg_vals = item.get(neg_key)
                    if neg_vals is None:
                        continue
                    if hasattr(neg_vals, "numel") and neg_vals.numel() > 0:
                        neg_list.extend(neg_vals.tolist())
                negs_per_query.append(neg_list)

            is_dynamic_flags.append(bool(item.get("is_dynamic", False)))

            if "query_embedding" in item:
                base_query_ids.append(int(item.get("query_id", -1)))
                base_is_graph_node.append(False)
                base_query_embeddings.append(item["query_embedding"])
            else:
                base_query_ids.append(int(item["query_id"]))
                base_is_graph_node.append(True)
                base_query_embeddings.append(None)

        # loss5: enforce pair cap if configured (raise to prompt smaller batch instead of dropping positives).
        if self.expand_per_positive:
            total_pos = sum(len(p) for p in positives_per_query)
            if self.max_candidates is not None:
                safe_cap = self.max_candidates // max(1, 1 + self.loss5_neg_count)
                if total_pos > safe_cap:
                    print(
                        f"[Collate] loss5 expanded positives ({total_pos}) exceed safe cap ({safe_cap}) from max-candidates-per-batch={self.max_candidates} "
                        f"with {self.loss5_neg_count} neg/query. Proceeding; negatives may be trimmed.",
                        flush=True,
                    )

        # rebuild candidate pool from (possibly trimmed) positives and negatives
        candidate_pool = set()
        for pos_list, neg_list in zip(positives_per_query, negs_per_query):
            candidate_pool.update(pos_list)
            candidate_pool.update(neg_list)

        # Apply candidate cap while keeping all positives
        candidate_ids = sorted(candidate_pool)
        original_candidate_count = len(candidate_ids)
        if self.max_candidates is not None and len(candidate_ids) > self.max_candidates:
            positive_set = {pid for sub in positives_per_query for pid in sub}
            negative_pool = [pid for pid in candidate_ids if pid not in positive_set]
            remaining = max(0, self.max_candidates - len(positive_set))
            if remaining < 0:
                remaining = 0  # In rare cases where positives exceed cap, keep all positives
            sampled_negs = (
                self.rng.sample(negative_pool, remaining) if remaining > 0 and len(negative_pool) > remaining else negative_pool[:remaining]
            )
            candidate_ids = sorted(positive_set.union(sampled_negs))
        else:
            original_candidate_count = len(candidate_ids)

        if not candidate_ids:
            raise RuntimeError("Empty candidate pool encountered in batch.")

        col_index = {pid: idx for idx, pid in enumerate(candidate_ids)}
        row_query_ids: List[int] = []
        row_positive_mask: List[torch.Tensor] = []
        retrieval_masks: List[torch.Tensor] = []
        row_is_dynamic: List[bool] = []
        row_is_graph_node: List[bool] = []
        row_query_embeddings: List[torch.Tensor | None] = []

        if self.expand_per_positive:
            for qid, positives, is_dyn, is_graph, q_emb in zip(
                base_query_ids, positives_per_query, is_dynamic_flags, base_is_graph_node, base_query_embeddings
            ):
                for pos_id in positives:
                    mask_row = torch.zeros(len(candidate_ids), dtype=torch.float32, device="cpu")
                    retrieval_row = torch.ones(len(candidate_ids), dtype=torch.bool, device="cpu")
                    pos_col = col_index.get(pos_id)
                    if pos_col is not None:
                        mask_row[pos_col] = 1.0
                        # Mask out other positives for the same query so they are never treated as negatives.
                        for other_pos in positives:
                            if other_pos == pos_id:
                                continue
                            other_col = col_index.get(other_pos)
                            if other_col is not None:
                                retrieval_row[other_col] = False
                    row_query_ids.append(qid)
                    row_positive_mask.append(mask_row)
                    retrieval_masks.append(retrieval_row)
                    row_is_dynamic.append(is_dyn)
                    row_is_graph_node.append(is_graph)
                    row_query_embeddings.append(q_emb)
        else:
            for is_graph, q_emb, positives, qid in zip(base_is_graph_node, base_query_embeddings, positives_per_query, base_query_ids):
                mask_row = torch.zeros(len(candidate_ids), dtype=torch.float32, device='cpu')
                retrieval_row = torch.ones(len(candidate_ids), dtype=torch.bool, device="cpu")
                first_pos_col = None
                for pid in positives:
                    col = col_index.get(pid)
                    if col is not None:
                        mask_row[col] = 1.0
                        if first_pos_col is None:
                            first_pos_col = col
                        else:
                            retrieval_row[col] = False
                row_positive_mask.append(mask_row)
                retrieval_masks.append(retrieval_row)
                row_is_graph_node.append(is_graph)
                row_query_embeddings.append(q_emb)
                row_query_ids.append(qid)

        effective_query_ids = row_query_ids if self.expand_per_positive else base_query_ids

        positive_mask = (
            torch.stack(row_positive_mask, dim=0)
            if row_positive_mask
            else torch.zeros((0, len(candidate_ids)), dtype=torch.float32, device="cpu")
        )
        retrieval_mask = torch.stack(retrieval_masks, dim=0) if retrieval_masks else None

        # Query embeddings:
        # - mathlib queries: feature_cache full_node lookup
        # - external queries: use precomputed item["query_embedding"] (recommended) or encode query_text if needed
        query_embeddings_list: List[torch.Tensor] = []
        graph_rows: List[int] = [i for i, is_graph in enumerate(row_is_graph_node) if is_graph]
        ext_rows: List[int] = [i for i, is_graph in enumerate(row_is_graph_node) if not is_graph]

        # Prepare a placeholder list and fill in row order.
        row_embs: List[torch.Tensor | None] = [None] * len(effective_query_ids)

        if graph_rows:
            graph_qids = [effective_query_ids[i] for i in graph_rows]
            graph_embs = self.feature_cache.get_full_nodes(graph_qids, device="cpu")
            for j, row_i in enumerate(graph_rows):
                row_embs[row_i] = graph_embs[j].detach().cpu()

        if ext_rows:
            missing_texts: List[str] = []
            missing_rows: List[int] = []
            for row_i in ext_rows:
                emb = row_query_embeddings[row_i]
                if emb is not None:
                    row_embs[row_i] = emb.detach().cpu()
                else:
                    text = batch[row_i].get("query_text")  # may fail if expand_per_positive
                    # In expand_per_positive mode, batch indices don't align with rows; require precomputed embeddings.
                    if self.expand_per_positive:
                        raise RuntimeError("External queries require precomputed query_embedding when loss5 is enabled.")
                    if not isinstance(text, str) or not text:
                        raise RuntimeError("External query missing query_text.")
                    missing_texts.append(text)
                    missing_rows.append(row_i)

            if missing_texts:
                # Encode on the fly (CPU) as a fallback; recommended to precompute query_embedding for speed.
                with torch.no_grad():
                    enc = self.feature_cache.text_encoder(missing_texts).detach().cpu()
                for j, row_i in enumerate(missing_rows):
                    row_embs[row_i] = enc[j]

        for emb in row_embs:
            if emb is None:
                raise RuntimeError("Failed to build query embeddings for batch.")
            query_embeddings_list.append(emb)
        query_embeddings = torch.stack(query_embeddings_list, dim=0) if query_embeddings_list else torch.empty((0, self.feature_cache.statement_dim), dtype=torch.float32)
        premise_inputs = self.builder.build_batch(candidate_ids, device='cpu', seed_text_mode="premise")

        # Safety check: ensure every query has at least one positive
        pos_counts_per_row = positive_mask.sum(dim=1)
        if (pos_counts_per_row == 0).any():
            zero_rows = torch.where(pos_counts_per_row == 0)[0].tolist()
            problem_queries = [effective_query_ids[i] for i in zero_rows]
            raise RuntimeError(
                f"Batch has queries with no positives in candidate pool: {problem_queries}. "
                f"This should not happen - check data filtering logic."
            )

        return {
            "query_ids": torch.tensor(effective_query_ids, dtype=torch.long),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "query_embeddings": query_embeddings,
            "premise_inputs": premise_inputs,
            "positive_mask": positive_mask,
            "retrieval_mask": retrieval_mask,
            "retrieval_labels": (
                torch.tensor([mask.argmax().item() for mask in row_positive_mask], dtype=torch.long)
                if self.expand_per_positive
                else None
            ),
            "is_dynamic": torch.tensor(row_is_dynamic if self.expand_per_positive else is_dynamic_flags, dtype=torch.bool),
            "query_is_graph_node": torch.tensor(row_is_graph_node, dtype=torch.bool),
            "candidate_count": torch.tensor(len(candidate_ids), dtype=torch.long),
            "positive_count": torch.tensor(int(positive_mask.sum().item()), dtype=torch.long),
            "negative_count": torch.tensor(int(positive_mask.numel() - positive_mask.sum().item()), dtype=torch.long),
            "dropped_candidates": torch.tensor(max(0, original_candidate_count - len(candidate_ids)), dtype=torch.long),
        }


def run_epoch(
    loader: DataLoader,
    model: PremiseSelectorModel,
    optimizer: torch.optim.Optimizer,
    *,
    train: bool,
    loss_fn,
    lr_schedule: Dict[str, float | int | str] | None = None,
    update_state: Dict[str, int] | None = None,
    distributed: bool = False,
    log_progress: bool = True,
    memory_bank_manager: MemoryBankManager | None = None,
    grad_accum_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    cross_batch_memory: CrossBatchMemory | None = None,
    wandb_logger: WandbLogger | None = None,
    resource_monitor: ResourceMonitor | None = None,
    phase: str = "train",
    epoch_index: int = 1,
) -> Dict[str, float]:
    forward_model = model  # Keep the DDP wrapper for forward() so gradients sync correctly
    base_model = model.module if isinstance(model, DDP) else model

    if train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    metric_sums: Dict[str, float] = {}
    metric_sums_fixed: Dict[str, float] = {}
    metric_sums_dynamic: Dict[str, float] = {}
    valid_rows = 0
    valid_rows_fixed = 0
    valid_rows_dynamic = 0
    batch_count = 0
    dropped_sum = 0.0

    iterator = loader
    progress = None
    if log_progress:
        progress = tqdm(loader, desc="train" if train else "eval")
        iterator = progress

    accum_steps = max(1, grad_accum_steps)
    in_accum = 0
    if train:
        optimizer.zero_grad()

    total_batches = len(loader)
    this_rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    metric_group_timeout = os.environ.get("METRIC_GROUP_TIMEOUT_MINUTES") or os.environ.get("DDP_TIMEOUT_MINUTES")
    metric_group, metric_backend = _get_metric_group(distributed, timeout_minutes=metric_group_timeout)

    for batch_idx, batch in enumerate(iterator, start=1):
        # Move batch data to model device
        device = next(base_model.parameters()).device
        positive_mask = batch["positive_mask"].to(device)
        query_embeddings = batch["query_embeddings"].to(device)
        premise_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch["premise_inputs"].items()}
        retrieval_mask = batch.get("retrieval_mask")
        if retrieval_mask is not None:
            retrieval_mask = retrieval_mask.to(device)
        retrieval_labels = batch.get("retrieval_labels")
        if retrieval_labels is not None:
            retrieval_labels = retrieval_labels.to(device)

        try:
            with torch.set_grad_enabled(train):
                # Use automatic mixed precision if enabled
                if use_amp and train:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        q, p, logits = forward_model(query_embeddings, premise_inputs)
                        extra_negs = cross_batch_memory.sample() if cross_batch_memory else None
                        aug_mask = positive_mask
                        aug_p = p
                        aug_retrieval = retrieval_mask
                        if extra_negs is not None and extra_negs.numel() > 0:
                            aug_p = torch.cat([aug_p, extra_negs], dim=0)
                            pad = torch.zeros(positive_mask.size(0), extra_negs.size(0), device=positive_mask.device, dtype=positive_mask.dtype)
                            aug_mask = torch.cat([aug_mask, pad], dim=1)
                            if aug_retrieval is not None:
                                pad_mask = torch.ones(
                                    retrieval_mask.size(0),
                                    extra_negs.size(0),
                                    device=retrieval_mask.device,
                                    dtype=retrieval_mask.dtype,
                                )
                                aug_retrieval = torch.cat([aug_retrieval, pad_mask], dim=1)
                            logits = base_model.compute_logits(q, aug_p)
                        raw_loss = loss_fn(logits, aug_mask, retrieval_mask=aug_retrieval, labels=retrieval_labels)
                        loss = raw_loss / accum_steps
                else:
                    q, p, logits = forward_model(query_embeddings, premise_inputs)
                    extra_negs = cross_batch_memory.sample() if cross_batch_memory else None
                    aug_mask = positive_mask
                    aug_p = p
                    aug_retrieval = retrieval_mask
                    if extra_negs is not None and extra_negs.numel() > 0:
                        aug_p = torch.cat([aug_p, extra_negs], dim=0)
                        pad = torch.zeros(positive_mask.size(0), extra_negs.size(0), device=positive_mask.device, dtype=positive_mask.dtype)
                        aug_mask = torch.cat([aug_mask, pad], dim=1)
                        if aug_retrieval is not None:
                            pad_mask = torch.ones(
                                retrieval_mask.size(0),
                                extra_negs.size(0),
                                device=retrieval_mask.device,
                                dtype=retrieval_mask.dtype,
                            )
                            aug_retrieval = torch.cat([aug_retrieval, pad_mask], dim=1)
                        logits = base_model.compute_logits(q, aug_p)
                    raw_loss = loss_fn(logits, aug_mask, retrieval_mask=aug_retrieval, labels=retrieval_labels)
                    loss = raw_loss
                    if train:
                        loss = raw_loss / accum_steps
        except RuntimeError as exc:
            msg = str(exc).lower()
            if train and ("out of memory" in msg or "cuda error" in msg):
                if log_progress:
                    print(f"[Rank {this_rank}] OOM/RuntimeError on batch {batch_idx}; skipping batch. Error: {exc}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise

        # Synchronize NaN/Inf detection so all ranks skip the same batches.
        local_has_nan = torch.isnan(raw_loss) or torch.isinf(raw_loss)
        if distributed and dist.is_initialized():
            has_nan_tensor = torch.tensor([1.0 if local_has_nan else 0.0], 
                                          dtype=torch.float32, device=device)
            # Use the Gloo metric group for error-status synchronization.
            try:
                dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX, group=metric_group)
            except Exception:
                # Fallback to default group if metric_group fails
                dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
            any_rank_has_nan = has_nan_tensor.item() > 0.5
        else:
            any_rank_has_nan = local_has_nan

        if any_rank_has_nan:
            if log_progress and this_rank == 0:
                print(f"[DDP Sync] NaN/Inf loss detected on batch {batch_idx}; all ranks skipping.", flush=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if train:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            in_accum += 1

        if train:
            should_step = (in_accum >= accum_steps) or (batch_idx == total_batches)
            if should_step:
                if lr_schedule is not None and update_state is not None:
                    step_idx = int(update_state.get("step", 0))
                    lr = _compute_lr(
                        base_lr=float(lr_schedule["base_lr"]),
                        min_lr=float(lr_schedule["min_lr"]),
                        scheduler=str(lr_schedule["scheduler"]),
                        step=step_idx,
                        total_steps=int(lr_schedule["total_steps"]),
                        warmup_steps=int(lr_schedule["warmup_steps"]),
                    )
                    _apply_lr(optimizer, lr)
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if lr_schedule is not None and update_state is not None:
                    update_state["step"] = int(update_state.get("step", 0)) + 1
                optimizer.zero_grad()
                in_accum = 0
            if memory_bank_manager is not None:
                query_ids = batch["query_ids"].tolist()
                query_embs = q.detach().cpu()
                is_graph = batch.get("query_is_graph_node")
                if is_graph is not None:
                    mask = is_graph.to(torch.bool).cpu()
                    if mask.numel() == query_embs.size(0):
                        keep_idx = torch.where(mask)[0]
                        if keep_idx.numel() > 0:
                            memory_bank_manager.log_batch(
                                [query_ids[i] for i in keep_idx.tolist()],
                                query_embs.index_select(0, keep_idx),
                                batch["candidate_ids"].tolist(),
                                p.detach().cpu(),
                            )
                    else:
                        memory_bank_manager.log_batch(
                            query_ids,
                            query_embs,
                            batch["candidate_ids"].tolist(),
                            p.detach().cpu(),
                        )
                else:
                    memory_bank_manager.log_batch(
                        query_ids,
                        query_embs,
                        batch["candidate_ids"].tolist(),
                        p.detach().cpu(),
                    )
            if cross_batch_memory is not None:
                cross_batch_memory.add(p)

        loss_value = raw_loss.item()
        loss_sum += loss_value
        batch_count += 1
        base_logits = logits.detach()
        base_mask = aug_mask.detach()
        batch_metrics, batch_valid = batch_ranking_metrics(base_logits, base_mask)
        for key, value in batch_metrics.items():
            if key == "valid_queries":
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + value
        valid_rows += batch_valid

        is_dynamic = batch.get("is_dynamic")
        if is_dynamic is not None:
            dyn_mask = is_dynamic.to(base_logits.device).bool()
            fixed_mask = ~dyn_mask
            if fixed_mask.any():
                fixed_logits = base_logits[fixed_mask]
                fixed_pos = base_mask[fixed_mask]
                batch_metrics_fixed, batch_valid_fixed = batch_ranking_metrics(fixed_logits, fixed_pos)
                for key, value in batch_metrics_fixed.items():
                    if key == "valid_queries":
                        continue
                    metric_sums_fixed[key] = metric_sums_fixed.get(key, 0.0) + value
                valid_rows_fixed += batch_valid_fixed
            if dyn_mask.any():
                dyn_logits = base_logits[dyn_mask]
                dyn_pos = base_mask[dyn_mask]
                batch_metrics_dyn, batch_valid_dyn = batch_ranking_metrics(dyn_logits, dyn_pos)
                for key, value in batch_metrics_dyn.items():
                    if key == "valid_queries":
                        continue
                    metric_sums_dynamic[key] = metric_sums_dynamic.get(key, 0.0) + value
                valid_rows_dynamic += batch_valid_dyn
        if progress is not None:
            progress.set_postfix(loss=loss_value)

        # Periodically clear CUDA cache to prevent memory accumulation
        if batch_idx % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

            # Log memory usage for monitoring
            mem_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9    # GB
            mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            mem_free = mem_total - mem_reserved

            if log_progress:
                print(f"\n[Memory Monitor] Iteration {batch_idx}: "
                      f"Allocated={mem_allocated:.2f}GB, "
                      f"Reserved={mem_reserved:.2f}GB, "
                      f"Free={mem_free:.2f}GB, "
                      f"Total={mem_total:.2f}GB")

        if resource_monitor is not None:
            global_step = (epoch_index - 1) * total_batches + batch_idx
            resource_monitor.maybe_log(phase, global_step, wandb_logger)

        # Log candidate counts periodically (rank 0 only)
        if batch_idx % 50 == 0 and this_rank == 0:
            cand_cnt = int(batch.get("candidate_count", torch.tensor(0)).item())
            pos_cnt = int(batch.get("positive_count", torch.tensor(0)).item())
            neg_cnt = int(batch.get("negative_count", torch.tensor(0)).item())
            dropped_cnt = int(batch.get("dropped_candidates", torch.tensor(0)).item())
            print(f"[BatchStats] step={batch_idx} candidates={cand_cnt} dropped={dropped_cnt} pos={pos_cnt} neg={neg_cnt}", flush=True)
        dropped_sum += float(batch.get("dropped_candidates", torch.tensor(0)).item())

    metric_sums["dropped_candidates_sum"] = metric_sums.get("dropped_candidates_sum", 0.0) + dropped_sum
    metric_sums["batches"] = metric_sums.get("batches", 0.0) + batch_count

    if distributed and dist.is_initialized():
        sync_device = next(model.parameters()).device
        reduce_device = torch.device("cpu") if metric_backend == "gloo" else sync_device
        if sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)

        stats_tensor = torch.tensor([loss_sum, float(batch_count), float(valid_rows)], device=reduce_device)
        dist.all_reduce(stats_tensor, group=metric_group)
        stats_tensor = stats_tensor.to("cpu")
        loss_sum, batch_count, valid_rows = stats_tensor.tolist()
        valid_rows = int(valid_rows)

        def _reduce_metric_dict(metric_dict: Dict[str, float]) -> None:
            # Reduce the same sorted union of metric keys on every rank.
            local_keys = list(metric_dict.keys())
            world = dist.get_world_size(group=metric_group)
            gathered: List[List[str]] = [None for _ in range(world)]  # type: ignore[list-item]
            dist.all_gather_object(gathered, local_keys, group=metric_group)
            all_keys = sorted({k for sub in gathered if sub for k in sub})

            for key in all_keys:
                tensor = torch.tensor(metric_dict.get(key, 0.0), device=reduce_device)
                dist.all_reduce(tensor, group=metric_group)
                metric_dict[key] = tensor.item()

        _reduce_metric_dict(metric_sums)
        _reduce_metric_dict(metric_sums_fixed)

        valid_rows_fixed_tensor = torch.tensor(float(valid_rows_fixed), device=reduce_device)
        dist.all_reduce(valid_rows_fixed_tensor, group=metric_group)
        valid_rows_fixed = int(valid_rows_fixed_tensor.item())

        _reduce_metric_dict(metric_sums_dynamic)

        valid_rows_dynamic_tensor = torch.tensor(float(valid_rows_dynamic), device=reduce_device)
        dist.all_reduce(valid_rows_dynamic_tensor, group=metric_group)
        valid_rows_dynamic = int(valid_rows_dynamic_tensor.item())
    else:
        metric_sums["dropped_candidates_sum"] = metric_sums.get("dropped_candidates_sum", 0.0) + dropped_sum
        metric_sums["batches"] = metric_sums.get("batches", 0.0) + batch_count

    avg_loss = loss_sum / max(batch_count, 1.0)
    metrics = {"loss": avg_loss}
    drop_sum_metric = metric_sums.pop("dropped_candidates_sum", 0.0)
    drop_batches_metric = metric_sums.pop("batches", 0.0)
    denom = max(valid_rows, 1)
    for key, value in metric_sums.items():
        metrics[key] = value / denom
    metrics["avg_dropped_candidates"] = drop_sum_metric / max(drop_batches_metric, 1.0)
    metrics["valid_queries"] = valid_rows
    if valid_rows_fixed > 0:
        for key, value in metric_sums_fixed.items():
            metrics[f"fixed_{key}"] = value / valid_rows_fixed
        metrics["fixed_valid_queries"] = valid_rows_fixed
    if valid_rows_dynamic > 0:
        for key, value in metric_sums_dynamic.items():
            metrics[f"dynamic_{key}"] = value / valid_rows_dynamic
        metrics["dynamic_valid_queries"] = valid_rows_dynamic
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RelGT-based premise selector on mathlib.")
    parser.add_argument("--graph-path", type=str, required=True)
    parser.add_argument(
        "--premise-filter-config",
        type=str,
        default="",
        help="Optional JSON config for filtering premise candidates/positives (default: disabled).",
    )
    parser.add_argument(
        "--v19-align",
        action="store_true",
        help=(
            "Align data/positives/splits with v19 behavior: disable premise_filter_config, "
            "force ratio splits, and restrict positives to configured edge types."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Learning rate schedule applied per optimizer step (after grad accumulation).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio of total update steps (0 disables).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps override (takes precedence over warmup-ratio when >0).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Final learning rate floor for linear/cosine schedules.",
    )
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument(
        "--subgraph-mode",
        type=str,
        default="0",
        help=(
            "Subgraph sampling mode. "
            "0: random 2-hop BFS (default); "
            "1/2/3: outgoing / incoming / both 1-hop neighbors (full set, MLP selector trims if enabled); "
            "4/5/6: same as 1/2/3 but only theorem/lemma/definition neighbors; "
            "7-12: modes 1-6 without selector trimming; "
            "13: global random nodes (dummy edges/degrees/hops, no selector). "
            "Accepts numeric IDs or aliases like out_1hop, in_1hop_structured_full."
        ),
    )
    parser.add_argument(
        "--select-top-k",
        type=int,
        default=None,
        help="Learned top-k node selector inside the encoder (keep this many nodes per subgraph).",
    )
    parser.add_argument(
        "--subgraph-hop1-budget",
        type=int,
        default=0,
        help="Optional cap for 1-hop nodes in random_2hop sampling (0 disables).",
    )
    parser.add_argument(
        "--subgraph-hop2-budget",
        type=int,
        default=0,
        help="Optional cap for 2-hop nodes in random_2hop sampling (0 disables).",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--local-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--global-dim",
        type=int,
        default=None,
        help="Dimensionality for the global attention pathway (default: hidden-dim).",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-encoder-device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-val-queries", type=int, default=0)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument(
        "--train-repeat",
        type=int,
        default=1,
        help="Repeat the training query list this many times (for ablations).",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="mathlib",
        choices=["mathlib", "combined", "mixed"],
        help="mathlib: original graph splits; combined: external JSONL only; mixed: union of mathlib and external.",
    )
    parser.add_argument("--combined-jsonl-path", type=str, default=None, help="Path to external combined JSONL.")
    parser.add_argument("--combined-val-count", type=int, default=1000, help="External val size.")
    parser.add_argument("--combined-test-count", type=int, default=1000, help="External test size.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="ratio",
        choices=["ratio", "fixed_holdout"],
        help="Dataset split strategy. ratio: use train/val ratios; fixed_holdout: val/test counts with holdout files.",
    )
    parser.add_argument("--val-count", type=int, default=2000, help="Val size when split-mode=fixed_holdout.")
    parser.add_argument("--test-count", type=int, default=2000, help="Test size when split-mode=fixed_holdout.")
    parser.add_argument(
        "--holdout-per-split",
        type=int,
        default=1000,
        help="Queries per split drawn from holdout files when split-mode=fixed_holdout.",
    )
    parser.add_argument(
        "--out-dir", type=str, default="runs/premise_selector",
        help="Output directory, used exactly as provided; checkpoints are saved directly inside it.",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="alibaba-gte-modernbert-ft",
        help=(
            "Embedding model to use (alibaba-gte-modernbert-ft)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and sampling.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for text encoder (default: 512)",
    )
    parser.add_argument(
        "--sub-node-max-length",
        type=int,
        default=512,
        help="Maximum sequence length for premise sub_node text (name+signature+lean4_code).",
    )
    parser.add_argument(
        "--proof-pattern-mode",
        type=str,
        default="none",
        choices=["none", "id_only", "id_text", "full_text"],
        help="How to encode proof pattern fields for premise features: none (disabled), id_only (numeric id), "
        "id_text (identified_proof_pattern + proof_pattern_explanation via text encoder), "
        "full_text (identified_proof_pattern + explanation + why_smart via text encoder).",
    )
    parser.add_argument(
        "--proof-pattern-embedding",
        type=str,
        default=None,
        help="Embedding model for proof pattern text (defaults to --embedding when not set). Ignored if mode is none or id_only.",
    )
    parser.add_argument(
        "--proof-pattern-max-length",
        type=int,
        default=768,
        help="Max token length when encoding proof pattern text (concatenated fields).",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned memory for DataLoader transfers.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches loaded in advance per worker (requires num_workers > 0).",
    )
    parser.add_argument(
        "--max-candidates-per-batch",
        type=int,
        default=None,
        help="Upper bound on candidates (positives kept, negatives downsampled).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for validation/test loaders (default: half of train batch).",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers for validation/test (default: half of train workers).",
    )
    parser.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=1,
        help="Prefetch factor for validation/test loaders (default: 1).",
    )
    parser.add_argument(
        "--eval-pin-memory",
        action="store_true",
        help="Use pinned memory for validation/test loaders (default: disabled).",
    )
    parser.add_argument(
        "--eval-dynamic-fraction",
        type=float,
        default=0.05,
        help="Fraction of validation/test queries that keep dynamic candidate sampling (default: 0.05).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default=None,
        help="Directory to store/load precomputed node features.",
    )
    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        default=512,
        help="How many nodes to encode per chunk when building the feature cache.",
    )
    parser.add_argument(
        "--force-precompute-features",
        action="store_true",
        help="Recompute feature cache even if cached tensors already exist.",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="nccl",
        help="Backend to use for torch.distributed (when launched via torchrun).",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Optional path to a checkpoint file or directory to resume training from.",
    )
    parser.add_argument(
        "--memory-bank-dir",
        type=str,
        default=None,
        help="Directory to dump per-epoch embeddings for ANN-based hard negatives.",
    )
    parser.add_argument(
        "--memory-bank-topk",
        type=int,
        default=256,
        help="How many ANN neighbors to retain per query when building the memory bank.",
    )
    parser.add_argument(
        "--memory-bank-chunk-size",
        type=int,
        default=1024,
        help="Queries processed per chunk when computing ANN similarities.",
    )
    parser.add_argument(
        "--memory-bank-per-query",
        type=int,
        default=0,
        help="How many negatives to draw from the memory bank for each query (counts toward the random quota).",
    )
    parser.add_argument(
        "--memory-bank-build-device",
        type=str,
        default=None,
        help="Device to run ANN similarity (default: CUDA if available).",
    )
    parser.add_argument(
        "--random-negatives-per-query",
        type=int,
        default=None,
        help="Override for random_per_query in negative sampling (default: config value).",
    )
    parser.add_argument(
        "--random-hard-negatives-per-query",
        type=int,
        default=None,
        help="Override for random_hard_per_query in negative sampling (default: config value).",
    )
    parser.add_argument(
        "--hard-negatives-per-query",
        type=int,
        default=None,
        help="Override for hard_per_query in negative sampling (default: config value).",
    )
    parser.add_argument(
        "--global-negatives-per-query",
        type=int,
        default=None,
        help="Override for global_per_query in negative sampling (default: config value).",
    )
    parser.add_argument(
        "--fixed-retrieval-negatives",
        action="store_true",
        help=(
            "Override ALL negatives with fixed checkpoint top-k retrieval negatives "
            "(no random/hard/global/memory-bank/cross-batch)."
        ),
    )
    parser.add_argument(
        "--fixed-retrieval-topk",
        type=int,
        default=10,
        help="How many non-positive premises to use per query when fixed-retrieval-negatives is enabled.",
    )
    parser.add_argument(
        "--fixed-retrieval-chunk-size",
        type=int,
        default=256,
        help="Query chunk size when building fixed retrieval negatives (matmul chunking).",
    )
    parser.add_argument(
        "--fixed-retrieval-build-device",
        type=str,
        default=None,
        help="Device used to build fixed retrieval negatives (default: cuda if available).",
    )
    parser.add_argument(
        "--fixed-retrieval-cache-dir",
        type=str,
        default=None,
        help="Where to cache fixed retrieval negatives (default: <out_dir>/fixed_retrieval_negatives).",
    )
    parser.add_argument(
        "--fixed-retrieval-force-rebuild",
        action="store_true",
        help="Recompute fixed retrieval negatives even if cache exists.",
    )
    parser.add_argument(
        "--fixed-retrieval-strict",
        action="store_true",
        help="Error if any query has no cached fixed negatives.",
    )
    parser.add_argument(
        "--mathlib-fixed-retrieval-checkpoint",
        type=str,
        default=None,
        help="Checkpoint used to mine fixed negatives for mathlib graph queries.",
    )
    parser.add_argument(
        "--mathlib-fixed-premise-embeddings",
        type=str,
        default=None,
        help="Premise embeddings file used to mine fixed negatives for mathlib graph queries.",
    )
    parser.add_argument(
        "--combined-fixed-retrieval-checkpoint",
        type=str,
        default=None,
        help="Checkpoint used to mine fixed negatives for combined/external queries.",
    )
    parser.add_argument(
        "--combined-fixed-premise-embeddings",
        type=str,
        default=None,
        help="Premise embeddings file used to mine fixed negatives for combined/external queries.",
    )
    parser.add_argument(
        "--cross-batch-negs",
        type=int,
        default=32,
        help="How many cross-batch negatives to concatenate per query (set 0 to disable).",
    )
    parser.add_argument(
        "--cross-batch-queue",
        type=int,
        default=4096,
        help="Total cross-batch negative pool capacity.",
    )
    parser.add_argument(
        "--resource-monitor-interval",
        type=float,
        default=0.0,
        help="Interval (seconds) to print/log system stats (0 disables).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (enable logging if provided).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Run name for wandb logging.",
    )
    parser.add_argument(
        "--loss-reduction",
        type=str,
        choices=["query", "positive"],
        default="query",
        help="query: average per query (sum positive prob); positive: average per positive (sum then divide by num positives).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["query_infonce", "positive_infonce", "positive_infonce_diversity", "loss5"],
        default="positive_infonce",
        help="Loss formulation to optimize.",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.0,
        help="Weight for the diversity regularizer (only used by positive_infonce_diversity).",
    )
    parser.add_argument(
        "--diversity-temperature",
        type=float,
        default=1.0,
        help="Temperature for the diversity regularizer softmax.",
    )
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=20.0,
        help="Scale logits before loss (loss5/CE).",
    )
    parser.add_argument(
        "--loss5-neg-source",
        type=str,
        default=None,
        choices=["global", "random_hard"],
        help="When loss-type=loss5: source for per-positive negatives (global or random_hard).",
    )
    parser.add_argument(
        "--loss5-neg-count",
        type=int,
        default=3,
        help="When loss-type=loss5: negatives per positive (Hammer default=3).",
    )
    parser.add_argument(
        "--loss5-pair-batch-size",
        type=int,
        default=None,
        help="Optional upper bound on expanded positives per batch for loss5. If absent, collate auto-checks against max-candidates.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--subgraph-cache-dir",
        type=str,
        default=None,
        help="Directory to store/load precomputed subgraph structures (huge speedup for large max_nodes).",
    )
    parser.add_argument(
        "--force-precompute-subgraphs",
        action="store_true",
        help="Recompute subgraph cache even if it already exists.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision training for faster computation.",
    )
    return parser.parse_args()


def _compute_lr(
    *,
    base_lr: float,
    min_lr: float,
    scheduler: str,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    min_lr = max(0.0, float(min_lr))
    base_lr = float(base_lr)
    final_factor = (min_lr / base_lr) if base_lr > 0 else 0.0
    final_factor = max(0.0, min(1.0, final_factor))

    # Warmup: linearly ramp from ~0 -> 1.0 over warmup_steps updates.
    if warmup_steps > 0 and step < warmup_steps:
        warm = float(step + 1) / float(warmup_steps)
        return base_lr * warm

    if scheduler == "constant" or total_steps == warmup_steps:
        return base_lr

    # Decay portion (inclusive of step 0 after warmup).
    denom = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(denom)
    progress = max(0.0, min(1.0, progress))

    if scheduler == "linear":
        factor = final_factor + (1.0 - final_factor) * (1.0 - progress)
        return base_lr * factor

    if scheduler == "cosine":
        import math
        factor = final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * factor

    return base_lr


def _apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _wait_for_feature_cache(feature_cache: NodeFeatureCache, cache_dir: str | Path, modes=("premise", "statement")) -> None:
    cache_path = Path(cache_dir)
    targets = [feature_cache._cache_file(cache_path, mode) for mode in modes]
    while not all(path.exists() for path in targets):
        time.sleep(5)


def _build_loss5_slices(dataset: PremiseSelectionDataset, slice_cap: int) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """Split queries with many positives into multiple slices capped at slice_cap."""
    slices: List[Tuple[int, int, int]] = []
    lengths: List[int] = []
    max_pos = getattr(dataset.neg_cfg, "max_positives", 32)
    for idx, qid in enumerate(dataset.query_ids):
        pos_ids = dataset.graph.sample_positives(qid, max_pos)
        if not pos_ids:
            continue
        start = 0
        while start < len(pos_ids):
            end = min(start + slice_cap, len(pos_ids))
            slices.append((idx, start, end))
            lengths.append(end - start)
            start = end
    return slices, lengths


class Loss5SlicedDataset(torch.utils.data.Dataset):
    """Wraps a PremiseSelectionDataset and slices positives for loss5 batching."""

    def __init__(self, base: PremiseSelectionDataset, slices: List[Tuple[int, int, int]]):
        self.base = base
        self.slices = slices  # (base_idx, start, end)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx, start, end = self.slices[idx]
        item = self.base[base_idx]
        pos = item["positive_ids"]
        item["positive_ids"] = pos[start:end]
        return item


class Loss5PairBatchSampler(Sampler[List[int]]):
    """Static batch sampler that groups sliced items by expanded positive count for loss5."""

    def __init__(
        self,
        slice_lengths: List[int],
        *,
        pair_cap: int,
        neg_per_query: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.slice_lengths = slice_lengths
        self.pair_cap = max(1, int(pair_cap))
        self.neg_per_query = max(0, int(neg_per_query))
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))
        self.seed = seed
        self.shuffle = shuffle
        self._batches = self._build_batches()
        self._rank_batches = self._shard_batches(self._batches, self.world_size, self.rank)

    def _build_batches(self) -> List[List[int]]:
        q_indices = list(range(len(self.slice_lengths)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(q_indices)

        batches: List[List[int]] = []
        current: List[int] = []
        current_pairs = 0
        for idx in q_indices:
            est_pairs = max(1, self.slice_lengths[idx])  # expanded positives
            if est_pairs > self.pair_cap:
                # Put this slice in its own batch; allow exceeding pair_cap to keep all positives.
                if current:
                    batches.append(current)
                    current = []
                    current_pairs = 0
                batches.append([idx])
                continue

            if current and (current_pairs + est_pairs) > self.pair_cap:
                batches.append(current)
                current = []
                current_pairs = 0

            current.append(idx)
            current_pairs += est_pairs

        if current:
            batches.append(current)
        return batches

    def _shard_batches(self, batches: List[List[int]], world_size: int, rank: int) -> List[List[int]]:
        if world_size <= 1:
            return batches
        total = len(batches)
        keep = total - (total % world_size)
        batches = batches[:keep]
        if keep == 0:
            return []
        chunk = keep // world_size
        start = rank * chunk
        end = start + chunk
        return batches[start:end]

    def __iter__(self):
        return iter(self._rank_batches)

    def __len__(self) -> int:
        return len(self._rank_batches)


def compute_dataset_stats(dataset: PremiseSelectionDataset) -> Dict[str, float]:
    """Compute average positive/negative counts per query in the dataset."""
    total_positives = 0
    total_random_negatives = 0
    total_hard_negatives = 0
    total_global_negatives = 0
    total_queries = len(dataset)

    for i in range(total_queries):
        sample = dataset[i]
        total_positives += len(sample["positive_ids"])
        total_random_negatives += len(sample["random_negative_ids"])
        total_hard_negatives += len(sample["hard_negative_ids"])
        total_global_negatives += len(sample.get("global_negative_ids", ()))

    avg_positives = total_positives / max(total_queries, 1)
    avg_random_negatives = total_random_negatives / max(total_queries, 1)
    avg_hard_negatives = total_hard_negatives / max(total_queries, 1)
    avg_global_negatives = total_global_negatives / max(total_queries, 1)
    avg_total_negatives = (total_random_negatives + total_hard_negatives + total_global_negatives) / max(total_queries, 1)

    return {
        "total_queries": total_queries,
        "avg_positives": avg_positives,
        "avg_random_negatives": avg_random_negatives,
        "avg_hard_negatives": avg_hard_negatives,
        "avg_global_negatives": avg_global_negatives,
        "avg_total_negatives": avg_total_negatives,
    }


def format_metrics_display(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics into a human-readable summary string."""
    lines = []

    # Basic metrics.
    if "loss" in metrics:
        lines.append(f"{prefix}Loss: {metrics['loss']:.4f}")

    # Recall@k metrics (sorted by numeric k).
    recall_keys = [k for k in metrics.keys() if k.startswith("recall@")]
    if recall_keys:
        recall_keys = sorted(recall_keys, key=lambda x: int(x.split('@')[1]))
        recall_values = [f"{k.replace('recall@', '')}:{metrics[k]:.4f}" for k in recall_keys]
        lines.append(f"{prefix}Recall@k: {' | '.join(recall_values)}")

    # Coverage@k metrics (sorted by numeric k).
    coverage_keys = [k for k in metrics.keys() if k.startswith("coverage@")]
    if coverage_keys:
        coverage_keys = sorted(coverage_keys, key=lambda x: int(x.split('@')[1]))
        coverage_values = [f"{k.replace('coverage@', '')}:{metrics[k]:.4f}" for k in coverage_keys]
        lines.append(f"{prefix}Coverage@k: {' | '.join(coverage_values)}")

    # NDCG@k metrics (sorted by numeric k).
    ndcg_keys = [k for k in metrics.keys() if k.startswith("ndcg@")]
    if ndcg_keys:
        ndcg_keys = sorted(ndcg_keys, key=lambda x: int(x.split('@')[1]))
        ndcg_values = [f"{k.replace('ndcg@', '')}:{metrics[k]:.4f}" for k in ndcg_keys]
        lines.append(f"{prefix}NDCG@k: {' | '.join(ndcg_values)}")

    # MRR
    if "mrr" in metrics:
        lines.append(f"{prefix}MRR: {metrics['mrr']:.4f}")

    def _render_group(group_prefix: str, label: str):
        group_lines = []
        group_recall = [k for k in metrics if k.startswith(f"{group_prefix}recall@")]
        if group_recall:
            group_recall = sorted(group_recall, key=lambda x: int(x.split('@')[1]))
            vals = [f"{k.split('@',1)[1]}:{metrics[k]:.4f}" for k in group_recall]
            group_lines.append(f"{prefix}{label} Recall@k: {' | '.join(vals)}")
        group_cov = [k for k in metrics if k.startswith(f"{group_prefix}coverage@")]
        if group_cov:
            group_cov = sorted(group_cov, key=lambda x: int(x.split('@')[1]))
            vals = [f"{k.split('@',1)[1]}:{metrics[k]:.4f}" for k in group_cov]
            group_lines.append(f"{prefix}{label} Coverage@k: {' | '.join(vals)}")
        group_ndcg = [k for k in metrics if k.startswith(f"{group_prefix}ndcg@")]
        if group_ndcg:
            group_ndcg = sorted(group_ndcg, key=lambda x: int(x.split('@')[1]))
            vals = [f"{k.split('@',1)[1]}:{metrics[k]:.4f}" for k in group_ndcg]
            group_lines.append(f"{prefix}{label} NDCG@k: {' | '.join(vals)}")
        if f"{group_prefix}mrr" in metrics:
            group_lines.append(f"{prefix}{label} MRR: {metrics[f'{group_prefix}mrr']:.4f}")
        if group_lines:
            lines.append("")
            lines.extend(group_lines)

    _render_group("fixed_", "Fixed")
    _render_group("dynamic_", "Dynamic")

    return "\n".join(lines).strip()


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any CUDA operations
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass

    args = parse_args()

    if args.v19_align:
        if args.premise_filter_config:
            print("[Config] v19-align enabled: ignoring --premise-filter-config", flush=True)
        args.premise_filter_config = ""
        if args.proof_pattern_mode not in ("none", "id_only"):
            print("[Config] v19-align enabled: forcing --proof-pattern-mode=none", flush=True)
        args.proof_pattern_mode = "none"
        args.proof_pattern_embedding = None
        if args.split_mode != "ratio":
            print("[Config] v19-align enabled: forcing --split-mode=ratio", flush=True)
        args.split_mode = "ratio"

    # Normalize and validate subgraph mode aliases early so all downstream components share the same value.
    args.subgraph_mode = canonicalize_subgraph_mode(args.subgraph_mode)
    if subgraph_mode_disables_selector(args.subgraph_mode) and args.select_top_k is not None:
        print(
            f"[Config] subgraph_mode={args.subgraph_mode} disables learned top-k; overriding --select-top-k to None",
            flush=True,
        )
        args.select_top_k = None

    if args.fixed_retrieval_negatives:
        # Enforce the ablation contract: all negatives must come from the fixed checkpoint retrieval.
        args.random_negatives_per_query = 0
        args.hard_negatives_per_query = 0
        args.global_negatives_per_query = 0
        args.memory_bank_per_query = 0
        args.cross_batch_negs = 0
        # Keep memory bank directory unset so we don't log/build it.
        args.memory_bank_dir = None

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # CRITICAL: Initialize distributed process group FIRST before any other operations
    # This must happen before torch.cuda.set_device() and loading models
    ddp_timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MINUTES", "30")))
    if distributed:
        print(f"[Rank {local_rank}] Initializing distributed process group...", flush=True)
        if args.ddp_backend == "nccl" and torch.cuda.is_available():
            # For NCCL, we need to set device before init_process_group
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=args.ddp_backend,
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=ddp_timeout,
            )
        else:
            dist.init_process_group(backend=args.ddp_backend, timeout=ddp_timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}] Process group initialized, world_size={world_size}", flush=True)
    else:
        rank = 0
        world_size = 1

    barrier_timeout = os.environ.get("CACHE_BARRIER_TIMEOUT_MINUTES") or os.environ.get("DDP_TIMEOUT_MINUTES", "30")
    barrier_group, barrier_backend = _get_barrier_group(distributed, timeout_minutes=barrier_timeout)

    # Now set device after process group is initialized
    if torch.cuda.is_available():
        if not distributed:  # Only set if not already set above
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    text_device = torch.device(args.text_encoder_device) if args.text_encoder_device else device
    is_main_process = (rank == 0)

    process_seed = int(args.seed) + int(rank) * 100003
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    embedding_suffix = f"_{args.embedding}"
    out_dir = Path(args.out_dir)

    # Create output directory and write metadata on rank 0
    if is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[output] Directory: {out_dir}; checkpoints are saved directly inside it.", flush=True)
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "args": vars(args),
            "world_size": world_size,
            "embedding_model": args.embedding,
        }
        (out_dir / "run_args.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.premise_filter_config:
            cfg_path = Path(args.premise_filter_config)
            if cfg_path.exists():
                (out_dir / "premise_filter_config.json").write_text(
                    cfg_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            else:
                print(
                    f"[premise_filter] Config not found at {cfg_path}; nothing copied to out_dir.",
                    flush=True,
                )

    # CRITICAL: Barrier before loading graph to prevent file I/O conflicts
    if distributed:
        print(f"[Rank {rank}] Waiting at pre-graph barrier...", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] Passed pre-graph barrier", flush=True)

    print(f"[Rank {rank}] Loading graph...", flush=True)
    print(f"[Rank {rank}] Creating MathlibGraphConfig...", flush=True)
    graph_cfg_kwargs = {"graph_path": args.graph_path}
    cfg_fields = getattr(MathlibGraphConfig, "__dataclass_fields__", {})
    if "premise_filter_config" in cfg_fields:
        graph_cfg_kwargs["premise_filter_config"] = args.premise_filter_config or None
    graph_cfg = MathlibGraphConfig(**graph_cfg_kwargs)
    if args.v19_align:
        if "edge_type_filter" in cfg_fields:
            graph_cfg = replace(graph_cfg, edge_type_filter=True)
        # Align premise types with v19 defaults (broader derived set).
        v19_premise_types = set()
        for base in (
            "theorem",
            "lemma",
            "definition",
            "instance",
            "structure",
            "class",
            "inductive",
            "abbrev",
            "opaque",
        ):
            v19_premise_types.add(base)
            v19_premise_types.add(f"{base}_derived")
        v19_premise_types.add("inductive_component_derived")
        graph_cfg = replace(graph_cfg, premise_types=v19_premise_types)
        print(f"[Config] v19-align enabled: premise_types={len(v19_premise_types)}", flush=True)
    print(f"[Rank {rank}] MathlibGraphConfig created", flush=True)
    neg_cfg = graph_cfg.negative_sampling
    updated = False
    if args.memory_bank_per_query is not None:
        neg_cfg = replace(neg_cfg, memory_bank_per_query=max(0, args.memory_bank_per_query))
        updated = True
    if args.random_negatives_per_query is not None:
        neg_cfg = replace(neg_cfg, random_per_query=max(0, args.random_negatives_per_query))
        updated = True
    if args.random_hard_negatives_per_query is not None:
        neg_cfg = replace(neg_cfg, random_hard_per_query=max(0, args.random_hard_negatives_per_query))
        updated = True
    if args.hard_negatives_per_query is not None:
        neg_cfg = replace(neg_cfg, hard_per_query=max(0, args.hard_negatives_per_query))
        updated = True
    if args.global_negatives_per_query is not None:
        neg_cfg = replace(neg_cfg, global_per_query=max(0, args.global_negatives_per_query))
        updated = True
    split_cfg = graph_cfg.split
    split_updated = False
    if args.split_mode and args.split_mode != split_cfg.mode:
        split_cfg = replace(split_cfg, mode=args.split_mode)
        split_updated = True
    if args.val_count and args.split_mode == "fixed_holdout":
        split_cfg = replace(split_cfg, val_count=args.val_count)
        split_updated = True
    if args.test_count and args.split_mode == "fixed_holdout":
        split_cfg = replace(split_cfg, test_count=args.test_count)
        split_updated = True
    if args.holdout_per_split and args.split_mode == "fixed_holdout":
        split_cfg = replace(split_cfg, holdout_per_split=args.holdout_per_split)
        split_updated = True
    if updated or split_updated:
        graph_cfg = replace(graph_cfg, negative_sampling=neg_cfg if updated else graph_cfg.negative_sampling, split=split_cfg)
    print(f"[Rank {rank}] Calling MathlibGraph() constructor...", flush=True)
    sys.stderr.write(f"[Rank {rank}] About to call MathlibGraph()\n")
    sys.stderr.flush()
    graph = MathlibGraph(graph_cfg)
    sys.stderr.write(f"[Rank {rank}] MathlibGraph() returned\n")
    sys.stderr.flush()
    print(f"[Rank {rank}] Graph loaded successfully", flush=True)

    external_splits = None
    if args.dataset_mode in ("combined", "mixed"):
        if not args.combined_jsonl_path:
            raise ValueError("combined-jsonl-path is required when dataset-mode is combined or mixed.")
        if is_main_process:
            print(f"[Rank {rank}] Loading external JSONL from {args.combined_jsonl_path}...")
        external_splits = load_external_jsonl(
            args.combined_jsonl_path,
            graph,
            seed=args.seed,
            val_count=args.combined_val_count,
            test_count=args.combined_test_count,
        )
        if is_main_process:
            print(
                f"[Rank {rank}] External dataset sizes -> "
                f"train={len(external_splits['train'])}, "
                f"val={len(external_splits['val'])}, "
                f"test={len(external_splits['test'])}"
            )

    # Create text encoder with specified embedding model and max_length
    # All ranks load together (working version approach)
    print(f"[Rank {rank}] Loading text encoder ({args.embedding})...", flush=True)
    text_encoder_cfg = TextEncoderConfig(
        embedding_model=args.embedding,
        max_length=args.max_length
    )
    text_encoder = LeanTextEncoder(text_encoder_cfg, device=text_device)
    print(f"[Rank {rank}] Text encoder loaded successfully", flush=True)

    pattern_encoder = None
    if args.proof_pattern_mode not in ("none", "id_only"):
        pattern_model = args.proof_pattern_embedding or args.embedding
        if pattern_model == args.embedding:
            pattern_encoder = text_encoder
        else:
            if is_main_process:
                print(f"[Rank {rank}] Loading proof pattern encoder ({pattern_model})...", flush=True)
            pattern_cfg = TextEncoderConfig(
                embedding_model=pattern_model,
                max_length=args.proof_pattern_max_length,
            )
            pattern_encoder = LeanTextEncoder(pattern_cfg, device=text_device)
            if is_main_process:
                print(f"[Rank {rank}] Proof pattern encoder loaded successfully", flush=True)

    print(f"[Rank {rank}] Creating feature cache...", flush=True)
    feature_cache = NodeFeatureCache(
        graph,
        text_encoder,
        cache_device="cpu",
        sub_node_max_length=int(args.sub_node_max_length),
        pattern_mode=args.proof_pattern_mode,
        pattern_encoder=pattern_encoder,
        pattern_max_length=int(args.proof_pattern_max_length),
    )
    print(f"[Rank {rank}] Feature cache created", flush=True)

    # Feature cache handling - use working version's approach (no barrier!)
    if args.feature_cache_dir:
        base_cache_dir = Path(args.feature_cache_dir)
        # If feature_cache_dir already has embedding suffix, don't add again
        if not str(base_cache_dir).endswith(embedding_suffix):
            feature_cache_dir = Path(str(base_cache_dir) + embedding_suffix)
        else:
            feature_cache_dir = base_cache_dir

        if is_main_process:
            print(f"[Rank {rank}] Preparing feature cache in {feature_cache_dir}...")
            feature_cache.ensure_disk_cache(
                feature_cache_dir,
                batch_size=args.precompute_batch_size,
                force=args.force_precompute_features,
            )
        if distributed and not is_main_process:
            _wait_for_feature_cache(feature_cache, feature_cache_dir)
            feature_cache.ensure_disk_cache(
                feature_cache_dir,
                batch_size=args.precompute_batch_size,
                force=False,
            )

    # External query embedding cache (no graph node append).
    external_query_embeddings: Dict[str, torch.Tensor] = {}
    if args.dataset_mode in ("combined", "mixed"):
        assert external_splits is not None
        cache_root = Path(args.feature_cache_dir) if args.feature_cache_dir else out_dir
        cache_root = cache_root / "external_query_cache"
        if is_main_process:
            cache_root.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier(group=barrier_group)

        digest_src = (
            f"{args.combined_jsonl_path}|seed={args.seed}|val={args.combined_val_count}|test={args.combined_test_count}"
            f"|emb={args.embedding}|len={args.max_length}"
        )
        digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]

        def _cache_path(split: str) -> Path:
            return cache_root / f"combined_queries_{split}_{args.embedding}_len{args.max_length}_{digest}.pt"

        def _encode_texts(texts: List[str], *, batch_size: int) -> torch.Tensor:
            chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for start in tqdm(range(0, len(texts), batch_size), desc="encode_external_queries"):
                    chunk = texts[start : start + batch_size]
                    emb = text_encoder(chunk, max_length=args.max_length).detach().cpu()
                    chunks.append(emb)
            return (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.empty((0, text_encoder.embedding_dim), dtype=torch.float32)
            )

        encode_bs = max(1, min(64, args.precompute_batch_size))
        for split_name in ("train", "val", "test"):
            path = _cache_path(split_name)
            expected = len(external_splits[split_name])
            if is_main_process:
                need_build = True
                if path.exists():
                    try:
                        cached = load_torch_file(path, map_location="cpu", allow_mmap=True, description=str(path))
                        if isinstance(cached, torch.Tensor) and cached.size(0) == expected:
                            external_query_embeddings[split_name] = cached
                            need_build = False
                    except Exception:
                        need_build = True
                if need_build:
                    texts = [t for (_uid, t, _pos) in external_splits[split_name]]
                    enc = _encode_texts(texts, batch_size=encode_bs)
                    if enc.size(0) != expected:
                        raise RuntimeError(f"External embedding size mismatch: {split_name} got {enc.size(0)} expected {expected}")
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    torch.save(enc, tmp, _use_new_zipfile_serialization=True)
                    tmp.replace(path)
                    external_query_embeddings[split_name] = enc
            if distributed and not is_main_process:
                start = time.time()
                while not path.exists():
                    if time.time() - start > 3600:
                        raise TimeoutError(f"Timed out waiting for external query cache: {path}")
                    time.sleep(2)
                external_query_embeddings[split_name] = load_torch_file(
                    path, map_location="cpu", allow_mmap=True, description=str(path)
                )

    print(f"[Rank {rank}] Creating subgraph config (mode={args.subgraph_mode})...", flush=True)
    subgraph_cfg = SubgraphConfig(
        max_nodes=args.max_nodes,
        subgraph_mode=args.subgraph_mode,
        hop1_budget=args.subgraph_hop1_budget,
        hop2_budget=args.subgraph_hop2_budget,
    )
    print(f"[Rank {local_rank}] Subgraph config created")
    if subgraph_cfg.dummy_edges or subgraph_cfg.dummy_degrees or subgraph_cfg.dummy_hops:
        print(
            f"[Rank {local_rank}] Subgraph dummy flags: edges={subgraph_cfg.dummy_edges} "
            f"degrees={subgraph_cfg.dummy_degrees} hops={subgraph_cfg.dummy_hops}",
            flush=True,
        )
    if args.subgraph_hop1_budget or args.subgraph_hop2_budget:
        print(
            f"[Rank {local_rank}] Subgraph hop budgets: hop1={args.subgraph_hop1_budget}, hop2={args.subgraph_hop2_budget}",
            flush=True,
        )

    # Initialize subgraph cache if enabled
    subgraph_cache_path = None
    if args.subgraph_cache_dir:
        print(f"[Rank {local_rank}] Initializing subgraph cache...")
        subgraph_cache_dir = Path(args.subgraph_cache_dir)
        subgraph_builder_rank = 0

        print(f"[Rank {local_rank}] Getting cache file path...")
        cache_file_path = SubgraphCache(graph, subgraph_cfg, cache_device="cpu")._cache_file(subgraph_cache_dir)
        # Sanity check: cached max_nodes must match current args.max_nodes
        match = _CACHE_FILENAME_PATTERN.search(cache_file_path.name)
        if match:
            cached_n = int(match.group(1))
            if cached_n != args.max_nodes:
                raise RuntimeError(
                    f"Subgraph cache {cache_file_path} was built with max_nodes={cached_n}, "
                    f"but you set --max-nodes={args.max_nodes}. Please rebuild the cache or align the flag."
                )
        print(f"[Rank {local_rank}] Cache file path: {cache_file_path}")

        if distributed:
            if rank == subgraph_builder_rank:
                print(f"[Rank {rank}] Preparing subgraph cache in {subgraph_cache_dir} ...")
                subgraph_cache = SubgraphCache(graph, subgraph_cfg, cache_device="cpu")
                subgraph_cache.ensure_cache(
                    subgraph_cache_dir,
                    force=args.force_precompute_subgraphs,
                )
            # Single global builder; synchronize all ranks after cache is ready.
            print(f"[Rank {rank}] Waiting at subgraph-cache barrier...", flush=True)
            dist.barrier(group=barrier_group)
            print(f"[Rank {rank}] Passed subgraph-cache barrier", flush=True)
            subgraph_cache_path = str(cache_file_path)
            if rank == subgraph_builder_rank:
                print("Subgraph cache ready!")
        else:
            subgraph_cache = SubgraphCache(graph, subgraph_cfg, cache_device="cpu")
            subgraph_cache.ensure_cache(
                subgraph_cache_dir,
                force=args.force_precompute_subgraphs,
            )
            subgraph_cache_path = str(cache_file_path)

    # Preload subgraph cache into the shared map so DataLoader workers (forked)
    # can reuse the same mmap'ed object instead of each reloading from disk.
    if subgraph_cache_path:
        try:
            import premise_selector.subgraph as subgraph_module

            # Only load if not already present (e.g., builder rank may have created it).
            cache_path_obj = Path(subgraph_cache_path)
            if cache_path_obj not in subgraph_module._SHARED_SUBGRAPH_CACHE:
                warm_cache = SubgraphCache(graph, subgraph_cfg, cache_device="cpu")
                warm_cache._load_from_disk(cache_path_obj, verbose=is_main_process)
                subgraph_module._SHARED_SUBGRAPH_CACHE[cache_path_obj] = warm_cache
                if is_main_process:
                    print(f"[Rank {local_rank}] Preloaded subgraph cache into shared map for workers.")
        except Exception as exc:  # pragma: no cover
            if is_main_process:
                print(f"[Rank {local_rank}] Warning: failed to preload subgraph cache: {exc}")

    print(f"[Rank {local_rank}] Creating SubgraphBuilder...")
    # Pass cache path instead of cache object to avoid multiprocessing issues
    builder = SubgraphBuilder(graph, feature_cache, subgraph_cfg, subgraph_cache=None, subgraph_cache_path=subgraph_cache_path)
    print(f"[Rank {local_rank}] SubgraphBuilder created")

    # Barrier after cache preparation - all ranks must be ready before proceeding
    if distributed:
        print(
            f"[Rank {rank}] Waiting at post-cache-prep barrier (backend={barrier_backend}, timeout={barrier_timeout} min)...",
            flush=True,
        )
        try:
            dist.barrier(group=barrier_group)
        except Exception as exc:
            print(f"[Rank {rank}] Barrier failed: {exc}")
            raise
        print(f"[Rank {rank}] All ranks ready, proceeding to training...", flush=True)

    # NOTE: Process group was already initialized at the beginning of main()
    # rank, world_size, and is_main_process are already set
    # NOTE: out_dir was already created and metadata written earlier

    print(f"[Rank {rank}] Loading metrics history...", flush=True)
    metrics_path = out_dir / "metrics.json"
    metrics_history: List[Dict[str, float]] = []
    if is_main_process and metrics_path.exists():
        try:
            metrics_history = json.loads(metrics_path.read_text())
        except Exception:
            metrics_history = []
    print(f"[Rank {rank}] Metrics history loaded", flush=True)

    print(f"[Rank {rank}] Creating WandbLogger...", flush=True)
    default_wandb_name = args.wandb_run_name or f"{Path(args.out_dir).name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = (
        WandbLogger(
            enabled=is_main_process and args.wandb_project is not None,
            project=args.wandb_project,
            run_name=default_wandb_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        if args.wandb_project
        else None
    )
    print(f"[Rank {rank}] WandbLogger created", flush=True)

    print(f"[Rank {rank}] Creating ResourceMonitor...", flush=True)
    resource_monitor = (
        ResourceMonitor(
            interval_s=args.resource_monitor_interval,
            device_ids=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [],
        )
        if is_main_process and args.resource_monitor_interval > 0
        else None
    )
    print(f"[Rank {rank}] ResourceMonitor created", flush=True)

    print(f"[Rank {rank}] Creating datasets (mode={args.dataset_mode})...", flush=True)
    datasets: Dict[str, torch.utils.data.Dataset] = {}

    def _make_mathlib(split: str, fixed_eval: bool, max_q: int | None):
        effective_fixed_eval = fixed_eval and (not args.fixed_retrieval_negatives)
        return PremiseSelectionDataset(
            graph,
            split,
            max_queries=max_q or None,
            fixed_eval=effective_fixed_eval,
            dynamic_fraction=args.eval_dynamic_fraction if effective_fixed_eval else 0.0,
            log_progress=is_main_process if effective_fixed_eval else False,
            repeat_factor=(args.train_repeat if split == "train" else 1),
        )

    if args.dataset_mode == "mathlib":
        datasets = {
            "train": _make_mathlib("train", False, args.max_train_queries),
            "val": _make_mathlib("val", True, args.max_val_queries),
            "test": _make_mathlib("test", True, args.max_test_queries),
        }
    elif args.dataset_mode == "combined":
        if external_splits is None:
            raise RuntimeError("External splits not loaded.")
        datasets = {
            "train": ExternalTextPremiseDataset(external_splits["train"], query_embeddings=external_query_embeddings.get("train")),
            "val": ExternalTextPremiseDataset(external_splits["val"], query_embeddings=external_query_embeddings.get("val")),
            "test": ExternalTextPremiseDataset(external_splits["test"], query_embeddings=external_query_embeddings.get("test")),
        }
    else:  # mixed
        if external_splits is None:
            raise RuntimeError("External splits not loaded.")
        math_train = _make_mathlib("train", False, args.max_train_queries)
        math_val = _make_mathlib("val", True, args.max_val_queries)
        math_test = _make_mathlib("test", True, args.max_test_queries)
        ext_train = ExternalTextPremiseDataset(external_splits["train"], query_embeddings=external_query_embeddings.get("train"))
        ext_val = ExternalTextPremiseDataset(external_splits["val"], query_embeddings=external_query_embeddings.get("val"))
        ext_test = ExternalTextPremiseDataset(external_splits["test"], query_embeddings=external_query_embeddings.get("test"))
        datasets = {
            "train": ConcatDataset([math_train, ext_train]),
            "val": ConcatDataset([math_val, ext_val]),
            "test": ConcatDataset([math_test, ext_test]),
        }
    if is_main_process:
        print(
            f"Dataset sizes -> train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
        )

    # ------------------------------------------------------------------
    # Optional ablation: fixed retrieval negatives mined from a checkpoint.
    # ------------------------------------------------------------------
    if args.fixed_retrieval_negatives:
        cache_dir = Path(args.fixed_retrieval_cache_dir) if args.fixed_retrieval_cache_dir else (out_dir / "fixed_retrieval_negatives")
        if is_main_process:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier(group=barrier_group)

        build_device = (
            torch.device(args.fixed_retrieval_build_device)
            if args.fixed_retrieval_build_device
            else device
        )
        top_k = int(args.fixed_retrieval_topk)
        chunk_size = max(1, int(args.fixed_retrieval_chunk_size))

        use_mathlib_fixed = args.dataset_mode in ("mathlib", "mixed")
        use_combined_fixed = args.dataset_mode in ("combined", "mixed")

        if use_mathlib_fixed and (not args.mathlib_fixed_retrieval_checkpoint or not args.mathlib_fixed_premise_embeddings):
            raise ValueError(
                "--mathlib-fixed-retrieval-checkpoint and --mathlib-fixed-premise-embeddings are required "
                "when --fixed-retrieval-negatives is enabled and dataset-mode includes mathlib."
            )
        if use_combined_fixed and (not args.combined_fixed_retrieval_checkpoint or not args.combined_fixed_premise_embeddings):
            raise ValueError(
                "--combined-fixed-retrieval-checkpoint and --combined-fixed-premise-embeddings are required "
                "when --fixed-retrieval-negatives is enabled and dataset-mode includes combined."
            )

        projector_mathlib: FrozenQueryProjector | None = None
        projector_combined: FrozenQueryProjector | None = None

        if is_main_process and use_mathlib_fixed:
            projector_mathlib = FrozenQueryProjector.from_checkpoint(
                args.mathlib_fixed_retrieval_checkpoint,
                input_dim=int(feature_cache.statement_dim),
                device=build_device,
            )

        if is_main_process and use_combined_fixed:
            input_dim = int(text_encoder.embedding_dim)
            if external_query_embeddings:
                any_split = next(iter(external_query_embeddings.values()))
                if isinstance(any_split, torch.Tensor) and any_split.numel() > 0:
                    input_dim = int(any_split.size(-1))
            projector_combined = FrozenQueryProjector.from_checkpoint(
                args.combined_fixed_retrieval_checkpoint,
                input_dim=input_dim,
                device=build_device,
            )

        def _wait_for_file(path: Path, *, timeout_s: int = 24 * 3600) -> None:
            start = time.time()
            while not path.exists():
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timed out waiting for fixed negatives cache: {path}")
                time.sleep(5)

        def _project_graph_queries(qids: Sequence[int], projector: FrozenQueryProjector) -> torch.Tensor:
            # Project statement embeddings (feature_cache full_node) through checkpoint query head.
            bs = max(256, min(8192, int(args.precompute_batch_size or 512)))
            chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for start in tqdm(range(0, len(qids), bs), desc="project_mathlib_queries", disable=not is_main_process):
                    batch_ids = qids[start : start + bs]
                    feats = feature_cache.get_full_nodes(batch_ids, device=build_device)
                    proj = projector(feats).detach().cpu()
                    chunks.append(proj)
            return torch.cat(chunks, dim=0) if chunks else torch.empty((0, int(args.proj_dim)), dtype=torch.float32)

        def _project_external_queries(features: torch.Tensor, projector: FrozenQueryProjector) -> torch.Tensor:
            bs = max(256, min(8192, int(args.precompute_batch_size or 512)))
            chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for start in tqdm(range(0, int(features.size(0)), bs), desc="project_combined_queries", disable=not is_main_process):
                    feat = features[start : start + bs].to(build_device)
                    proj = projector(feat).detach().cpu()
                    chunks.append(proj)
            return torch.cat(chunks, dim=0) if chunks else torch.empty((0, int(args.proj_dim)), dtype=torch.float32)

        # Build/load stores per split, then wrap datasets (including inside ConcatDataset).
        stores_mathlib: Dict[str, CandidateStore] = {}
        stores_combined: Dict[str, CandidateStore] = {}

        for split_name in ("train", "val", "test"):
            split_ds = datasets[split_name]
            # Split ConcatDataset so we can wrap constituents with their respective stores.
            parts = split_ds.datasets if isinstance(split_ds, ConcatDataset) else [split_ds]
            math_part = next((d for d in parts if isinstance(d, PremiseSelectionDataset)), None)
            ext_part = next((d for d in parts if isinstance(d, ExternalTextPremiseDataset)), None)

            if math_part is not None and use_mathlib_fixed:
                cache_path = default_cache_path(
                    cache_dir,
                    tag="mathlib",
                    split_name=split_name,
                    ckpt_path=str(Path(args.mathlib_fixed_retrieval_checkpoint)),
                    premise_embeddings_path=str(Path(args.mathlib_fixed_premise_embeddings)),
                    top_k=top_k,
                    chunk_size=chunk_size,
                )
                if is_main_process:
                    qids = list(getattr(math_part, "query_ids", []))
                    assert projector_mathlib is not None
                    projected = _project_graph_queries(qids, projector_mathlib)
                    build_or_load_candidate_store(
                        tag="mathlib",
                        split_name=split_name,
                        cache_dir=cache_dir,
                        ckpt_path=args.mathlib_fixed_retrieval_checkpoint,
                        premise_embeddings_path=args.mathlib_fixed_premise_embeddings,
                        query_ids=qids,
                        projected_query_embs=projected,
                        positives=graph.positives,
                        top_k=top_k,
                        chunk_size=chunk_size,
                        build_device=build_device,
                        force_rebuild=args.fixed_retrieval_force_rebuild,
                    )
                if distributed:
                    dist.barrier(group=barrier_group)
                if not is_main_process:
                    _wait_for_file(cache_path)
                store, _meta = CandidateStore.load(cache_path)
                stores_mathlib[split_name] = store

            if ext_part is not None and use_combined_fixed:
                cache_path = default_cache_path(
                    cache_dir,
                    tag="combined",
                    split_name=split_name,
                    ckpt_path=str(Path(args.combined_fixed_retrieval_checkpoint)),
                    premise_embeddings_path=str(Path(args.combined_fixed_premise_embeddings)),
                    top_k=top_k,
                    chunk_size=chunk_size,
                )
                if is_main_process:
                    examples = list(getattr(ext_part, "examples", []))
                    uids = [int(uid) for (uid, _text, _pos) in examples]
                    pos_map: Dict[int, set[int]] = {int(uid): set(int(p) for p in pos) for (uid, _t, pos) in examples}
                    feats = external_query_embeddings.get(split_name)
                    if feats is None:
                        raise RuntimeError(f"Missing external_query_embeddings for split={split_name}")
                    assert projector_combined is not None
                    projected = _project_external_queries(feats, projector_combined)
                    build_or_load_candidate_store(
                        tag="combined",
                        split_name=split_name,
                        cache_dir=cache_dir,
                        ckpt_path=args.combined_fixed_retrieval_checkpoint,
                        premise_embeddings_path=args.combined_fixed_premise_embeddings,
                        query_ids=uids,
                        projected_query_embs=projected,
                        positives=pos_map,
                        top_k=top_k,
                        chunk_size=chunk_size,
                        build_device=build_device,
                        force_rebuild=args.fixed_retrieval_force_rebuild,
                    )
                if distributed:
                    dist.barrier(group=barrier_group)
                if not is_main_process:
                    _wait_for_file(cache_path)
                store, _meta = CandidateStore.load(cache_path)
                stores_combined[split_name] = store

        def _wrap_dataset(ds: torch.utils.data.Dataset, split_name: str) -> torch.utils.data.Dataset:
            if isinstance(ds, PremiseSelectionDataset) and split_name in stores_mathlib:
                return FixedNegativeStoreDataset(ds, store=stores_mathlib[split_name], strict=args.fixed_retrieval_strict)
            if isinstance(ds, ExternalTextPremiseDataset) and split_name in stores_combined:
                return FixedNegativeStoreDataset(ds, store=stores_combined[split_name], strict=args.fixed_retrieval_strict)
            if isinstance(ds, ConcatDataset):
                return ConcatDataset([_wrap_dataset(part, split_name) for part in ds.datasets])
            return ds

        for split_name in ("train", "val", "test"):
            datasets[split_name] = _wrap_dataset(datasets[split_name], split_name)

    # Synchronize all ranks before printing dataset information.
    if distributed:
        dist.barrier()

    # Print dataset information.
    if is_main_process:
        print("\n" + "="*80)
        print("Dataset information")
        print("="*80)
        for split_name, dataset in datasets.items():
            print(f"{split_name.upper()} Split: {len(dataset)} queries")
        print("="*80 + "\n")

    collate_fn_seed = int(args.seed) + int(rank) * 100003
    collate_fn = CollateFunction(
        builder,
        feature_cache,
        seed=collate_fn_seed,
        max_candidates=args.max_candidates_per_batch,
        expand_per_positive=(args.loss_type == "loss5"),
        graph=graph if args.loss_type == "loss5" else None,
        loss5_neg_source=args.loss5_neg_source if args.loss_type == "loss5" else None,
        loss5_neg_count=args.loss5_neg_count if args.loss_type == "loss5" else 0,
        loss5_pair_batch_size=args.loss5_pair_batch_size if args.loss_type == "loss5" else None,
    )

    def _worker_init_fn(worker_id: int) -> None:
        # Keep collate-side sampling reproducible across runs and ranks.
        worker_seed = collate_fn_seed + int(worker_id)
        collate_fn.reseed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(process_seed)
    train_sampler = None
    train_batch_sampler = None
    if args.loss_type == "loss5" and isinstance(datasets["train"], PremiseSelectionDataset):
        pair_cap = args.loss5_pair_batch_size
        base_cap = args.max_candidates_per_batch if args.max_candidates_per_batch else 1500
        if pair_cap is None or pair_cap <= 0:
            pair_cap = max(1, base_cap // max(1, 1 + args.loss5_neg_count))
        slices, slice_lengths = _build_loss5_slices(datasets["train"], pair_cap)
        datasets["train"] = Loss5SlicedDataset(datasets["train"], slices)
        train_batch_sampler = Loss5PairBatchSampler(
            slice_lengths,
            pair_cap=pair_cap,
            neg_per_query=args.loss5_neg_count,
            world_size=world_size if distributed else 1,
            rank=rank if distributed else 0,
            seed=args.seed,
            shuffle=True,
        )
        if is_main_process:
            print(f"[Loss5] pair_cap={pair_cap}, slices={len(slices)}, batches per rank={len(train_batch_sampler)}")
    else:
        train_sampler = (
            DistributedSampler(
                datasets["train"],
                shuffle=True,
                drop_last=True,  # CRITICAL: Must be True to ensure all ranks process same number of batches
                seed=args.seed,
            )
            if distributed
            else None
        )
    val_sampler = (
        DistributedSampler(
            datasets["val"],
            shuffle=False,
            drop_last=False,  # Keep all validation samples; DistributedSampler pads to equal per-rank length.
            seed=args.seed,
        )
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            datasets["test"],
            shuffle=False,
            drop_last=False,  # Keep all test samples; DistributedSampler pads to equal per-rank length.
            seed=args.seed,
        )
        if distributed
        else None
    )

    loader_common = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
        "worker_init_fn": _worker_init_fn if args.num_workers > 0 else None,
        "generator": loader_generator,
    }
    if args.prefetch_factor and args.num_workers > 0:
        loader_common["prefetch_factor"] = args.prefetch_factor

    eval_batch_size = args.eval_batch_size if args.eval_batch_size else max(1, args.batch_size // 2)
    if args.eval_num_workers is not None:
        eval_num_workers = args.eval_num_workers
    else:
        eval_num_workers = max(0, args.num_workers // 2)
    eval_common = {
        "batch_size": eval_batch_size,
        "num_workers": eval_num_workers,
        "collate_fn": collate_fn,
        "pin_memory": args.eval_pin_memory,
        "persistent_workers": args.persistent_workers and eval_num_workers > 0,
        "worker_init_fn": _worker_init_fn if eval_num_workers > 0 else None,
        "generator": loader_generator,
    }
    if args.eval_prefetch_factor and eval_num_workers > 0:
        eval_common["prefetch_factor"] = args.eval_prefetch_factor

    if is_main_process:
        print("Creating DataLoaders...")
    train_loader_kwargs = loader_common.copy()
    if train_batch_sampler is not None:
        train_loader_kwargs["batch_sampler"] = train_batch_sampler
    else:
        train_loader_kwargs["batch_size"] = args.batch_size
        train_loader_kwargs["sampler"] = train_sampler
        train_loader_kwargs["shuffle"] = (train_sampler is None)

    loaders = {
        "train": DataLoader(
            datasets["train"],
            **train_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            sampler=val_sampler,
            shuffle=False,
            **eval_common,
        ),
        "test": DataLoader(
            datasets["test"],
            sampler=test_sampler,
            shuffle=False,
            **eval_common,
        ),
    }
    if is_main_process:
        print("DataLoaders created successfully")

    # Synchronize after DataLoader creation (worker process spawning can be slow)
    if distributed:
        print(f"[Rank {rank}] Waiting at DataLoader barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Passed DataLoader barrier")

    if is_main_process:
        print("Creating model...")
    encoder = RelGTPremiseEncoder(
        num_nodes=len(graph.nodes),
        max_neighbor_hop=subgraph_cfg.max_hop,
        node_type_map=graph.type_to_idx,
        feature_dim=feature_cache.feature_dim,
        channels=args.hidden_dim,
        local_num_layers=args.local_layers,
        global_dim=args.global_dim,
        heads=args.heads,
        ff_dropout=args.dropout,
        attn_dropout=args.dropout,
        sample_node_len=args.max_nodes,
        select_top_k=args.select_top_k,
    )
    model = PremiseSelectorModel(
        encoder,
        query_dim=feature_cache.statement_dim,
        proj_dim=args.proj_dim,
    ).to(device)
    if is_main_process:
        print("Model created and moved to device")

    # Synchronize before DDP wrapping
    if distributed:
        print(f"[Rank {rank}] Waiting at model barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Passed model barrier, wrapping with DDP...")
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,  # Allow optional paths (e.g., selector/degree features) without crashing
        )
        print(f"[Rank {rank}] DDP wrapping completed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Initialize AMP scaler if enabled
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    if args.use_amp and is_main_process:
        print("Automatic Mixed Precision (AMP) enabled")

    memory_bank_manager = (
        MemoryBankManager(
            graph,
            log_dir=args.memory_bank_dir,
            world_size=world_size,
            rank=rank,
            top_k=args.memory_bank_topk,
            chunk_size=args.memory_bank_chunk_size,
            build_device=args.memory_bank_build_device,
            distributed=distributed,
            negatives_per_query=graph.cfg.negative_sampling.memory_bank_per_query,
        )
        if args.memory_bank_dir
        else None
    )

    cross_batch_memory = (
        CrossBatchMemory(
            capacity=args.cross_batch_queue,
            sample_size=args.cross_batch_negs,
            device=device,
        )
        if args.cross_batch_negs > 0
        else None
    )

    loss_fn = build_loss_fn(
        args.loss_type,
        args.loss_reduction,
        diversity_weight=args.diversity_weight,
        diversity_temperature=args.diversity_temperature,
        loss_scale=args.loss_scale,
    )

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] | None = None
    start_epoch = 1
    global_update_step = 0
    if args.resume_path:
        resume_path = Path(args.resume_path)
        if resume_path.is_dir():
            candidates = sorted(
                resume_path.glob("checkpoint_epoch_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoint_epoch_*.pt files found in {resume_path}")
            resume_path = candidates[-1]
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {resume_path}")
        ckpt = load_torch_file(resume_path, map_location=device)
        model_state = ckpt.get("model_state")
        optimizer_state = ckpt.get("optimizer_state")
        if model_state is None or optimizer_state is None:
            raise ValueError(f"Checkpoint {resume_path} missing model/optimizer state.")

        # Resize hop encoder embeddings if max_hop differs.
        hop_key = "premise_encoder.hop_encoder.embedding.weight"
        target_model = model.module if isinstance(model, DDP) else model
        target_state = target_model.state_dict()
        if hop_key in model_state and hop_key in target_state:
            src_w = model_state[hop_key]
            tgt_w = target_state[hop_key]
            if src_w.shape != tgt_w.shape:
                min_rows = min(src_w.size(0), tgt_w.size(0))
                resized = torch.zeros_like(tgt_w)
                resized[:min_rows] = src_w[:min_rows]
                if tgt_w.size(0) > src_w.size(0) and min_rows > 0:
                    resized[min_rows:] = src_w[min_rows - 1 : min_rows].expand(tgt_w.size(0) - min_rows, -1)
                model_state[hop_key] = resized
                if is_main_process:
                    print(
                        f"[Resume] Resized hop_encoder from {tuple(src_w.shape)} to {tuple(tgt_w.shape)} to load checkpoint.",
                        flush=True,
                    )

        target_model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(optimizer_state)
        epoch_value = ckpt.get("epoch", 0)
        try:
            epoch_int = int(epoch_value)
        except (TypeError, ValueError):
            epoch_int = 0
        start_epoch = epoch_int + 1
        global_update_step = int(ckpt.get("global_update_step", 0) or 0)
        best_metrics = ckpt.get("metrics")
        if best_metrics is not None:
            best_val_loss = best_metrics.get("val_loss", float("inf"))
        if is_main_process:
            print(f"Resuming from checkpoint {resume_path} (epoch {epoch_int}), continuing with epoch {start_epoch}")

    # LR schedule setup (per optimizer update step)
    import math
    steps_per_epoch = int(math.ceil(len(loaders["train"]) / max(1, args.grad_accum_steps)))
    total_steps = int(max(1, steps_per_epoch * args.epochs))
    warmup_steps = int(args.warmup_steps) if args.warmup_steps and args.warmup_steps > 0 else int(total_steps * float(args.warmup_ratio))
    warmup_steps = max(0, min(warmup_steps, total_steps))
    lr_schedule = {
        "scheduler": args.lr_scheduler,
        "base_lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
    }
    if is_main_process:
        print(
            f"LR schedule: {args.lr_scheduler} base_lr={args.lr} min_lr={args.min_lr} "
            f"warmup_steps={warmup_steps} total_steps={total_steps} (steps_per_epoch={steps_per_epoch})"
        )

    def _extract_epoch(path: Path, prefix: str) -> Optional[int]:
        stem = path.stem
        if not stem.startswith(prefix):
            return None
        try:
            return int(stem.split(prefix, 1)[1])
        except (ValueError, IndexError):
            return None

    def _should_keep_epoch_file(epoch_val: int, latest: int) -> bool:
        return epoch_val == latest or (epoch_val % 5 == 0)

    def prune_epoch_files(directory: Path, pattern: str, prefix: str, latest_epoch: int) -> None:
        """Keep latest + multiples of 5; drop other epoch-tagged files."""
        if not is_main_process or not directory.exists():
            return
        for path in directory.glob(pattern):
            epoch_val = _extract_epoch(path, prefix)
            if epoch_val is None:
                continue
            if _should_keep_epoch_file(epoch_val, latest_epoch):
                continue
            path.unlink(missing_ok=True)

    def save_checkpoint(epoch_label: int | str, metrics: Dict[str, float], filename: str):
        if not is_main_process:
            return
        model_to_save = model.module if isinstance(model, DDP) else model
        checkpoint = {
            "epoch": epoch_label,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_update_step": global_update_step,
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / filename)

    if start_epoch > args.epochs:
        if is_main_process:
            print(f"Start epoch {start_epoch} exceeds total epochs {args.epochs}; skipping training loop.")
    for epoch in range(start_epoch, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            print(f"Epoch {epoch}")
        update_state = {"step": global_update_step}
        train_metrics = run_epoch(
            loaders["train"],
            model,
            optimizer,
            train=True,
            loss_fn=loss_fn,
            lr_schedule=lr_schedule,
            update_state=update_state,
            distributed=distributed,
            log_progress=is_main_process,
            memory_bank_manager=memory_bank_manager,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_amp=args.use_amp,
            cross_batch_memory=cross_batch_memory,
            wandb_logger=wandb_logger,
            resource_monitor=resource_monitor,
            phase="train",
            epoch_index=epoch,
        )
        global_update_step = int(update_state.get("step", global_update_step))
        val_metrics = run_epoch(
            loaders["val"],
            model,
            optimizer,
            train=False,
            loss_fn=loss_fn,
            distributed=distributed,
            log_progress=is_main_process,
            scaler=scaler,
            use_amp=args.use_amp,
            cross_batch_memory=None,
            wandb_logger=wandb_logger,
            resource_monitor=resource_monitor,
            phase="val",
            epoch_index=epoch,
        )
        if is_main_process:
            print("\n" + "="*80)
            print(f"Epoch {epoch} Results")
            print("="*80)
            print("\nTRAIN:")
            print(format_metrics_display(train_metrics, prefix="  "))
            print("\nVALIDATION:")
            print(format_metrics_display(val_metrics, prefix="  "))
            print("="*80 + "\n")

        epoch_record = {"epoch": epoch}
        epoch_record.update({f"train_{k}": v for k, v in train_metrics.items()})
        epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
        if is_main_process:
            metrics_history.append(epoch_record)
            if wandb_logger is not None:
                log_payload = {"epoch": epoch}
                log_payload.update({f"train/{k}": v for k, v in train_metrics.items()})
                log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                wandb_logger.log(log_payload)

        save_checkpoint(epoch, epoch_record, f"checkpoint_epoch_{epoch}.pt")
        prune_epoch_files(out_dir, "checkpoint_epoch_*.pt", "checkpoint_epoch_", epoch)

        val_loss = val_metrics.get("loss", float("inf"))
        if is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = epoch_record
            save_checkpoint(epoch, epoch_record, "best.pt")

        if memory_bank_manager and memory_bank_manager.enabled():
            memory_bank_manager.finalize_epoch(epoch)
            if is_main_process and memory_bank_manager.log_dir:
                prune_epoch_files(
                    Path(memory_bank_manager.log_dir),
                    "memory_bank_epoch*.pt",
                    "memory_bank_epoch",
                    epoch,
                )

    test_metrics = run_epoch(
        loaders["test"],
        model,
        optimizer,
        train=False,
        loss_fn=loss_fn,
        distributed=distributed,
        log_progress=is_main_process,
        scaler=scaler,
        use_amp=args.use_amp,
        cross_batch_memory=None,
        wandb_logger=wandb_logger,
        resource_monitor=resource_monitor,
        phase="test",
        epoch_index=args.epochs + 1,
    )
    if is_main_process:
        print("\n" + "="*80)
        print("Final Test Results")
        print("="*80)
        print("\nTEST:")
        print(format_metrics_display(test_metrics, prefix="  "))
        print("="*80 + "\n")
    test_record = {"epoch": "test_final"}
    test_record.update({f"test_{k}": v for k, v in test_metrics.items()})
    if is_main_process:
        metrics_history.append(test_record)
        if wandb_logger is not None:
            wandb_logger.log({f"test/{k}": v for k, v in test_metrics.items()})

        metrics_path = out_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics_history, fp, indent=2)

        if best_metrics is not None:
            print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_metrics['epoch']})")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
