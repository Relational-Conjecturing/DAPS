from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import (
    GNNPEEncoder,
    NeighborHopEncoder,
    NeighborNodeTypeEncoder,
)
from model import RelGTLayer


class NodeSelector(nn.Module):
    """Scores nodes so the encoder can keep a learned top-k subset.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(4, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> scores [B, N]
        return self.net(x).squeeze(-1)


class RelGTPremiseEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: Dict[str, int],
        feature_dim: int,
        channels: int = 512,
        local_num_layers: int = 3,
        global_dim: int | None = None,
        heads: int = 8,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_type: str = "full",
        num_centroids: int = 4096,
        sample_node_len: int = 64,
        select_top_k: int | None = None,
        selector_dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_proj = nn.Linear(feature_dim, channels)
        self.layer_norm_feature = nn.LayerNorm(channels)
        self.type_encoder = NeighborNodeTypeEncoder(node_type_map=node_type_map, embedding_dim=channels)
        self.hop_encoder = NeighborHopEncoder(max_neighbor_hop=max_neighbor_hop, embedding_dim=channels)
        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)
        self.layer_norm_degree = nn.LayerNorm(channels)
        self.degree_proj = nn.Linear(2, channels)

        self.pe_encoder = GNNPEEncoder(embedding_dim=channels)

        self.in_mixture = nn.Sequential(
            nn.Linear(4 * channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

        self.conv_type = conv_type
        self.output_dim = channels * 2 if conv_type == "full" else channels

        self.select_top_k = max(1, select_top_k) if select_top_k else None
        self.effective_nodes = (
            min(sample_node_len, self.select_top_k) if self.select_top_k else sample_node_len
        )
        self.selector = NodeSelector(channels, dropout=selector_dropout) if self.select_top_k else None

        effective_global_dim = global_dim or channels
        self.global_dim = effective_global_dim
        self.conv = RelGTLayer(
            in_channels=channels,
            out_channels=channels,
            local_num_layers=local_num_layers,
            global_dim=effective_global_dim,
            num_nodes=num_nodes,
            heads=heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            conv_type=conv_type,
            num_centroids=num_centroids,
            sample_node_len=self.effective_nodes,
        )

    def reset_parameters(self):
        self.feature_proj.reset_parameters()
        self.layer_norm_feature.reset_parameters()
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.layer_norm_type.reset_parameters()
        self.layer_norm_hop.reset_parameters()
        self.layer_norm_pe.reset_parameters()
        self.layer_norm_degree.reset_parameters()
        self.degree_proj.reset_parameters()
        for layer in self.in_mixture:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.conv.reset_parameters()

    def forward(
        self,
        neighbor_types: torch.Tensor,
        neighbor_hops: torch.Tensor,
        neighbor_features: torch.Tensor,
        in_degrees: torch.Tensor,
        out_degrees: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        feature_emb = self.layer_norm_feature(self.feature_proj(neighbor_features))
        type_emb = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
        hop_emb = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
        pe_emb = self.layer_norm_pe(self.pe_encoder(edge_index, batch))
        deg_raw = torch.stack([in_degrees, out_degrees], dim=-1).float()
        deg_emb = self.layer_norm_degree(self.degree_proj(torch.log1p(deg_raw)))
        seed_type_idx = neighbor_types[:, 0].long()
        seed_ctx = self.layer_norm_type(self.type_encoder(seed_type_idx)).unsqueeze(1).expand_as(type_emb)

        x_set = torch.cat([feature_emb, type_emb, hop_emb, pe_emb], dim=-1)
        x_set = self.in_mixture(x_set)
        x_set = x_set + deg_emb + seed_ctx
        if self.selector is not None and x_set.size(1) > self.effective_nodes:
            k_neighbors = self.effective_nodes - 1
            if k_neighbors == 0:
                x_set = x_set[:, :1, :]
            else:
                scores = self.selector(x_set)
                topk_scores, topk_idx = torch.topk(scores[:, 1:], k=k_neighbors, dim=1)
                neighbors = self._gather_topk(x_set, topk_idx + 1, target_len=k_neighbors)
                neighbors = neighbors * torch.sigmoid(topk_scores).unsqueeze(-1)
                x_set = torch.cat([x_set[:, :1, :], neighbors], dim=1)
        elif self.selector is not None and x_set.size(1) < self.effective_nodes:
            # Pad if the sampled subgraph is smaller than desired
            pad_len = self.effective_nodes - x_set.size(1)
            pad = torch.zeros(x_set.size(0), pad_len, x_set.size(2), device=x_set.device, dtype=x_set.dtype)
            x_set = torch.cat([x_set, pad], dim=1)
        x_set = self.conv(x_set, x_set[:, 0, :], node_indices)
        if x_set.dim() == 2:
            x_set = x_set.unsqueeze(1)
        seed_repr = x_set[:, 0, :]
        return seed_repr

    @staticmethod
    def _gather_topk(x: torch.Tensor, idx: torch.Tensor, target_len: int) -> torch.Tensor:
        """Gather top-k nodes per sample and pad to target_len if needed."""
        # x: [B, N, D], idx: [B, K]
        b, _, d = x.shape
        k = idx.size(1)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
        gathered = x.gather(1, idx_exp)
        if k == target_len:
            return gathered
        pad_len = target_len - k
        pad = torch.zeros(b, pad_len, d, device=x.device, dtype=x.dtype)
        return torch.cat([gathered, pad], dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PremiseSelectorModel(nn.Module):
    def __init__(
        self,
        encoder: RelGTPremiseEncoder,
        query_dim: int,
        proj_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.premise_encoder = encoder
        hidden_dim = encoder.output_dim
        self.query_encoder = QueryEncoder(query_dim, hidden_dim)
        self.query_head = ProjectionHead(hidden_dim, proj_dim)
        self.premise_head = ProjectionHead(hidden_dim, proj_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def load_state_dict(self, state_dict, strict: bool = True):
        hop_key = "premise_encoder.hop_encoder.embedding.weight"
        if hop_key in state_dict:
            tgt = self.premise_encoder.hop_encoder.embedding.weight
            src = state_dict[hop_key]
            if src.shape != tgt.shape:
                resized = torch.zeros_like(tgt)
                min_rows = min(src.size(0), tgt.size(0))
                if min_rows > 0:
                    resized[:min_rows] = src[:min_rows]
                    if tgt.size(0) > src.size(0):
                        resized[min_rows:] = src[min_rows - 1 : min_rows].expand(tgt.size(0) - min_rows, -1)
                state_dict = dict(state_dict)
                state_dict[hop_key] = resized
        return super().load_state_dict(state_dict, strict=strict)

    def encode_queries(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        encoded = self.query_encoder(query_embeddings)
        return self.query_head(encoded)

    def encode_premises(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        seed_repr = self.premise_encoder(
            batch_inputs["neighbor_types"],
            batch_inputs["neighbor_hops"],
            batch_inputs["neighbor_features"],
            batch_inputs["in_degrees"],
            batch_inputs["out_degrees"],
            edge_index=batch_inputs["edge_index"],
            batch=batch_inputs["batch"],
            node_indices=batch_inputs["node_indices"],
        )
        return self.premise_head(seed_repr)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        premise_inputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.encode_queries(query_embeddings)
        p = self.encode_premises(premise_inputs)
        logits = self.compute_logits(q, p)
        return q, p, logits

    def compute_logits(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (q @ p.T) / self.temperature
