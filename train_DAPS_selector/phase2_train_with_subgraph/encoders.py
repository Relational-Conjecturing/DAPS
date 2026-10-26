from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GINConv


class NeighborNodeTypeEncoder(nn.Module):
    """Encode neighbor node types via embeddings."""

    def __init__(self, node_type_map: dict[str, int], embedding_dim: int):
        super().__init__()
        num_types = max(node_type_map.values()) + 1 if node_type_map else 0
        self.embedding = nn.Embedding(num_embeddings=num_types + 1, embedding_dim=embedding_dim)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()

    def forward(self, type_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_indices)


class NeighborHopEncoder(nn.Module):
    """Encode hop distances via embeddings."""

    def __init__(self, max_neighbor_hop: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_neighbor_hop + 2, embedding_dim=embedding_dim)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()

    def forward(self, hop_distances: torch.Tensor) -> torch.Tensor:
        shifted = hop_distances + 1
        return self.embedding(shifted)


class GNNPEEncoder(nn.Module):
    """GNN-based positional encoder used by RelGT."""

    def __init__(self, embedding_dim: int, num_layers: int = 4, pooling: str = "none", pe_dim: int = 0):
        super().__init__()
        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.layer_embedding_dim = embedding_dim // 4
        self.pe_dim = pe_dim

        if self.pe_dim > 0:
            self.input_proj = nn.Linear(self.pe_dim, self.layer_embedding_dim)
        else:
            self.input_proj = nn.Linear(1, self.layer_embedding_dim)

        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.layer_embedding_dim, self.layer_embedding_dim * 2),
                nn.BatchNorm1d(self.layer_embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.layer_embedding_dim * 2, self.layer_embedding_dim),
            )
            self.conv.append(GINConv(mlp, train_eps=True))

        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(self.layer_embedding_dim))

        if self.pooling == "cat":
            final_input_dim = self.layer_embedding_dim * num_layers
        elif self.pooling in ["none", "mean", "max"]:
            final_input_dim = self.layer_embedding_dim
        else:
            raise ValueError("Invalid pooling method. Choose from 'none', 'cat', 'mean', 'max'.")

        self.final_transform = nn.Linear(final_input_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for conv in self.conv:
            for layer in conv.nn:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        nn.init.xavier_uniform_(self.final_transform.weight)
        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index: torch.Tensor, batch: torch.Tensor, precomputed_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = edge_index.device
        total_nodes = batch.size(0)

        if precomputed_pe is not None:
            if self.pe_dim == 0:
                raise ValueError("Received precomputed positional encodings but pe_dim is 0.")
            x_input = precomputed_pe.to(device).view(total_nodes, -1)
        elif self.pe_dim > 0:
            data = Data(edge_index=edge_index, num_nodes=total_nodes)
            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)
            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes, 1, device=device)

        x = self.input_proj(x_input)

        outputs = []
        for i, conv in enumerate(self.conv):
            x_res = x
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            outputs.append(x)

        if self.pooling == "none":
            x_final = outputs[-1]
        elif self.pooling == "cat":
            x_final = torch.cat(outputs, dim=-1)
        elif self.pooling == "mean":
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.mean(outputs_tensor, dim=-1)
        elif self.pooling == "max":
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.max(outputs_tensor, dim=-1)[0]
        else:
            x_final = outputs[-1]

        x = self.final_transform(x_final)

        batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        if batch_size == 0:
            return x.view(0, 0, x.size(-1))
        nodes_per_graph = total_nodes // batch_size
        return x.view(batch_size, nodes_per_graph, -1)
