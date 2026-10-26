#!/usr/bin/env python3
"""Minimal helpers to load DAPS checkpoints for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from premise_selector.data import MathlibGraph
from premise_selector.model import PremiseSelectorModel, RelGTPremiseEncoder
from premise_selector.subgraph import SubgraphConfig


def torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Dict:
    """Load checkpoint tensors and metadata onto the requested device."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(
    ckpt: Dict,
    graph: MathlibGraph,
    device: torch.device,
) -> PremiseSelectorModel:
    state_dict = ckpt["model_state"]
    args = ckpt.get("args", {})

    hidden_dim = args.get("hidden_dim", 512)
    proj_dim = args.get("proj_dim", 256)
    local_layers = args.get("local_layers", 3)
    heads = args.get("heads", 8)
    global_dim = args.get("global_dim") or hidden_dim
    dropout = args.get("dropout", 0.1)
    max_nodes = args.get("max_nodes", 64)
    select_top_k = args.get("select_top_k")
    selector_dropout = args.get("selector_dropout", 0.1)
    conv_type = args.get("conv_type", "full")
    num_centroids = args.get("num_centroids", 4096)

    feature_dim = state_dict["premise_encoder.feature_proj.weight"].shape[1]
    query_dim = state_dict["query_encoder.net.0.weight"].shape[1]

    encoder = RelGTPremiseEncoder(
        num_nodes=len(graph.nodes),
        max_neighbor_hop=SubgraphConfig().max_hop,
        node_type_map=graph.type_to_idx,
        feature_dim=feature_dim,
        channels=hidden_dim,
        local_num_layers=local_layers,
        global_dim=global_dim,
        heads=heads,
        ff_dropout=dropout,
        attn_dropout=dropout,
        conv_type=conv_type,
        num_centroids=num_centroids,
        sample_node_len=max_nodes,
        select_top_k=select_top_k,
        selector_dropout=selector_dropout,
    )

    model = PremiseSelectorModel(
        encoder,
        query_dim=query_dim,
        proj_dim=proj_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
