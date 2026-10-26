"""
Utilities for adapting Relational Graph Transformer (RelGT) to Lean premise selection.

This package bundles the mathlib-specific data pipeline, text encoders, and
training utilities that wrap the upstream RelGT backbone.
"""

__all__ = ["config", "data", "text_encoder", "model", "features", "subgraph", "train"]
