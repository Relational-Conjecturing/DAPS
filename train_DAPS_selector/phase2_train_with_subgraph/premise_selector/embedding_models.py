"""Embedding model configurations and registry (minimal, phase2 setting)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import os
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PHASE2_DIR = _THIS_FILE.parents[1]
_DEFAULT_PHASE1_TEXT_ENCODER_PATH = (
    _PHASE2_DIR.parent
    / "phase1_train_text_encoder"
    / "runs"
    / "Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc"
    / "final"
).resolve()


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str  # Short name for command line
    model_path: str  # HuggingFace model path
    max_length: int = 512
    use_t5: bool = True  # Whether to use T5EncoderModel (True) or AutoModel (False)
    pooling_method: str = "mean"  # "mean", "last_token", or "cls"
    normalize: bool = False  # Whether to apply L2 normalization
    padding_side: str = "right"  # "left" or "right" padding
    use_flash_attention: bool = False  # Whether to use flash_attention_2


# Registry of available embedding models
EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    "alibaba-gte-modernbert-ft": EmbeddingModelConfig(
        name="alibaba-gte-modernbert-ft",
        model_path=os.environ.get(
            "ALIBABA_GTE_MODERNBERT_FT_PATH",
            str(_DEFAULT_PHASE1_TEXT_ENCODER_PATH),
        ),
        max_length=512,
        use_t5=False,
        pooling_method="mean",
        normalize=True,
        padding_side="right",
        use_flash_attention=False,
    ),
}


def get_embedding_model_config(model_name: str) -> EmbeddingModelConfig:
    """Get embedding model configuration by name.

    Args:
        model_name: Short name of the model (e.g., "reprover", "bge-base")

    Returns:
        EmbeddingModelConfig for the requested model

    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in EMBEDDING_MODELS:
        available = ", ".join(EMBEDDING_MODELS.keys())
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Available models: {available}"
        )
    return EMBEDDING_MODELS[model_name]


def list_embedding_models() -> list[str]:
    """List all available embedding model names."""
    return list(EMBEDDING_MODELS.keys())
