from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

# Disable torchvision fake-operator registration.
import os
os.environ.setdefault("TORCH_LIBRARY_DISABLE_FAKE_IMPL", "1")
# Use the PyTorch backend for Transformers.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .embedding_models import get_embedding_model_config, EmbeddingModelConfig


@dataclass
class TextEncoderConfig:
    model_name: str = "alibaba-gte-modernbert-ft"
    max_length: int = 512  # Can be overridden via command line
    dtype: torch.dtype = torch.float32
    trainable: bool = False
    embedding_model: str = "alibaba-gte-modernbert-ft"  # Short name from embedding_models.py


class LeanTextEncoder(nn.Module):
    """Lightweight wrapper around text encoders (supports T5 and AutoModel-based embeddings)."""

    def __init__(self, cfg: TextEncoderConfig, device: torch.device | str | None = None):
        super().__init__()
        self.cfg = cfg

        # Get embedding model config
        emb_cfg = get_embedding_model_config(cfg.embedding_model)
        self.emb_cfg = emb_cfg

        # Use model path from embedding config
        model_path = emb_cfg.model_path
        # Use max_length from TextEncoderConfig (can be overridden by user)
        self.max_length = cfg.max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set padding side according to model config
        self.tokenizer.padding_side = emb_cfg.padding_side

        # Load encoder (T5 or AutoModel)
        if emb_cfg.use_t5:
            # Lazy import to avoid triggering tensorflow dependency on non-T5 runs.
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(model_path)
        else:
            # Try to use flash_attention_2 if requested
            if emb_cfg.use_flash_attention:
                try:
                    self.encoder = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2"
                    )
                except Exception:
                    # Fallback to default if flash_attention_2 is not available
                    self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            else:
                self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        self.encoder.to(device)
        if not cfg.trainable:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the encoder."""
        if hasattr(self.encoder.config, 'd_model'):
            return self.encoder.config.d_model
        elif hasattr(self.encoder.config, 'hidden_size'):
            return self.encoder.config.hidden_size
        else:
            raise ValueError("Cannot determine embedding dimension from model config")

    def forward(self, texts: Sequence[str], max_length: int | None = None) -> torch.Tensor:
        """Encode a batch of texts.

        Args:
            texts: iterable of strings to encode.
            max_length: optional override for tokenizer max_length. If None, uses cfg.max_length.
        """
        req_len = max_length or self.max_length
        model_max = getattr(self.encoder.config, "max_position_embeddings", None)
        if model_max is not None and req_len > int(model_max):
            print(f"[TextEncoder] max_length {req_len} exceeds model limit {model_max}; clamping.", flush=True)
            req_len = int(model_max)
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=req_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)

        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state  # [B, T, hidden_dim]
        else:
            hidden = outputs[0]  # [B, T, hidden_dim]

        # Apply pooling according to model config
        pooling_method = self.emb_cfg.pooling_method

        if pooling_method == "mean":
            # Mean pooling
            pooled = hidden.mean(dim=1)
        elif pooling_method == "cls":
            # CLS token (first token)
            pooled = hidden[:, 0, :]
        elif pooling_method == "last_token":
            # Last token pooling (accounting for padding)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                # Get the index of the last non-padded token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1  # [B]
                batch_size = hidden.shape[0]
                # Extract the last token for each sequence
                pooled = hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]
            else:
                # Fallback: use the last token position
                pooled = hidden[:, -1, :]
        else:
            # Fallback to mean pooling
            pooled = hidden.mean(dim=1)

        # Apply normalization if requested
        if self.emb_cfg.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.to(dtype=self.cfg.dtype)
