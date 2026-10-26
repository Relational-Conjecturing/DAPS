from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch_file(
    path: str | Path,
    *,
    map_location: Any = None,
    allow_mmap: bool = False,
    description: str | None = None,
) -> Any:
    """Load tensor files with checkpoint metadata and optional memory mapping."""
    kwargs = {"map_location": map_location, "weights_only": False}
    if allow_mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        # Retry with the arguments supported by the installed PyTorch version.
        fallback_kwargs = {"map_location": map_location}
        if allow_mmap:
            fallback_kwargs["mmap"] = True
        try:
            return torch.load(path, **fallback_kwargs)
        except TypeError:
            return torch.load(path, map_location=map_location)
    except RuntimeError as exc:
        if allow_mmap and "mmap can only be used" in str(exc):
            target = description or path
            print(f"mmap load failed for {target}, retrying without mmap...")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise
