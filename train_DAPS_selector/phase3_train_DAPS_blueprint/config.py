from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


_PHASE3_DIR = Path(__file__).resolve().parent


def _default_data_dir() -> str:
    return str((_PHASE3_DIR / "blueprint_dataset" / "full_collection").resolve())


def _default_summary_csv_path() -> str:
    return str((_PHASE3_DIR / "blueprint_dataset" / "summary.csv").resolve())


def _default_output_root() -> str:
    return str((_PHASE3_DIR / "outputs").resolve())


def _default_splits_path() -> str:
    return str((_PHASE3_DIR / "splits.json").resolve())


def _default_checkpoint_path() -> str:
    return str(
        (
            _PHASE3_DIR.parent
            / "phase2_train_with_subgraph"
            / "runs"
            / "phase2_subgraph"
            / "checkpoint_epoch_80.pt"
        ).resolve()
    )


def _default_text_encoder_path() -> str:
    return str(
        (
            _PHASE3_DIR.parent
            / "phase1_train_text_encoder"
            / "runs"
            / "Alibaba-NLP-gte-modernbert-base-lr4e-4-bs32-nneg3-ne2-wr0.04-mb32-ga2-gc"
            / "final"
        ).resolve()
    )


@dataclass
class TrainConfig:
    data_dir: str = field(default_factory=_default_data_dir)
    summary_csv_path: str = field(default_factory=_default_summary_csv_path)
    output_root: str = field(default_factory=_default_output_root)
    splits_path: str = field(default_factory=_default_splits_path)
    checkpoint_path: str = field(default_factory=_default_checkpoint_path)
    text_encoder_path: str = field(default_factory=_default_text_encoder_path)
    text_encoder_key: str = "blueprint-alibaba-gte-modernbert-ft"
    text_encoder_max_length: int = 1024
    freeze_text_encoder: bool = True

    batch_size: int = 16
    epochs: int = 120
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    weight_decay: float = 0.0

    num_repo_negatives: int = 15
    num_global_negatives: int = 45
    # Sample 2-hop subgraph pool size; MLP selector will pick top-k.
    # 32 neighbors + seed node => 33 nodes total.
    max_subgraph_nodes: int = 33
    subgraph_select_k: int = 33
    subgraph_hop: int = 2

    hidden_dim: int = 512
    proj_dim: int = 256
    local_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    temperature: float = 0.07

    seed: int = 42
    device: str = "auto"
    eval_ks: str = "1,2,4,8,16,32"
    eval_chunk_size: int = 64
    log_every: int = 20

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0

    def make_output_dir(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(self.output_root)
        out_dir = root / f"blueprint_run_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
