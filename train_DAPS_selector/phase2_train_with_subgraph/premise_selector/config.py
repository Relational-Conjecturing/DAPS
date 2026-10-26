from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Set


def _with_derivatives(*base_types: str) -> Set[str]:
    """Expand base Lean node types with their `_derived` counterparts."""
    expanded = set()
    for base_type in base_types:
        expanded.add(base_type)
        expanded.add(f"{base_type}_derived")
    return expanded


DEFAULT_QUERY_TYPES: Set[str] = _with_derivatives("theorem", "lemma")

# Keep premises to semantically meaningful mathematical objects.
# Aligned with coverage_checker_v1/premise_filter_config.json (allowed_subcategories).
DEFAULT_PREMISE_TYPES: Set[str] = {
    "theorem",
    "theorem_derived",
    "lemma",
    "lemma_derived",
    "definition",
    "definition_derived",
    "instance",
    "structure",
    "structure_derived",
    "class",
    "inductive",
    "inductive_derived",
    "abbrev",
    "abbrev_derived",
    "opaque",
    "opaque_derived",
}

# Edge types that imply semantic dependencies between the query and a premise.
DEFAULT_POSITIVE_EDGE_TYPES: Set[str] = {
    "DEPENDS_ON",
    "USES_DEFINITION",
    "USES_DEFINITION_DERIVED",
    "USES_INSTANCE",
    "USES_INSTANCE_DERIVED",
    "USES_STRUCTURE",
    "USES_STRUCTURE_DERIVED",
    "USES_CLASS",
    "USES_CLASS_DERIVED",
    "USES_INDUCTIVE",
    "USES_INDUCTIVE_DERIVED",
    "USES_INDUCTIVE_COMPONENT_DERIVED",
    "USES_RECURSOR",
    "USES_RECURSOR_DERIVED",
    "USES_OPAQUE",
    "USES_OPAQUE_DERIVED",
    "USES_ABBREV",
    "USES_ABBREV_DERIVED",
}

# Meta-like node types we never treat as premises.
EXCLUDED_NODE_KINDS: Set[str] = {
    "file",
    "syntax",
    "namespace_command",
    "section_command",
    "set_option",
}


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "ratio"  # "ratio" or "fixed_holdout"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    val_count: int = 2000
    test_count: int = 2000
    holdout_per_split: int = 1000

    def num_splits(self) -> int:
        total = self.train_ratio + self.val_ratio
        if total >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for test split")
        return 3


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Hyper-parameters for simple + hard negative sampling."""

    random_per_query: int = 64
    hard_per_query: int = 32
    random_hard_per_query: int = 0  # import-closure + same-file-above-line candidates
    max_positives: int = 32
    include_same_file: bool = True
    same_file_only_above_line: bool = True
    memory_bank_per_query: int = 0
    global_per_query: int = 32


@dataclass(frozen=True)
class MathlibGraphConfig:
    graph_path: str
    query_types: Set[str] = field(default_factory=lambda: set(DEFAULT_QUERY_TYPES))
    premise_types: Set[str] = field(default_factory=lambda: set(DEFAULT_PREMISE_TYPES))
    positive_edge_types: Set[str] = field(default_factory=lambda: set(DEFAULT_POSITIVE_EDGE_TYPES))
    edge_type_filter: bool = False
    excluded_node_kinds: Set[str] = field(default_factory=lambda: set(EXCLUDED_NODE_KINDS))
    split: SplitConfig = field(default_factory=SplitConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    premise_filter_config: str | None = None
    split_cache_json: str | None = None

    def validate(self) -> None:
        if not self.graph_path:
            raise ValueError("graph_path must be provided")
        if not self.query_types:
            raise ValueError("query_types cannot be empty")
        if not self.premise_types:
            raise ValueError("premise_types cannot be empty")
        if self.edge_type_filter and not self.positive_edge_types:
            raise ValueError("positive_edge_types cannot be empty when edge_type_filter=True")


def normalize_types(types: Iterable[str]) -> Set[str]:
    return {t.strip() for t in types if t and t.strip()}


TYPE_PRIORITY: Sequence[str] = [
    "theorem",
    "theorem_derived",
    "lemma",
    "lemma_derived",
    "definition",
    "definition_derived",
    "instance",
    "instance_derived",
]
