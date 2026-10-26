from __future__ import annotations

from typing import Set


DEFAULT_QUERY_TYPES: Set[str] = {"theorem", "theorem_derived", "lemma", "lemma_derived"}

DEFAULT_PREMISE_TYPES: Set[str] = {
    "theorem",
    "theorem_derived",
    "lemma",
    "lemma_derived",
    "definition",
    "definition_derived",
    "instance",
    "instance_derived",
    "structure",
    "structure_derived",
    "class",
    "class_derived",
    "inductive",
    "inductive_derived",
    "abbrev",
    "abbrev_derived",
    "opaque",
    "opaque_derived",
    "inductive_component_derived",
}

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
