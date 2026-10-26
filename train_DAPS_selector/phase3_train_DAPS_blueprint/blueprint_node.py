from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BlueprintNode:
    node_id: int
    full_id: str
    raw_id: str
    repo: str
    kind: str
    proof_status: str
    nlp_text: str
    lean4_code: Optional[str] = None
    lean_decl_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    outgoing: List[int] = field(default_factory=list)
    incoming: List[int] = field(default_factory=list)
    subcategory: str = ""

    def statement_text(self) -> str:
        return self.nlp_text or ""

    def neighbor_text(self) -> str:
        parts = [f"id: {self.raw_id}"]
        if self.nlp_text:
            parts.append(f"nlp: {self.nlp_text}")
        if self.lean_decl_name:
            parts.append(f"lean_decl_name: {self.lean_decl_name}")
        if self.lean4_code:
            parts.append(f"lean4:\n{self.lean4_code}")
        return "\n\n".join(parts)

    def premise_text(self) -> str:
        return self.neighbor_text()
