from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .constants import DEFAULT_POSITIVE_EDGE_TYPES, DEFAULT_PREMISE_TYPES, DEFAULT_QUERY_TYPES


@dataclass(frozen=True)
class NodeText:
    name: str
    subcategory: str  # e.g., "theorem", "lemma", "definition", etc.
    module: Optional[str]
    source_path: Optional[str]
    line: Optional[int]
    column: Optional[int]
    signature: str
    statement: str  # the proof goal / theorem statement
    lean4_code: str


@dataclass
class MathlibGraphLite:
    nodes: List[NodeText]
    name_to_id: Dict[str, int]
    positives: Dict[int, List[int]]  # query_id -> list[premise_id]
    query_ids: List[int]
    premise_ids: List[int]

    @property
    def source_paths(self) -> List[Optional[str]]:
        return [n.source_path for n in self.nodes]


def _load_with_json(graph_path: Path) -> tuple[list[dict], list[dict]]:
    with graph_path.open("r", encoding="utf-8") as fp:
        graph = json.load(fp)
    return graph.get("nodes") or [], graph.get("edges") or []


def _iter_ijson_items(graph_path: Path, prefix: str) -> Iterable[dict]:
    import ijson  # type: ignore

    with graph_path.open("rb") as fp:
        for obj in ijson.items(fp, prefix):
            yield obj


def load_mathlib_graph_lite(
    graph_path: str | Path,
    *,
    query_types: Set[str] = DEFAULT_QUERY_TYPES,
    premise_types: Set[str] = DEFAULT_PREMISE_TYPES,
    positive_edge_types: Set[str] = DEFAULT_POSITIVE_EDGE_TYPES,
) -> MathlibGraphLite:
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    use_ijson = False
    try:
        import ijson  # type: ignore  # noqa: F401

        use_ijson = True
    except Exception:
        use_ijson = False

    nodes_raw: List[dict] = []
    edges_raw: List[dict] = []
    if use_ijson:
        nodes_raw = list(_iter_ijson_items(graph_path, "nodes.item"))
        edges_raw = list(_iter_ijson_items(graph_path, "edges.item"))
    else:
        nodes_raw, edges_raw = _load_with_json(graph_path)

    nodes: List[NodeText] = []
    name_to_id: Dict[str, int] = {}
    for n in nodes_raw:
        if bool(n.get("in_blacklist", False)) or n.get("subcategory") == "file":
            continue
        # Match Phase2's loaded-node IDs used by fixed splits and embeddings.
        nid = len(nodes)
        name = n.get("name") or ""
        subcategory = n.get("subcategory") or ""
        module = n.get("module")
        source_path = n.get("source_path")
        line = n.get("line")
        column = n.get("column")
        signature = n.get("signature") or ""
        statement = (n.get("statement") or "").strip()
        lean4_code = n.get("lean4_code") or ""
        nodes.append(
            NodeText(
                name=name,
                subcategory=subcategory,
                module=module,
                source_path=source_path,
                line=line,
                column=column,
                signature=signature,
                statement=statement,
                lean4_code=lean4_code,
            )
        )
        if name:
            name_to_id[name] = nid

    positives: Dict[int, List[int]] = {}
    tmp: Dict[int, set[int]] = {}
    for e in edges_raw:
        et = e.get("type")
        if et not in positive_edge_types:
            continue
        src = e.get("source_name")
        tgt = e.get("target_name")
        if not src or not tgt:
            continue
        sid = name_to_id.get(src)
        tid = name_to_id.get(tgt)
        if sid is None or tid is None:
            continue
        tmp.setdefault(sid, set()).add(tid)

    for qid, s in tmp.items():
        positives[qid] = sorted(s)

    query_ids: List[int] = []
    premise_ids: List[int] = []
    for nid, n in enumerate(nodes):
        if n.subcategory in premise_types:
            premise_ids.append(nid)
        if n.subcategory in query_types and n.statement and nid in positives:
            query_ids.append(nid)

    return MathlibGraphLite(
        nodes=nodes,
        name_to_id=name_to_id,
        positives=positives,
        query_ids=query_ids,
        premise_ids=premise_ids,
    )


def build_premise_text(
    node: NodeText,
    *,
    include_lean4_code: bool = True,
    include_statement: bool = False,
    lean4_code_max_chars: int = 0,
) -> str:
    parts: List[str] = []
    parts.append(f"name: {node.name}")
    if node.signature:
        parts.append(f"signature: {node.signature}")
    if include_statement and node.statement:
        parts.append(f"statement: {node.statement}")
    if include_lean4_code and node.lean4_code:
        code = node.lean4_code
        if lean4_code_max_chars and lean4_code_max_chars > 0 and len(code) > lean4_code_max_chars:
            code = code[:lean4_code_max_chars]
        parts.append(f"lean4:\n{code}")
    return "\n\n".join(parts).strip()
