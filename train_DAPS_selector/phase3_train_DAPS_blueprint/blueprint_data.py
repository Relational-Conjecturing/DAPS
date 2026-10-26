from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from blueprint_node import BlueprintNode


def normalize_proof_status(value: object) -> str:
    if value is None:
        return "nlp_only"
    text = str(value).strip().lower()
    if not text:
        return "nlp_only"
    text = text.replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text == "all formalized":
        return "all_formalized"
    if text == "statement formalized":
        return "statement_formalized"
    if text == "nlp only":
        return "nlp_only"
    if text == "formalized":
        return "all_formalized"
    return text.replace(" ", "_")


class BlueprintGraph:
    def __init__(
        self,
        jsonl_dir: str,
        *,
        log_every: int = 500,
        strict: bool = False,
        shuffle_seed: int | None = 42,
    ):
        self.jsonl_dir = Path(jsonl_dir)
        if not self.jsonl_dir.exists():
            raise FileNotFoundError(f"Blueprint jsonl dir not found: {self.jsonl_dir}")

        self.nodes: List[BlueprintNode] = []
        self.id_by_full: Dict[str, int] = {}
        self.repo_nodes: Dict[str, List[int]] = {}
        self.full_id_by_id: List[str] = []
        self.pad_type = "__pad__"
        self.type_to_idx: Dict[str, int] = {}
        self.query_ids: List[int] = []

        self._load_graph(log_every=log_every, strict=strict)
        if shuffle_seed is not None:
            self._shuffle_node_ids(shuffle_seed)

    def _load_graph(self, *, log_every: int = 500, strict: bool = False) -> None:
        jsonl_paths = sorted(self.jsonl_dir.glob("*.jsonl"))
        if not jsonl_paths:
            raise FileNotFoundError(f"No jsonl files found in {self.jsonl_dir}")

        proof_counter = Counter()
        kind_counter = Counter()
        unknown_proof = 0
        unknown_kind = 0
        dep_counter = Counter()
        total_edges = 0
        missing_deps = 0
        duplicate_full_ids = 0
        for jsonl_path in jsonl_paths:
            repo = jsonl_path.stem
            repo_list: List[int] = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        msg = f"JSON parse error in {jsonl_path}:{line_idx}: {exc}"
                        if strict:
                            raise ValueError(msg) from exc
                        print(f"[BlueprintGraph] {msg}")
                        continue

                    raw_id = str(obj.get("id", ""))
                    if not raw_id:
                        msg = f"Missing id in {jsonl_path}:{line_idx}"
                        if strict:
                            raise ValueError(msg)
                        print(f"[BlueprintGraph] {msg}")
                        continue

                    full_id = f"{repo}/{raw_id}"
                    if full_id in self.id_by_full:
                        duplicate_full_ids += 1
                        continue

                    proof_status = normalize_proof_status(obj.get("proof_status"))
                    kind = str(obj.get("kind", "ellipse"))
                    if proof_status not in ("all_formalized", "statement_formalized", "nlp_only"):
                        unknown_proof += 1
                    if kind not in ("ellipse", "box"):
                        unknown_kind += 1
                    nlp_text = str(obj.get("nlp_text", "") or "")
                    lean4_code = obj.get("lean4_code")
                    lean_decl_name = obj.get("lean_decl_name")
                    dependencies = obj.get("dependencies", []) or []
                    if not isinstance(dependencies, list):
                        dependencies = []

                    subcategory = f"{proof_status}|{kind}"

                    node_id = len(self.nodes)
                    node = BlueprintNode(
                        node_id=node_id,
                        full_id=full_id,
                        raw_id=raw_id,
                        repo=repo,
                        kind=kind,
                        proof_status=proof_status,
                        nlp_text=nlp_text,
                        lean4_code=lean4_code,
                        lean_decl_name=lean_decl_name,
                        dependencies=[str(x) for x in dependencies],
                        outgoing=[],
                        incoming=[],
                        subcategory=subcategory,
                    )

                    self.nodes.append(node)
                    self.id_by_full[full_id] = node_id
                    self.full_id_by_id.append(full_id)
                    repo_list.append(node_id)

                    proof_counter[proof_status] += 1
                    kind_counter[kind] += 1
                    dep_counter[len(dependencies)] += 1

                    if log_every and node_id % log_every == 0 and node_id > 0:
                        print(f"[BlueprintGraph] Loaded {node_id} nodes...")

            self.repo_nodes[repo] = repo_list

        # Build edges after all nodes loaded
        for node in self.nodes:
            seen = set()
            for dep_raw in node.dependencies:
                full_dep = f"{node.repo}/{dep_raw}"
                dep_id = self.id_by_full.get(full_dep)
                if dep_id is None:
                    missing_deps += 1
                    continue
                if dep_id in seen:
                    continue
                seen.add(dep_id)
                node.outgoing.append(dep_id)
                self.nodes[dep_id].incoming.append(node.node_id)
                total_edges += 1

        # Filter out small repos (<=5 nodes)
        min_repo_size = 6
        small_repos = [repo for repo, nodes in self.repo_nodes.items() if len(nodes) <= 5]
        if small_repos:
            small_repo_set = set(small_repos)
            excluded_nodes = set()
            for repo in small_repos:
                excluded_nodes.update(self.repo_nodes[repo])

            # Remove from repo_nodes
            for repo in small_repos:
                del self.repo_nodes[repo]

            print(f"[BlueprintGraph] Filtered {len(small_repos)} repos with <=5 nodes ({len(excluded_nodes)} nodes)")

            # Mark excluded nodes as invalid (we'll skip them in queries)
            for nid in excluded_nodes:
                self.nodes[nid].outgoing = []
                self.nodes[nid].incoming = []

        type_set = sorted({node.subcategory for node in self.nodes})
        type_set.append(self.pad_type)
        self.type_to_idx = {t: i for i, t in enumerate(type_set)}

        self.query_ids = [node.node_id for node in self.nodes if node.outgoing]

        print("[BlueprintGraph] Load summary")
        print(f"  repos: {len(self.repo_nodes)} (after filtering)")
        print(f"  nodes: {len(self.nodes)}")
        print(f"  edges: {total_edges}")
        print(f"  queries (non-empty deps): {len(self.query_ids)}")
        print(f"  missing deps: {missing_deps}")
        if duplicate_full_ids:
            print(f"  duplicate full_id entries skipped: {duplicate_full_ids}")
        print(f"  proof_status counts: {dict(proof_counter)}")
        with_lean = proof_counter.get("all_formalized", 0) + proof_counter.get("statement_formalized", 0)
        nlp_only = proof_counter.get("nlp_only", 0)
        print(f"  proof_status (grouped): {{'with_lean': {with_lean}, 'nlp_only': {nlp_only}}}")
        print(f"  kind counts: {dict(kind_counter)}")
        print(f"  deps per node (sample): {dict(dep_counter.most_common(5))}")
        if unknown_proof or unknown_kind:
            print(f"  unknown proof_status: {unknown_proof} unknown kind: {unknown_kind}")

    def _shuffle_node_ids(self, seed: int) -> None:
        n = len(self.nodes)
        if n == 0:
            return
        rng = random.Random(seed)
        perm = list(range(n))
        rng.shuffle(perm)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(perm)}

        new_nodes: List[BlueprintNode] = [None] * n  # type: ignore[list-item]
        for new_id, old_id in enumerate(perm):
            node = self.nodes[old_id]
            node.node_id = new_id
            new_nodes[new_id] = node

        self.nodes = new_nodes

        for node in self.nodes:
            node.outgoing = [old_to_new[nid] for nid in node.outgoing]
            node.incoming = [old_to_new[nid] for nid in node.incoming]

        for repo, ids in list(self.repo_nodes.items()):
            self.repo_nodes[repo] = [old_to_new[nid] for nid in ids]

        self.id_by_full = {node.full_id: node.node_id for node in self.nodes}
        self.full_id_by_id = [node.full_id for node in self.nodes]
        self.query_ids = [node.node_id for node in self.nodes if node.outgoing]
        print(f"[BlueprintGraph] Shuffled node ids with seed={seed}")

    def get_node(self, node_id: int) -> BlueprintNode:
        return self.nodes[node_id]

    def full_id(self, node_id: int) -> str:
        return self.full_id_by_id[node_id]

    def sample_negatives(
        self,
        query_id: int,
        num_repo_negatives: int,
        num_global_negatives: int,
        *,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        if rng is None:
            rng = random
        query = self.get_node(query_id)
        exclude = set(query.outgoing)
        exclude.add(query_id)

        repo_nodes = self.repo_nodes.get(query.repo, [])
        repo_candidates = [nid for nid in repo_nodes if nid not in exclude]
        repo_k = min(num_repo_negatives, len(repo_candidates))
        repo_sample = rng.sample(repo_candidates, repo_k) if repo_k > 0 else []

        need_extra = max(0, num_repo_negatives - len(repo_sample))
        global_need = num_global_negatives + need_extra

        global_candidates = [
            nid
            for nid in range(len(self.nodes))
            if nid not in exclude and nid not in repo_sample
        ]
        global_k = min(global_need, len(global_candidates))
        global_sample = rng.sample(global_candidates, global_k) if global_k > 0 else []

        return repo_sample + global_sample


class BlueprintQueryDataset(Dataset):
    def __init__(self, query_ids: Sequence[int]):
        self.query_ids = list(query_ids)

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> int:
        return self.query_ids[idx]


def split_queries(
    query_ids: Sequence[int],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    shuffled = list(query_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    return train_ids, val_ids, test_ids


def save_splits(
    graph: BlueprintGraph,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    output_path: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    def to_records(node_ids: Sequence[int]) -> List[dict]:
        records = []
        for nid in node_ids:
            node = graph.get_node(nid)
            records.append(
                {
                    "id": node.raw_id,
                    "repo": node.repo,
                    "full_id": node.full_id,
                }
            )
        return records

    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "train": to_records(train_ids),
        "val": to_records(val_ids),
        "test": to_records(test_ids),
        "stats": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
        },
    }

    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[BlueprintGraph] Splits saved to {output_path}")


def load_splits(
    graph: BlueprintGraph,
    *,
    splits_path: str,
) -> Tuple[List[int], List[int], List[int]]:
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    def to_ids(records: Sequence[dict]) -> List[int]:
        ids = []
        for rec in records:
            full_id = rec.get("full_id") or f"{rec.get('repo')}/{rec.get('id')}"
            if full_id in graph.id_by_full:
                ids.append(graph.id_by_full[full_id])
        return ids

    train_ids = to_ids(data.get("train", []))
    val_ids = to_ids(data.get("val", []))
    test_ids = to_ids(data.get("test", []))
    print(
        f"[BlueprintGraph] Loaded splits from {splits_path}: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


def ensure_splits(
    graph: BlueprintGraph,
    *,
    splits_path: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    path = Path(splits_path)
    if path.exists():
        return load_splits(graph, splits_path=splits_path)

    train_ids, val_ids, test_ids = split_queries(
        graph.query_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    save_splits(
        graph,
        train_ids,
        val_ids,
        test_ids,
        output_path=splits_path,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return train_ids, val_ids, test_ids
