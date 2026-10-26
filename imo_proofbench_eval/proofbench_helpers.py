"""Premise prompt formatting used by the paper (matched context, top_k=16)."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple

ANSWER_SYSTEM_PROMPT = (
    "You are an olympiad math solver. Provide clear reasoning and end with a concise final answer "
    "prefixed by 'Answer:'. When you use any context item below, state its name and statement "
    "(if provided) in natural language."
)

def one_line(text: Any) -> str:
    return " ".join(str(text).split())

def extract_statement(premise: Dict[str, Any]) -> Optional[str]:
    for key in ("statement", "lean4_code"):
        val = premise.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

def format_matched_premises(premises: List[Dict[str, Any]], top_k: int) -> str:
    """Format premise_list for matched mode (name/signature/statement)."""
    if not premises:
        return "No premises provided."
    lines: List[str] = []
    for i, p in enumerate(premises[:top_k], 1):
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not isinstance(name, str) or not name:
            continue
        signature = p.get("signature")
        statement = extract_statement(p)
        parts: List[str] = [f"name: {name}"]
        if signature:
            parts.append(f"signature: {one_line(signature)}")
        if statement:
            parts.append(f"statement: {one_line(statement)}")
        else:
            parts.append("statement: <missing>")
        lines.append(f"{i}. " + "\n   ".join(parts))
    if not lines:
        return "No premises provided."
    header = (
        "The following are Lean4 statements from mathlib (name, signature, statement). "
        "You may cite them without proof, but explain their names and statements in natural language."
    )
    return header + "\n" + "\n\n".join(lines)

def build_answer_messages(problem: str, context_text: Optional[str]) -> List[Dict[str, str]]:
    """Build chat messages for answer generation."""
    if context_text:
        user_content = f"{problem}\n\nContext:\n{context_text}"
    else:
        user_content = problem
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["_row_id"] = i
            yield obj


def get_problem_id(item: Dict[str, Any]) -> str:
    for key in ("Problem ID", "problem_id", "id"):
        pid = item.get(key)
        if pid is not None:
            return str(pid)
    return str(item.get("_row_id", ""))


def extract_problem_text(item: Dict[str, Any]) -> str:
    for key in ("Problem", "problem", "prompt"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def load_existing_keys(path: Path) -> set[Tuple[str, str, int]]:
    """Load completed (input_file, problem_id, sample_index) tuples for resume."""
    done: set[Tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("problem_id") or obj.get("_row_id")
            if pid is None:
                continue
            sample_idx = obj.get("sample_index", 0)
            try:
                sample_idx = int(sample_idx)
            except Exception:
                sample_idx = 0
            input_file = obj.get("input_file") or ""
            done.add((str(input_file), str(pid), sample_idx))
    return done


def matched_context(item: Dict[str, Any], top_k: int) -> Tuple[str, List[str], int, str]:
    premise_list = item.get("premise_list") or []
    if not isinstance(premise_list, list):
        premise_list = []
    valid = [p for p in premise_list if isinstance(p, dict) and p.get("name")]
    top_items = valid[:top_k]
    names = [p.get("name") for p in top_items if p.get("name")]
    return format_matched_premises(top_items, top_k), names, len(top_items), "premise_list"
