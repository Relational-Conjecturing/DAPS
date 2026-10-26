#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lean import parser for the graph_pipeline pipeline.

Only extracts `import ...` statements from Lean source files.
The implementation is intentionally lightweight yet handles doc comments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def _update_block_depth(line: str, depth: int) -> int:
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if ch == "/" and nxt == "-":
            depth += 1
            i += 2
            continue
        if ch == "-" and nxt == "/" and depth > 0:
            depth -= 1
            i += 2
            continue
        i += 1
    return depth


def extract_imports_from_text(content: str) -> List[str]:
    """
    Return a list of module names imported by the Lean source.

    The parser skips:
      * blank lines
      * single-line comments (`-- ...`)
      * block comments (`/- ... -/`)
    """
    modules: List[str] = []
    depth = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            depth = _update_block_depth(raw_line, depth)
            continue
        if line.startswith("--"):
            depth = _update_block_depth(raw_line, depth)
            continue

        depth = _update_block_depth(raw_line, depth)
        if depth > 0:
            continue

        if not line.startswith("import "):
            continue

        payload = line[len("import ") :].strip()
        if not payload:
            continue

        # Remove inline comment
        if "--" in payload:
            payload = payload.split("--", 1)[0].strip()
        # Remove trailing block comment opener
        if "/-" in payload:
            payload = payload.split("/-", 1)[0].strip()

        if not payload:
            continue

        tokens = payload.split()
        for token in tokens:
            clean = token.strip()
            if not clean:
                continue
            modules.append(clean)

    return modules


def extract_imports_from_file(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as handle:
        return extract_imports_from_text(handle.read())

