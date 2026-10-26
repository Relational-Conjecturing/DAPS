#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline for constructing the lake graph with Lean code snippets.

Steps
-----
0. (Precondition) Lean environment is available via `lake`, and
   `LeanScripts/ExtractConstants.lean` is compilable.
1. Extract `lake_constants.json` for every Mathlib module, mirroring the
   `generated_nodes/mathlib...` layout with only the `.lean` source and the JSON.
2. Invoke `find_code/extract_lake_code.py` to enrich constants with Lean code
   snippets, which also renames the signature field.
3. Create one additional `file` node per Lean file, whose dependencies come
   from import statements.
4. Build the graph (nodes + edges) and save it under
   `generated_graphs/<version>/lake_graph_with_code_<version>.json`.
5. Produce statistics over node / edge types for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import time

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from lake_extractor import LakeExtractor, ExtractionResult, file_path_to_module_name  # type: ignore
    from import_parser import extract_imports_from_text  # type: ignore
    from toolkit_skiplist import SKIP_CONSTANTS  # type: ignore
else:
    from .lake_extractor import LakeExtractor, ExtractionResult, file_path_to_module_name
    from .import_parser import extract_imports_from_text
    from .toolkit_skiplist import SKIP_CONSTANTS


LOGGER = logging.getLogger("lake_graph_pipeline")
REPO_ROOT = Path(__file__).resolve().parents[1]
FIND_CODE_DIR = REPO_ROOT / "find_code"


def relative_to_repo(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        try:
            return p.resolve().relative_to(REPO_ROOT).as_posix()
        except Exception:  # noqa: BLE001
            return os.path.relpath(str(p), REPO_ROOT)


def normalize_cli_args(args: List[str]) -> List[str]:
    normalized: List[str] = []
    for token in args:
        if token.startswith("/"):
            normalized.append(relative_to_repo(token))
        else:
            normalized.append(token)
    return normalized


@dataclass
class PipelineConfig:
    mathlib_source: Path
    output_root: Path
    lake_project_dir: Path
    lean_scripts_dir: Path
    graph_output: Path
    stats_output: Path
    version: str
    limit: Optional[int]
    timeout: int
    workers: int
    skip_extract: bool
    skip_snippet: bool
    skip_graph: bool


def discover_lean_files(mathlib_source: Path, limit: Optional[int] = None) -> List[Path]:
    files = [p for p in sorted(mathlib_source.rglob("*.lean")) if p.is_file()]
    if limit is not None:
        files = files[:limit]
    LOGGER.info("Discovered %d Lean files under %s", len(files), mathlib_source)
    return files


def module_folder_for_file(output_root: Path, mathlib_source: Path, lean_path: Path) -> Path:
    rel = lean_path.relative_to(mathlib_source)
    return output_root / "Mathlib" / rel


def ensure_module_folder(module_folder: Path) -> None:
    if module_folder.exists() and module_folder.is_file():
        module_folder.unlink()
    module_folder.parent.mkdir(parents=True, exist_ok=True)
    module_folder.mkdir(exist_ok=True)


def write_constants_json(module_folder: Path, constants: List[Dict[str, object]]) -> None:
    dst = module_folder / "lake_constants.json"
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(constants, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def copy_lean_source(src: Path, dst_folder: Path) -> None:
    if not src.is_file():
        LOGGER.debug("Skipping non-file Lean path: %s", src)
        return
    dst = dst_folder / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = src.read_bytes()
    if dst.exists() and dst.is_file():
        try:
            if dst.read_bytes() == content:
                return
        except OSError:
            pass
    dst.write_bytes(content)


def run_extract_lake_code(mathlib_root: Path) -> None:
    script = FIND_CODE_DIR / "extract_lake_code.py"
    cmd = [sys.executable, str(script), "--mathlib-root", str(mathlib_root)]
    LOGGER.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def module_folder_to_module_name(module_folder: Path, mathlib_root: Path) -> str:
    rel = module_folder.relative_to(mathlib_root)
    if not rel.parts:
        raise ValueError(f"Cannot derive module name from {module_folder}")
    parts = list(rel.parts)
    last = parts[-1]
    if last.endswith(".lean"):
        parts[-1] = last[:-5]
    return "Mathlib." + ".".join(parts)


def module_name_to_path(module_name: str) -> Path:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("Module name is empty")
    rel_parts = parts
    if parts[0] == "Mathlib":
        rel_parts = parts[1:]
    if not rel_parts:
        raise ValueError(f"Unexpected module name {module_name}")
    if len(rel_parts) == 1:
        return Path(f"{rel_parts[0]}.lean")
    return Path(*rel_parts[:-1]) / f"{rel_parts[-1]}.lean"


def build_source_path(mathlib_source: Path, module_name: str) -> str:
    root = mathlib_source.parent.name
    rel_path = module_name_to_path(module_name)
    return f"/{root}/Mathlib/{rel_path.as_posix()}"


def load_constants_withcode(
    mathlib_root: Path,
    mathlib_source: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    nodes: List[Dict[str, object]] = []
    name_map: Dict[str, Dict[str, object]] = {}
    dependency_map: Dict[str, List[Dict[str, object]]] = {}
    for json_path in sorted(mathlib_root.rglob("lake_constants_withcode.json")):
        module_folder = json_path.parent
        module_name = module_folder_to_module_name(module_folder, mathlib_root)
        with json_path.open(encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse %s: %s", json_path, exc)
                continue
        for raw_entry in data:
            if raw_entry.get("line", 0) <= 0:
                continue
            entry = dict(raw_entry)
            if "signature" not in entry and "type" in entry:
                entry["signature"] = entry.pop("type")
            entry.setdefault("subcategory", "unknown")
            entry["module"] = module_name
            entry["source_path"] = build_source_path(mathlib_source, module_name)
            entry["in_blacklist"] = entry.get("in_blacklist", False) or entry["name"] in SKIP_CONSTANTS
            deps = entry.pop("dependencies", []) or []
            dependency_map[entry["name"]] = [dict(dep) for dep in deps]
            nodes.append(entry)
            name_map.setdefault(entry["name"], entry)
    LOGGER.info("Loaded %d constants with Lean code", len(nodes))
    return nodes, name_map, dependency_map


def create_file_nodes(
    mathlib_source: Path,
    existing_names: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    file_nodes: List[Dict[str, object]] = []
    dependency_map: Dict[str, List[Dict[str, object]]] = {}
    for lean_file in sorted(mathlib_source.rglob("*.lean")):
        if lean_file.is_dir():
            continue
        module_name = file_path_to_module_name(mathlib_source, lean_file)
        node_name = f"file_{module_name}"
        content = lean_file.read_text(encoding="utf-8")
        line_count = len(content.splitlines())
        imports = extract_imports_from_text(content)
        dependencies = [
            {"name": f"file_{name}", "explicit": True, "direct": True}
            for name in imports
        ]
        node = {
            "name": node_name,
            "category": "file",
            "signature": None,
            "line": 0,
            "column": 0,
            "isProp": False,
            "isHumanTheorem": False,
            "doc": None,
            "lean4_start": 0,
            "lean4_end": line_count,
            "lean4_code": content,
            "subcategory": "file",
            "statement": module_name,
            "proof": None,
            "module": module_name,
            "source_path": build_source_path(mathlib_source, module_name),
            "in_blacklist": False,
        }
        file_nodes.append(node)
        existing_names.setdefault(node_name, node)
        dependency_map[node_name] = [dict(dep) for dep in dependencies]
    LOGGER.info("Created %d file nodes", len(file_nodes))
    return file_nodes, dependency_map


EDGE_PRIMARY_DEP = {"theorem", "lemma", "axiom"}


def sanitize_label(label: str) -> str:
    text = label.upper() if label else "UNKNOWN"
    sanitized = re.sub(r"[^A-Z0-9]", "_", text)
    return sanitized or "UNKNOWN"


def resolve_edge_type(source_guess: Optional[str], target_guess: Optional[str]) -> str:
    if (source_guess or "").lower() == "file":
        return "IMPORTS"
    if (target_guess or "").lower() == "file":
        return "IMPORTS"
    guess = (target_guess or "unknown").lower()
    base = guess[:-len("_derived")] if guess.endswith("_derived") else guess
    if base in EDGE_PRIMARY_DEP:
        return "DEPENDS_ON"
    return f"USES_{sanitize_label(target_guess or 'unknown')}"


def build_edges(
    dependency_map: Dict[str, List[Dict[str, object]]],
    name_lookup: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    skipped_count = 0
    for source_name, deps in dependency_map.items():
        source_node = name_lookup.get(source_name)
        source_guess = source_node.get("subcategory") if source_node else None
        for dep in deps or []:
            target_name = dep.get("name")
            if not target_name:
                continue
            target_node = name_lookup.get(target_name)
            if target_node is not None:
                target_guess = target_node.get("subcategory")
            elif target_name.startswith("file_"):
                target_guess = "file"
            else:
                # Target node does not exist in the node list, skip this edge
                skipped_count += 1
                continue
            edge_type = resolve_edge_type(source_guess, target_guess)
            target_in_blacklist = bool(target_node is not None and target_node.get("in_blacklist", False))
            edges.append(
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "type": edge_type,
                    "explicit": bool(dep.get("explicit")),
                    "direct": bool(dep.get("direct", True)),
                    "in_blacklist": target_in_blacklist,
                }
            )
    LOGGER.info("Built %d edges (skipped %d edges to non-existent nodes)", len(edges), skipped_count)
    return edges


def write_graph(
    graph_output: Path,
    nodes: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    config: PipelineConfig,
    elapsed_seconds: float,
) -> None:
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": config.version,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "mathlib_source": relative_to_repo(config.mathlib_source),
        "output_root": relative_to_repo(config.output_root),
        "lake_project_dir": relative_to_repo(config.lake_project_dir),
        "graph_output": relative_to_repo(graph_output),
        "stats_output": relative_to_repo(config.stats_output),
        "workers": config.workers,
        "skip_extract": config.skip_extract,
        "skip_snippet": config.skip_snippet,
        "skip_graph": config.skip_graph,
        "limit": config.limit,
        "timeout": config.timeout,
        "elapsed_seconds": elapsed_seconds,
    }
    payload = {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }
    with graph_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOGGER.info("Graph written to %s", graph_output)


def write_stats(
    stats_output: Path,
    nodes: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    metadata: Dict[str, object],
) -> None:
    node_counter = Counter(node.get("subcategory", "unknown") for node in nodes)
    edge_counter = Counter(edge.get("type", "UNKNOWN") for edge in edges)
    total_nodes = sum(node_counter.values())
    total_edges = sum(edge_counter.values())
    skip_nodes = sum(1 for node in nodes if node.get("in_blacklist"))
    skip_edges = sum(1 for edge in edges if edge.get("in_blacklist"))
    stats = {
        "node_type_counts": dict(sorted(node_counter.items())),
        "node_type_ratios": {
            key: value / total_nodes if total_nodes else 0.0
            for key, value in sorted(node_counter.items())
        },
        "edge_type_counts": dict(sorted(edge_counter.items())),
        "edge_type_ratios": {
            key: value / total_edges if total_edges else 0.0
            for key, value in sorted(edge_counter.items())
        },
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "in_blacklist_nodes": skip_nodes,
        "in_blacklist_node_ratio": skip_nodes / total_nodes if total_nodes else 0.0,
        "in_blacklist_edges": skip_edges,
        "in_blacklist_edge_ratio": skip_edges / total_edges if total_edges else 0.0,
        "metadata": metadata,
    }
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with stats_output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOGGER.info("Statistics written to %s", stats_output)


def run_extraction_step(
    config: PipelineConfig,
    lean_files: List[Path],
) -> None:
    extractor = LakeExtractor(config.lake_project_dir, config.lean_scripts_dir)
    mathlib_root_output = config.output_root / "Mathlib"
    mathlib_root_output.mkdir(parents=True, exist_ok=True)

    total_files = len(lean_files)

    def process_single(index_path: Tuple[int, Path]) -> Tuple[int, str, int, bool, Optional[str]]:
        idx, lean_file = index_path
        module_folder = module_folder_for_file(config.output_root, config.mathlib_source, lean_file)
        ensure_module_folder(module_folder)
        copy_lean_source(lean_file, module_folder)
        module_name = file_path_to_module_name(config.mathlib_source, lean_file)
        try:
            result: ExtractionResult = extractor.extract_module_constants(
                module_name,
                timeout=config.timeout,
            )
            constants = result.constants
            success = result.success
            error = result.error
        except Exception as exc:  # noqa: BLE001
            constants = []
            success = False
            error = str(exc)
        write_constants_json(module_folder, constants)
        return idx, module_name, len(constants), success, error

    if config.workers <= 1:
        for idx, lean_file in enumerate(lean_files, 1):
            order, module_name, const_count, success, error = process_single((idx - 1, lean_file))
            status = "OK" if success else f"FAIL ({error})"
            LOGGER.info(
                "[%d/%d] %s -> %d constants, %s",
                order + 1,
                total_files,
                module_name,
                const_count,
                status,
            )
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(process_single, (idx, lean_file)): idx
                for idx, lean_file in enumerate(lean_files)
            }
            for future in as_completed(futures):
                order, module_name, const_count, success, error = future.result()
                status = "OK" if success else f"FAIL ({error})"
                LOGGER.info(
                    "[%d/%d] %s -> %d constants, %s",
                    order + 1,
                    total_files,
                    module_name,
                    const_count,
                    status,
                )


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Build lake graph with Lean code snippets.")
    parser.add_argument(
        "--mathlib-source",
        type=Path,
        required=True,
        help="Path to the Mathlib directory (the folder containing all Lean modules).",
    )
    parser.add_argument(
        "--lake-project-dir",
        type=Path,
        default=None,
        help="Directory where `lake` commands are executed (default: parent of mathlib-source).",
    )
    parser.add_argument(
        "--lean-scripts-dir",
        type=Path,
        default=REPO_ROOT / "LeanScripts",
        help="Directory containing ExtractConstants.lean (default: LeanScripts).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Target root directory for generated nodes (mirrors Mathlib structure).",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        required=True,
        help="Path to the final graph JSON file.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        required=True,
        help="Path to the statistics JSON output.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="mathlib4-4.22.0-full",
        help="Version label stored in metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of Lean files processed (useful for debugging).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout (seconds) for each Lean extraction call.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip Step 1 (Lean extraction).")
    parser.add_argument("--skip-snippet", action="store_true", help="Skip Step 2 (snippet enrichment).")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Step 3+ (graph construction).")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers during Lean extraction (default: max(1, cpu_count - 1)).",
    )

    args = parser.parse_args(argv)

    mathlib_source = args.mathlib_source.resolve()
    lake_project_dir = (
        args.lake_project_dir.resolve()
        if args.lake_project_dir is not None
        else mathlib_source.parent
    )
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    workers = args.workers if args.workers is not None else default_workers

    return PipelineConfig(
        mathlib_source=mathlib_source,
        output_root=args.output_root.resolve(),
        lake_project_dir=lake_project_dir,
        lean_scripts_dir=args.lean_scripts_dir.resolve(),
        graph_output=args.graph_output.resolve(),
        stats_output=args.stats_output.resolve(),
        version=args.version,
        limit=args.limit,
        timeout=args.timeout,
        skip_extract=args.skip_extract,
        skip_snippet=args.skip_snippet,
        skip_graph=args.skip_graph,
        workers=max(1, workers),
    )


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    cli_args = normalize_cli_args(raw_args)
    config = parse_args(argv)
    LOGGER.info(
        "Configured with mathlib-source=%s, output-root=%s, workers=%d",
        relative_to_repo(config.mathlib_source),
        relative_to_repo(config.output_root),
        config.workers,
    )

    mathlib_source = config.mathlib_source
    if not mathlib_source.exists():
        raise FileNotFoundError(f"Mathlib source directory not found: {mathlib_source}")

    lean_files = discover_lean_files(mathlib_source, config.limit)

    start_time = time.time()

    if not config.skip_extract:
        run_extraction_step(config, lean_files)

    mathlib_root_output = config.output_root / "Mathlib"

    if not config.skip_snippet:
        run_extract_lake_code(mathlib_root_output)

    if config.skip_graph:
        LOGGER.info("Graph construction step skipped.")
        return

    constant_nodes, name_lookup, constant_dependencies = load_constants_withcode(mathlib_root_output, config.mathlib_source)
    file_nodes, file_dependencies = create_file_nodes(config.mathlib_source, name_lookup)
    dependency_map: Dict[str, List[Dict[str, object]]] = {}
    dependency_map.update(constant_dependencies)
    dependency_map.update(file_dependencies)
    all_nodes = constant_nodes + file_nodes
    edges = build_edges(dependency_map, name_lookup)
    for const in constant_nodes:
        file_name = f"file_{const['module']}"
        edges.append(
            {
                "source_name": file_name,
                "target_name": const["name"],
                "type": "CONTAINS",
                "explicit": True,
                "direct": True,
                "in_blacklist": bool(const.get("in_blacklist", False)),
            }
        )
    elapsed = time.time() - start_time
    run_metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": config.version,
        "mathlib_source": relative_to_repo(config.mathlib_source),
        "output_root": relative_to_repo(config.output_root),
        "lake_project_dir": relative_to_repo(config.lake_project_dir),
        "graph_output": relative_to_repo(config.graph_output),
        "stats_output": relative_to_repo(config.stats_output),
        "workers": config.workers,
        "skip_extract": config.skip_extract,
        "skip_snippet": config.skip_snippet,
        "skip_graph": config.skip_graph,
        "limit": config.limit,
        "timeout": config.timeout,
        "elapsed_seconds": elapsed,
    }
    write_graph(config.graph_output, all_nodes, edges, config, elapsed)
    run_metadata["cli_args"] = cli_args
    write_stats(config.stats_output, all_nodes, edges, run_metadata)


if __name__ == "__main__":
    main()
