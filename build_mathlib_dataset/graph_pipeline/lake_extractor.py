#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Lake extractor used by the graph_pipeline pipeline.

This module provides a thin wrapper for invoking the Lean metaprogram
`LeanScripts/ExtractConstants.lean` via `lake env lean --run`.
It mirrors the essential behaviour of `python/lake_extractor.py`
but keeps the dependencies self-contained for the new pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional


LOGGER = logging.getLogger(__name__)


def file_path_to_module_name(mathlib_root: Path, file_path: Path) -> str:
    """
    Convert an absolute Lean file path to its corresponding module name.

    Parameters
    ----------
    mathlib_root:
        Path to the `Mathlib` directory (the folder that contains all Lean modules).
    file_path:
        Absolute path to a `.lean` file under `mathlib_root`.
    """
    rel_path = file_path.resolve().relative_to(mathlib_root.resolve())
    module = rel_path.with_suffix("")
    return "Mathlib." + ".".join(module.parts)


@dataclass
class ExtractionResult:
    module_name: str
    constants: List[Dict[str, Any]]
    elapsed: float
    success: bool
    error: Optional[str] = None


class LakeExtractor:
    """Lightweight helper that shells out to `lake env lean --run`."""

    def __init__(self, project_dir: Path, lean_scripts_dir: Path):
        """
        Parameters
        ----------
        project_dir:
            Directory where `lake` commands should be executed. Typically this is the
            Lean project root that already imported mathlib (e.g. `LeanScripts`).
        lean_scripts_dir:
            Directory containing `ExtractConstants.lean`.
        """
        self.project_dir = project_dir.resolve()
        self.lean_scripts_dir = lean_scripts_dir.resolve()
        self.extract_script = self.lean_scripts_dir / "ExtractConstants.lean"

        if not self.project_dir.exists():
            raise FileNotFoundError(f"Lake project directory not found: {self.project_dir}")
        if not self.extract_script.exists():
            raise FileNotFoundError(f"ExtractConstants.lean not found: {self.extract_script}")

    def extract_module_constants(
        self,
        module_name: str,
        timeout: int = 600,
    ) -> ExtractionResult:
        """Extract constants for a single Lean module."""
        start = time.time()
        cmd = [
            "lake",
            "env",
            "lean",
            "--run",
            str(self.extract_script),
            module_name,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return ExtractionResult(
                module_name=module_name,
                constants=[],
                elapsed=elapsed,
                success=False,
                error=f"timeout after {timeout}s",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            return ExtractionResult(
                module_name=module_name,
                constants=[],
                elapsed=elapsed,
                success=False,
                error=str(exc),
            )

        elapsed = time.time() - start
        if proc.returncode != 0:
            return ExtractionResult(
                module_name=module_name,
                constants=[],
                elapsed=elapsed,
                success=False,
                error=proc.stderr.strip() or f"return code {proc.returncode}",
            )

        constants: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                constants.append(json.loads(text))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to decode JSON line for %s: %s", module_name, exc)
                continue

        return ExtractionResult(
            module_name=module_name,
            constants=constants,
            elapsed=elapsed,
            success=True,
        )

    def batch_extract(
        self,
        module_names: Iterable[str],
        timeout: int = 600,
    ) -> List[ExtractionResult]:
        """Extract multiple modules sequentially."""
        results: List[ExtractionResult] = []
        for module in module_names:
            result = self.extract_module_constants(module, timeout=timeout)
            if result.success:
                LOGGER.info(
                    "Extracted %d constants from %s (%.2fs)",
                    len(result.constants),
                    module,
                    result.elapsed,
                )
            else:
                LOGGER.error(
                    "Failed to extract %s after %.2fs: %s",
                    module,
                    result.elapsed,
                    result.error or "unknown error",
                )
            results.append(result)
        return results

