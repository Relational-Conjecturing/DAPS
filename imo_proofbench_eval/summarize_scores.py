#!/usr/bin/env python3
"""Reproduce the experiment's mean normalized judge score and problem-level SE."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def problem_scores(path: Path, samples: int) -> tuple[dict[str, float], dict[str, int]]:
    """Keep the last valid score for each (problem, sample_index)."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    grouped: dict[str, dict[int, int]] = {}
    counts = {"rows": 0, "invalid_rows": 0, "incomplete_problems": 0}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}") from exc
            pid = record.get("problem_id")
            if pid is None:
                pid = record.get("_row_id")
            score = record.get("score")
            try:
                index = int(record.get("sample_index", 0))
            except (TypeError, ValueError):
                counts["invalid_rows"] += 1
                continue
            if pid is None or type(score) is not int or not 0 <= score <= 7:
                counts["invalid_rows"] += 1
                continue
            if 0 <= index < samples:
                grouped.setdefault(str(pid), {})[index] = score
    complete = {}
    for pid, scores in grouped.items():
        if set(scores) == set(range(samples)):
            complete[pid] = sum(scores.values()) / samples / 7 * 100
        else:
            counts["incomplete_problems"] += 1
    return complete, counts


def statistics_row(scores: dict[str, float], problem_ids: list[str]) -> dict[str, Any]:
    if not problem_ids:
        return {"n_problems": 0, "mean_pct": None, "std_pct": None, "se_pct": None}
    values = [scores[pid] for pid in problem_ids]
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n_problems": len(values),
        "mean_pct": statistics.mean(values),
        "std_pct": std,
        "se_pct": std / math.sqrt(len(values)),
    }


def summarize(manifest_path: Path, common_only: bool = False, samples_override: int | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = samples_override if samples_override is not None else int(manifest.get("samples", 8))
    runs = manifest["runs"]
    loaded: list[tuple[dict[str, Any], dict[str, float], dict[str, int]]] = []
    seen = set()
    for run in runs:
        identity = (run["model"], run["selector"])
        if identity in seen:
            raise ValueError(f"Duplicate run identity: {identity}")
        seen.add(identity)
        source = Path(run["graded_file"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        scores, counts = problem_scores(source, samples)
        loaded.append((run, scores, counts))
        print(f"[input] model={run['model']} selector={run['selector']} rows={counts['rows']} complete={len(scores)} invalid={counts['invalid_rows']} incomplete={counts['incomplete_problems']}")
    common: dict[str, set[str]] = {}
    for run, scores, _ in loaded:
        model = run["model"]
        common[model] = common[model] & set(scores) if model in common else set(scores)
    rows = []
    for run, scores, counts in loaded:
        pids = sorted(common[run["model"]] if common_only else scores)
        row = {
            "model": run["model"],
            "selector": run["selector"],
            "problem_scope": "common_within_model" if common_only else "all_valid_for_selector",
            **statistics_row(scores, pids),
            "problem_ids": pids,
            "basic": statistics_row(scores, [pid for pid in pids if pid.startswith("PB-Basic")]),
            "advanced": statistics_row(scores, [pid for pid in pids if pid.startswith("PB-Advanced")]),
            "input_counts": counts,
        }
        rows.append(row)
        mean = row["mean_pct"]
        print(f"[score] model={row['model']} selector={row['selector']} N={len(pids)} mean_pct={mean} SE={row['se_pct']}")
    return {
        "metric": "mean per-problem mean judge score, normalized by 7 and multiplied by 100",
        "error_bar": "sample standard deviation across problem means / sqrt(N)",
        "samples": samples,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with samples and runs[{model,selector,graded_file}].")
    parser.add_argument("--output", type=Path, required=True, help="JSON output; a sibling CSV is also written.")
    parser.add_argument("--samples", type=int, default=None, help="Restrict to sample indices 0..samples-1.")
    parser.add_argument("--common-only", action="store_true")
    args = parser.parse_args()
    result = summarize(args.manifest, args.common_only, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["model", "selector", "problem_scope", "n_problems", "mean_pct", "std_pct", "se_pct"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    print(f"[saved] {args.output.name} and {csv_path.name}")


if __name__ == "__main__":
    main()
