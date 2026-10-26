#!/usr/bin/env python
"""Grade IMOBench outputs with Gemini judge (ProofAutoGrader prompt)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import urllib.request


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
BENCHMARK_JSONL: Optional[Path] = None
SAMPLES_PER_PROBLEM = 8


PROMPT_TEMPLATE = """You are an expert grader for the International Mathematics Olympiad (IMO). Your task is to evaluate a proposed solution strictly and rigorously. Keep in mind the standards at the IMO are extremely high: only arguments that are logically sound, complete, and precise should be rewarded.

### General Scoring Rubric
Scores are assigned on a 0-7 scale. The general guidelines are:
* **7 Points (Correct):** The solution is complete, correct, and fully rigorous. If the submission contains incorrect attempts or lines of reasoning but ultimately presents a complete and correct solution, it should still be awarded full points; the presence of earlier, discarded work does not detract from the final correct proof.
* **6 Points (Almost Correct):** The solution is almost correct with a sound core argument, but contains minor errors in calculation or small gaps in logic. Missing proofs for major components, unjustified claims, or sketchy arguments are **not** eligible for 6 points.
* **1 Point (Partial Progress):** The solution demonstrates substantial progress explicitly mentioned in the grading guidelines. Initial observations, reformulating the problem without making substantive headway, or proving partial results not mentioned in the grading guidelines are generally **not** eligible for this score.
* **0 Points (Incorrect):** The solution doesn't make substantial progress that is a key step in the full solution or is fundamentally flawed. All partial progress without key results or lacking rigor also fall in this category.

### Input Data and Interpretation
You are provided with the following:
1. **Problem Statement:** The IMO problem.
2. **Ground Truth Solution:** A reference solution. Assume this solution is correct. It demonstrates one valid approach.
3. **Specific Grading Guidelines:** Criteria for awarding credit for this specific problem. These guidelines take precedence over the General Scoring Rubric, especially for partial credit.
4. **Proposed Solution:** The student submission.

### Evaluation Process
You must follow this structured process:
1. **Analyze References:** Meticulously read and understand the problem and Ground Truth Solution check the Specific Grading Guidelines. Identify the key steps for a complete solution and the criteria for partial credit.
2. **Step-by-Step Verification:** Verify the logical validity and rigor of every step. Identify all flaws, gaps, assumptions, and errors. **Make sure you fully understand every piece of logic behind each step of the proposed solution, you must be careful for solutions that 'pretend' to be correct.**
3. **Assess Progress:** Determine the extent of non-trivial progress made.
4. **Score Determination:** Compare the findings against the Specific Grading Guidelines and the General Rubric to determine the final score.

### Output Requirements
You must provide your final score in the format `<points>N out of 7</points>`. Ensure the `<points>` block is used **only once**, as your answer will be parsed based on the first `<points> </points>` block that appears in your whole response.

--------------------
**PROBLEM STATEMENT**
{problem_statement}

**GROUND-TRUTH SOLUTION**
{solution}

**SPECIFIC GRADING GUIDELINES**
{guidelines}

**PROPOSED SOLUTION**
{student_answer}
--------------------

Present your detailed thought process and formal justification based on the scoring rubric and grading guidelines, and finally present your final score in the format below.

[Select one of the following options]
- `<points>7 out of 7</points>`
- `<points>6 out of 7</points>`
- `<points>1 out of 7</points>`
- `<points>0 out of 7</points>`
"""

TAG_SCORE_RE = re.compile(r"<points>\s*([0-7])\s*out of 7\s*</points>", re.IGNORECASE)
PLAIN_SCORE_RE = re.compile(r"\b([0-7])\s*out\s*of\s*7\b", re.IGNORECASE)
SLASH_SCORE_RE = re.compile(r"\b([0-7])\s*/\s*7\b", re.IGNORECASE)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def pick_field(obj: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_prompt(problem: str, solution: str, guidelines: str, student_answer: str) -> str:
    return PROMPT_TEMPLATE.format(
        problem_statement=problem,
        solution=solution,
        guidelines=guidelines,
        student_answer=student_answer,
    )


class FixedWindowLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self.lock = threading.Lock()
        self.window_start = time.time()
        self.count = 0

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                if now - self.window_start >= 60.0:
                    self.window_start = now
                    self.count = 0
                if self.count < self.max_per_minute:
                    self.count += 1
                    return
                wait_s = 60.0 - (now - self.window_start)
            time.sleep(max(0.05, wait_s))


def call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: float,
    temperature: Optional[float],
) -> Dict[str, Any]:
    url = f"{GEMINI_BASE_URL.rstrip('/')}/models/{model}:generateContent"
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    if temperature is not None:
        payload["generationConfig"] = {"temperature": float(temperature)}
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def _last_match_score(pattern: re.Pattern, text: str) -> Optional[int]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except Exception:
        return None


def parse_score(text: str) -> Tuple[Optional[int], Optional[str]]:
    score = _last_match_score(TAG_SCORE_RE, text)
    if score is not None:
        return score, "tag"
    score = _last_match_score(PLAIN_SCORE_RE, text)
    if score is not None:
        return score, "plain"
    score = _last_match_score(SLASH_SCORE_RE, text)
    if score is not None:
        return score, "slash"
    return None, None


def build_format_retry_prompt(problem: str, solution: str, guidelines: str, student_answer: str) -> str:
    return (
        "Return ONLY one line in the exact format:\n"
        "<points>N out of 7</points>\n"
        "Choose N from {7,6,1,0} based on the rubric and the materials below.\n\n"
        "PROBLEM STATEMENT:\n"
        f"{problem}\n\n"
        "GROUND-TRUTH SOLUTION:\n"
        f"{solution}\n\n"
        "SPECIFIC GRADING GUIDELINES:\n"
        f"{guidelines}\n\n"
        "PROPOSED SOLUTION:\n"
        f"{student_answer}\n"
    )


def answer_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_problem_map(path: Path) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        pid = obj.get("Problem ID") or obj.get("problem_id") or obj.get("_row_id")
        if pid is None:
            continue
        mapping[str(pid)] = obj
    return mapping


def graded_key(rec: Dict[str, Any]) -> Tuple[str, str, int, str]:
    pid = str(rec.get("problem_id") or rec.get("_row_id") or "")
    sample_idx = rec.get("sample_index", 0)
    try:
        sample_idx = int(sample_idx)
    except Exception:
        sample_idx = 0
    ah = rec.get("answer_hash") or ""
    return (str(rec.get("input_file") or ""), pid, sample_idx, str(ah))


def load_done_keys(path: Path, require_score: bool) -> set[Tuple[str, str, int, str]]:
    if not path.exists():
        return set()
    keys: set[Tuple[str, str, int, str]] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if require_score and not isinstance(rec.get("score"), int):
                continue
            keys.add(graded_key(rec))
    return keys


def run_for_file(
    path: Path,
    api_key: str,
    model: str,
    timeout_s: float,
    temperature: Optional[float],
    sleep_s: float,
    max_retries: int,
    rate_limit_per_min: int,
    workers: int,
    dry_run: bool,
    run_date: Optional[str],
    summary_csv: str,
    resume_missing: bool,
    explicit_input: bool = False,
) -> None:
    out_path = path.with_name(path.stem + "_graded.jsonl")
    done_keys = load_done_keys(out_path, require_score=resume_missing)

    input_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    total = 0
    skipped = 0
    tasks: List[Dict[str, Any]] = []

    print(f"[file] {path} -> {out_path}")
    for rec in iter_jsonl(path):
        total += 1
        ans = rec.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            skipped += 1
            continue

        input_file = str(BENCHMARK_JSONL) if BENCHMARK_JSONL is not None else rec.get("input_file") or ""
        pid = str(rec.get("problem_id") or rec.get("_row_id") or "")
        sample_idx = rec.get("sample_index", 0)
        try:
            sample_idx = int(sample_idx)
        except Exception:
            sample_idx = 0
        ah = answer_hash(ans)
        key = (str(input_file), pid, sample_idx, ah)
        if key in done_keys:
            skipped += 1
            continue

        if not input_file:
            print(f"[skip] missing input_file pid={pid} sample={sample_idx}")
            skipped += 1
            continue

        input_path = Path(str(input_file))
        if not input_path.exists():
            print(f"[skip] input_file missing {input_path}")
            skipped += 1
            continue

        if str(input_path) not in input_cache:
            input_cache[str(input_path)] = load_problem_map(input_path)
        pmap = input_cache[str(input_path)]
        pobj = pmap.get(pid)
        if not pobj:
            print(f"[skip] missing problem_id={pid} in {input_path}")
            skipped += 1
            continue

        problem = pick_field(pobj, ["Problem", "problem", "statement", "prompt"])
        solution = pick_field(pobj, ["Solution", "solution", "ground_truth_solution", "reference_solution"])
        guidelines = pick_field(pobj, ["Grading guidelines", "guidelines", "grading_guidelines", "rubric"])

        tasks.append(
            {
                "input_file": str(input_file),
                "problem_id": rec.get("problem_id"),
                "_row_id": rec.get("_row_id"),
                "sample_index": sample_idx,
                "mode": rec.get("mode"),
                "selector": rec.get("selector"),
                "model": rec.get("model"),
                "answer": ans,
                "answer_hash": ah,
                "problem": problem,
                "solution": solution,
                "guidelines": guidelines,
                "source_name": path.name,
            }
        )

    if not tasks:
        print(f"[done] total={total} graded=0 skipped={skipped} errors=0")
        if not dry_run:
            update_summary_csv(path.parent if explicit_input else path.parent.parent, run_date, summary_csv, out_path if explicit_input else None)
        return

    if dry_run:
        preview = tasks[0]
        print(json.dumps({"tasks": len(tasks), "prompt": build_prompt(preview["problem"], preview["solution"], preview["guidelines"], preview["answer"])}, ensure_ascii=False, indent=2))
        return

    limiter = FixedWindowLimiter(max_per_minute=rate_limit_per_min)
    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    graded = 0
    errors = 0
    processed = 0
    t_start = time.time()

    def worker(task: Dict[str, Any]) -> None:
        nonlocal graded, errors, processed
        prompt = build_prompt(task["problem"], task["solution"], task["guidelines"], task["answer"])
        attempt = 0
        last_err: Optional[str] = None
        resp_raw: Dict[str, Any] = {}
        resp_text = ""
        t0 = time.time()
        while attempt <= max_retries:
            attempt += 1
            limiter.acquire()
            try:
                print(f"[req] {task['source_name']} pid={task['problem_id']} s={task['sample_index']} attempt={attempt}")
                resp_raw = call_gemini(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout_s=timeout_s,
                    temperature=temperature,
                )
                resp_text = extract_text(resp_raw)
                last_err = None
                break
            except Exception as exc:
                last_err = str(exc)
                print(f"[err] pid={task['problem_id']} s={task['sample_index']} attempt={attempt} err={last_err}")
                time.sleep(1.0)

        score_primary, score_primary_mode = parse_score(resp_text)
        score_retry = None
        score_retry_mode = None
        retry_raw: Dict[str, Any] = {}
        retry_text = ""
        if last_err is None and score_primary is None:
            retry_prompt = build_format_retry_prompt(
                task["problem"], task["solution"], task["guidelines"], task["answer"]
            )
            attempt = 0
            while attempt <= max_retries:
                attempt += 1
                limiter.acquire()
                try:
                    print(
                        f"[retry-score] pid={task['problem_id']} s={task['sample_index']} attempt={attempt}"
                    )
                    retry_raw = call_gemini(
                        api_key=api_key,
                        model=model,
                        prompt=retry_prompt,
                        timeout_s=timeout_s,
                        temperature=temperature,
                    )
                    retry_text = extract_text(retry_raw)
                    score_retry, score_retry_mode = parse_score(retry_text)
                    break
                except Exception as exc:
                    last_err = str(exc)
                    print(
                        f"[retry-err] pid={task['problem_id']} s={task['sample_index']} attempt={attempt} err={last_err}"
                    )
                    time.sleep(1.0)

        score = score_retry if score_retry is not None else score_primary
        score_mode = score_retry_mode if score_retry is not None else score_primary_mode
        status = "ok" if last_err is None else "error"
        out_rec = {
            "input_file": task["input_file"],
            "problem_id": task["problem_id"],
            "_row_id": task["_row_id"],
            "sample_index": task["sample_index"],
            "mode": task["mode"],
            "selector": task["selector"],
            "model": task["model"],
            "answer_hash": task["answer_hash"],
            "status": status,
            "score": score,
            "score_primary": score_primary,
            "score_retry": score_retry,
            "score_parse_mode": score_mode,
            "score_primary_mode": score_primary_mode,
            "score_retry_mode": score_retry_mode,
            "judge_model": model,
            "judge_elapsed_s": time.time() - t0,
            "judge_response_text": resp_text,
            "judge_response_raw": resp_raw,
            "judge_retry_response_text": retry_text,
            "judge_retry_response_raw": retry_raw,
            "error": last_err,
            "grading_input": {
                "problem_statement": task["problem"],
                "solution": task["solution"],
                "guidelines": task["guidelines"],
                "student_answer": task["answer"],
            },
        }

        if not dry_run:
            with write_lock:
                with out_path.open("a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        if sleep_s > 0:
            time.sleep(sleep_s)

        with stats_lock:
            processed += 1
            if status == "ok":
                graded += 1
            else:
                errors += 1
            if processed % 20 == 0 or processed == len(tasks):
                elapsed = time.time() - t_start
                print(
                    f"[progress] judge_model={model} source={task['source_name']} "
                    f"mode={task['mode']} selector={task['selector']} "
                    f"{processed}/{len(tasks)} ok={graded} err={errors} elapsed_s={elapsed:.1f}"
                )

    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            _ = fut.result()

    print(f"[done] total={total} graded={graded} skipped={skipped} errors={errors}")
    update_summary_csv(path.parent if explicit_input else path.parent.parent, run_date, summary_csv, out_path if explicit_input else None)


def update_summary_csv(model_dir: Path, run_date: Optional[str], csv_name: str, graded_file: Optional[Path] = None) -> None:
    if run_date:
        pattern = f"*_{run_date}_graded.jsonl"
    else:
        pattern = "*_graded.jsonl"

    graded_files = [graded_file] if graded_file is not None and graded_file.is_file() else [] if graded_file is not None else sorted(model_dir.rglob(pattern))
    if not graded_files:
        return

    rows: List[Dict[str, Any]] = []
    for gpath in graded_files:
        problem_scores: Dict[str, Dict[int, int]] = {}
        invalid_problems: List[str] = []
        for rec in iter_jsonl(gpath):
            pid = rec.get("problem_id") or rec.get("_row_id")
            if pid is None:
                continue
            pid = str(pid)
            score = rec.get("score")
            sample_idx = rec.get("sample_index", 0)
            try:
                sample_idx = int(sample_idx)
            except Exception:
                sample_idx = 0
            if isinstance(score, int):
                problem_scores.setdefault(pid, {})[sample_idx] = score
            else:
                problem_scores.setdefault(pid, {})

        per_problem_avg: List[float] = []
        per_problem_avg_basic: List[float] = []
        per_problem_avg_advanced: List[float] = []
        for pid, smap in problem_scores.items():
            selected = {i: smap[i] for i in range(SAMPLES_PER_PROBLEM) if i in smap}
            if len(selected) == SAMPLES_PER_PROBLEM:
                avg = sum(selected.values()) / float(SAMPLES_PER_PROBLEM)
                per_problem_avg.append(avg)
                if pid.startswith("PB-Basic"):
                    per_problem_avg_basic.append(avg)
                elif pid.startswith("PB-Advanced"):
                    per_problem_avg_advanced.append(avg)
            else:
                invalid_problems.append(pid)

        overall_avg = sum(per_problem_avg) / len(per_problem_avg) if per_problem_avg else None
        overall_basic = (
            sum(per_problem_avg_basic) / len(per_problem_avg_basic) if per_problem_avg_basic else None
        )
        overall_advanced = (
            sum(per_problem_avg_advanced) / len(per_problem_avg_advanced) if per_problem_avg_advanced else None
        )
        overall_avg_pct = None
        if overall_avg is not None:
            overall_avg_pct = f"{(overall_avg / 7.0) * 100.0:.2f}"
        overall_basic_pct = None
        if overall_basic is not None:
            overall_basic_pct = f"{(overall_basic / 7.0) * 100.0:.2f}"
        overall_advanced_pct = None
        if overall_advanced is not None:
            overall_advanced_pct = f"{(overall_advanced / 7.0) * 100.0:.2f}"

        parts = gpath.parts
        mode = parts[-2] if len(parts) >= 2 else ""
        fname = gpath.name
        selector = fname.split("_", 1)[0] if "_" in fname else ""

        rows.append(
            {
                "jsonl_path": str(gpath),
                "mode": mode,
                "selector": selector,
                "overall_avg": overall_avg_pct,
                "overall_basic_pct": overall_basic_pct,
                "overall_advanced_pct": overall_advanced_pct,
                "num_problems": len(problem_scores),
                "num_valid_problems": len(per_problem_avg),
                "invalid_gemini_scoring": ",".join(invalid_problems),
            }
        )

    csv_path = model_dir / csv_name
    header = [
        "jsonl_path",
        "mode",
        "selector",
        "overall_avg",
        "overall_basic_pct",
        "overall_advanced_pct",
        "num_problems",
        "num_valid_problems",
        "invalid_gemini_scoring",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            values = [row.get(h, "") for h in header]
            out_vals = []
            for v in values:
                s = "" if v is None else str(v)
                if any(c in s for c in [",", "\"", "\n"]):
                    s = "\"" + s.replace("\"", "\"\"") + "\""
                out_vals.append(s)
            f.write(",".join(out_vals) + "\n")


def main() -> None:
    global GEMINI_BASE_URL, BENCHMARK_JSONL, SAMPLES_PER_PROBLEM
    ap = argparse.ArgumentParser(description="Grade IMOBench outputs with Gemini.")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--answers-jsonl", type=Path)
    inputs.add_argument("--output-root", type=Path)
    ap.add_argument("--run-date", type=str, default=None)
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--base-url", default=GEMINI_BASE_URL)
    ap.add_argument("--benchmark-jsonl", type=Path, required=True, help="JSONL with problems, reference solutions and grading rubrics.")
    ap.add_argument("--samples", type=int, default=8, help="Samples required for per-problem summary (use 4 for formalizer ablation).")
    ap.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    ap.add_argument("--timeout", type=float, default=360.0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--rate-limit-per-min", type=int, default=100)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-csv", type=str, default="grading_summary.csv")
    ap.add_argument("--resume-missing", action="store_true")
    ap.add_argument("--no-resume-missing", dest="resume_missing", action="store_false")
    ap.set_defaults(resume_missing=True)
    args = ap.parse_args()

    GEMINI_BASE_URL = args.base_url
    BENCHMARK_JSONL = args.benchmark_jsonl.expanduser().resolve()
    if not BENCHMARK_JSONL.is_file():
        ap.error(f"Benchmark JSONL not found: {BENCHMARK_JSONL}")
    if args.samples <= 0:
        ap.error("--samples must be positive")
    SAMPLES_PER_PROBLEM = args.samples
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")

    if args.run_date:
        pattern = f"*_{args.run_date}.jsonl"
    else:
        pattern = "*.jsonl"

    files = [args.answers_jsonl.expanduser().resolve()] if args.answers_jsonl is not None else sorted(args.output_root.rglob(pattern))
    if any(not path.is_file() for path in files):
        ap.error("An answers JSONL file does not exist")
    if not files:
        print(f"[info] no files matched under {args.output_root} pattern={pattern}")
        return

    print(f"[start] files={len(files)} model={args.model} run_date={args.run_date}")
    for path in files:
        if path.name.endswith("_graded.jsonl") or path.name.endswith("_graded_old.jsonl"):
            continue
        run_for_file(
            path=path,
            api_key=api_key,
            model=args.model,
            timeout_s=float(args.timeout),
            temperature=args.temperature,
            sleep_s=float(args.sleep),
            max_retries=int(args.max_retries),
            rate_limit_per_min=int(args.rate_limit_per_min),
            workers=int(args.workers),
            dry_run=bool(args.dry_run),
            run_date=args.run_date,
            summary_csv=str(args.summary_csv),
            resume_missing=bool(args.resume_missing),
            explicit_input=args.answers_jsonl is not None,
        )

    print("[all_done]")


if __name__ == "__main__":
    main()
