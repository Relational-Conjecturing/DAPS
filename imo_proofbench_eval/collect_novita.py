#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import proofbench_helpers as pb


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------- logging --------------------------------------

_LOG_LOCK = threading.Lock()
_LOG_FH: Optional[Any] = None


def set_log_file(path: Path) -> None:
    global _LOG_FH
    with _LOG_LOCK:
        if _LOG_FH:
            try:
                _LOG_FH.close()
            except Exception:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = path.open("a", encoding="utf-8")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        if _LOG_FH:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()


def _is_balance_error(msg: str) -> bool:
    s = (msg or "").lower()
    return ("not_enough_balance" in s) or ("not enough balance" in s) or ("status code: 403" in s) or (" 403" in s)


def _fatal_balance_exit(msg: str) -> None:
    log(f"[fatal] NOT_ENOUGH_BALANCE detected: {msg}")
    os._exit(2)


# ---------------------------- helpers --------------------------------------


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")



def iter_jsonl_limited(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    count = 0
    for obj in pb.iter_jsonl(path):
        yield obj
        count += 1
        if limit is not None and count >= limit:
            break


def extract_problem_text_local(item: Dict[str, Any]) -> str:
    text = pb.extract_problem_text(item)
    if text:
        return text
    for key in ("formal_statement", "informal_statement", "statement"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def load_existing_keys(path: Path) -> set[Tuple[str, str, int]]:
    return pb.load_existing_keys(path)


def record_key(rec: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
    pid = rec.get("problem_id") or rec.get("_row_id")
    if pid is None:
        return None
    sample_idx = rec.get("sample_index", 0)
    try:
        sample_idx = int(sample_idx)
    except Exception:
        sample_idx = 0
    input_file = rec.get("input_file") or ""
    return (str(input_file), str(pid), sample_idx)


def is_valid_record(rec: Dict[str, Any]) -> bool:
    if rec.get("status") != "ok":
        return False
    ans = rec.get("answer")
    if not isinstance(ans, str) or not ans.strip():
        return False
    return True


def load_resume_state(path: Path, resume_fix_invalid: bool) -> Tuple[set, set, set, int, int]:
    if not path.exists():
        return set(), set(), set(), 0, 0

    records: List[Dict[str, Any]] = []
    last_idx: Dict[Tuple[str, str, int], int] = {}
    last_valid_idx: Dict[Tuple[str, str, int], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = len(records)
            records.append(rec)
            key = record_key(rec)
            if key is None:
                continue
            last_idx[key] = idx
            if is_valid_record(rec):
                last_valid_idx[key] = idx

    all_keys = set(last_idx.keys())
    valid_keys: set = set(last_valid_idx.keys())
    invalid_keys: set = set(k for k in all_keys if k not in valid_keys)

    kept = 0
    dropped = 0
    if resume_fix_invalid:
        keep_indices = set(last_valid_idx.values())
        if len(keep_indices) != len(records):
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for i, rec in enumerate(records):
                    if i in keep_indices:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        dropped += 1
            tmp.replace(path)

    return valid_keys, invalid_keys, all_keys, kept, dropped


@dataclass
class FixedWindowLimiter:
    max_per_minute: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    window_start: float = field(default_factory=time.time)
    count: int = 0

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


@dataclass
class RunStats:
    total: int
    completed: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_s: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total": self.total,
                "completed": self.completed,
                "errors": self.errors,
                "elapsed_s": time.time() - self.started_s,
            }


class ClientPool:
    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.local = threading.local()

    def get(self) -> Any:
        from openai import OpenAI

        client = getattr(self.local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
            self.local.client = client
        return client


def extract_usage(resp: Any) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    u = getattr(resp, "usage", None)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(u, key, None) if u is not None else None
        if isinstance(val, int):
            usage[key] = val
    if not usage:
        try:
            d = resp.to_dict()
            ud = d.get("usage") if isinstance(d, dict) else None
            if isinstance(ud, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = ud.get(key)
                    if isinstance(val, int):
                        usage[key] = val
        except Exception:
            pass
    return usage


def build_message_dump(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    system_parts: List[str] = []
    user_parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    return {
        "messages": messages,
        "system_prompt": "\n\n".join(system_parts),
        "user_prompt": "\n\n".join(user_parts),
    }


def call_llm(
    pool: ClientPool,
    limiter: FixedWindowLimiter,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    top_p: Optional[float],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    limiter.acquire()
    start = time.time()
    try:
        client = pool.get()
        req: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            req["temperature"] = temperature
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            req["max_completion_tokens"] = max_completion_tokens
        if reasoning_effort is not None:
            req["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            req["verbosity"] = verbosity
        if top_p is not None:
            req["top_p"] = top_p
        resp = client.chat.completions.create(**req)
        dur = time.time() - start
        content = ""
        if resp and resp.choices:
            msg = resp.choices[0].message
            content = msg.content or ""
        meta = {
            "duration": dur,
            "base_url": pool.base_url,
            "response_id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None) or model,
            "usage": extract_usage(resp),
        }
        return content, meta, None
    except Exception as exc:
        dur = time.time() - start
        if _is_balance_error(str(exc)):
            _fatal_balance_exit(str(exc))
        meta = {"duration": dur, "base_url": pool.base_url, "error": str(exc)[:400]}
        return "", meta, str(exc)


def compute_default_workers(avg_latency_s: float, rpm: int) -> int:
    workers = int((avg_latency_s * rpm) / 60.0 + 0.999)
    workers = min(64, max(1, workers + 4))
    return workers




def collect_for_model(
    model: str,
    input_jsonl: Path,
    output_root: Path,
    selectors: List[str],
    modes: List[str],
    top_k: int,
    samples: int,
    limit: Optional[int],
    base_url: str,
    api_key: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    top_p: Optional[float],
    request_timeout: float,
    llm_workers: int,
    limiter: FixedWindowLimiter,
    run_date: str,
    overwrite: bool,
    resume_fix_invalid: bool,
    record_input_messages: bool,
    call_api: Optional[Any] = None,
    item_iterator: Optional[Any] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    call_api = call_api or call_llm
    item_iterator = item_iterator or iter_jsonl_limited
    model_dir = output_root / safe_model_name(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_label = selectors[0]
    set_log_file(model_dir / f"run_{run_date}_{run_label}.log")

    log(f"[model] start model={model}")
    log(f"[config] selectors={selectors} modes={modes} top_k={top_k} samples={samples} limit={limit}")
    log(f"[config] llm_workers={llm_workers} rpm_limit={limiter.max_per_minute} base_url={base_url}")

    run_config = {
        "input_jsonl": str(input_jsonl),
        "output_root": str(model_dir),
        "run_date": run_date,
        "model": model,
        "modes": modes,
        "top_k": top_k,
        "samples": samples,
        "limit": limit,
        "base_url": base_url,
        "selectors": selectors,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "top_p": top_p,
        "request_timeout": request_timeout,
        "llm_workers": llm_workers,
        "rpm_limit": limiter.max_per_minute,
        "resume_fix_invalid": resume_fix_invalid,
        "record_input_messages": record_input_messages,
    }
    run_config.update(run_metadata or {})
    (model_dir / f"run_{run_date}_{run_label}.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pool = ClientPool(base_url=base_url, api_key=api_key, timeout_s=request_timeout)
    writer_lock = threading.Lock()

    def write_record(path: Path, rec: Dict[str, Any]) -> None:
        with writer_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    invalid_tasks: List[Tuple[Dict[str, Any], int, str, str, Path]] = []
    new_tasks: List[Tuple[Dict[str, Any], int, str, str, Path]] = []
    scheduled_keys: set = set()
    duplicate_input_samples = 0
    done_total = 0
    invalid_total = 0
    new_total = 0
    for selector in selectors:
        file_path = input_jsonl
        for mode in modes:
            out_dir = model_dir / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{selector}_{run_date}.jsonl"
            if overwrite and out_path.exists():
                out_path.unlink()
            valid_keys: set = set()
            invalid_keys: set = set()
            if resume_fix_invalid:
                valid_keys, invalid_keys, _, kept, dropped = load_resume_state(out_path, resume_fix_invalid=True)
                if kept or dropped:
                    log(
                        f"[resume] model={model} mode={mode} selector={selector} "
                        f"kept_valid={kept} dropped_invalid={dropped} file={file_path.name}"
                    )
            else:
                valid_keys = load_existing_keys(out_path)
            if valid_keys or invalid_keys:
                log(
                    f"[resume] model={model} mode={mode} selector={selector} "
                    f"valid={len(valid_keys)} invalid={len(invalid_keys)} file={file_path.name}"
                )
            for item in item_iterator(file_path, limit):
                pid = pb.get_problem_id(item) or str(item.get("_row_id", ""))
                for sample_idx in range(samples):
                    key = (str(file_path), str(pid), sample_idx)
                    task_key = (mode, *key)
                    if task_key in scheduled_keys:
                        duplicate_input_samples += 1
                        continue
                    scheduled_keys.add(task_key)
                    if key in invalid_keys:
                        invalid_tasks.append((item, sample_idx, mode, selector, file_path))
                        invalid_total += 1
                        continue
                    if key in valid_keys:
                        continue
                    new_tasks.append((item, sample_idx, mode, selector, file_path))
                    new_total += 1
            done_total += len(valid_keys)

    if duplicate_input_samples:
        log(f"[input] skipped duplicate problem/sample entries={duplicate_input_samples}")
    stats = RunStats(total=len(invalid_tasks) + len(new_tasks))
    log(
        f"[tasks] invalid={invalid_total} new={new_total} pending={len(invalid_tasks) + len(new_tasks)} "
        f"done={done_total} total={done_total + len(invalid_tasks) + len(new_tasks)}"
    )

    mode_order = {"matched": 0}
    invalid_tasks.sort(key=lambda t: mode_order.get(t[2], 9))
    new_tasks.sort(key=lambda t: mode_order.get(t[2], 9))

    def run_one(item: Dict[str, Any], sample_idx: int, mode: str, selector: str, file_path: Path) -> None:
        row_id = int(item.get("_row_id", 0))
        pid = pb.get_problem_id(item) or str(row_id)
        problem = extract_problem_text_local(item)
        if not problem:
            rec = {
                "problem_id": pid,
                "_row_id": row_id,
                "sample_index": sample_idx,
                "mode": mode,
                "selector": selector,
                "status": "error",
                "stage": "input",
                "reason": "missing problem text",
                "answer": "",
                "premise_count": 0,
                "premise_names": [],
                "premise_source": "none",
                "input_file": str(file_path),
            }
            out_path = model_dir / mode / f"{selector}_{run_date}.jsonl"
            write_record(out_path, rec)
            with stats.lock:
                stats.completed += 1
                stats.errors += 1
            return

        context_text, premise_names, premise_count, premise_source = pb.matched_context(item, top_k)
        messages = pb.build_answer_messages(problem, context_text)

        label = f"{mode} pid={pid} s={sample_idx}"
        log(f"[llm] send model={model} {label}")
        answer, meta, err = call_api(
            pool=pool,
            limiter=limiter,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            top_p=top_p,
        )
        if not err and not (answer or "").strip():
            err = "empty_answer"
        rec: Dict[str, Any] = {
            "problem_id": pid,
            "_row_id": row_id,
            "sample_index": sample_idx,
            "mode": mode,
            "selector": selector,
            "status": "ok" if not err else "error",
            "stage": "answer",
            "reason": None if not err else str(err)[:200],
            "answer": answer,
            "answer_duration": meta.get("duration", 0.0),
            "answer_url": meta.get("base_url", ""),
            "model": meta.get("model", model),
            "usage": meta.get("usage", {}),
            "prompt_tokens": meta.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": meta.get("usage", {}).get("completion_tokens"),
            "total_tokens": meta.get("usage", {}).get("total_tokens"),
            "premise_count": premise_count,
            "premise_names": premise_names,
            "premise_source": premise_source,
            "input_file": str(file_path),
        }
        if record_input_messages:
            rec.update(build_message_dump(messages))
        out_path = model_dir / mode / f"{selector}_{run_date}.jsonl"
        write_record(out_path, rec)
        with stats.lock:
            stats.completed += 1
            if err:
                stats.errors += 1
        if stats.completed % 20 == 0 or stats.completed == stats.total:
            snap = stats.snapshot()
            elapsed = float(snap["elapsed_s"])
            done = int(snap["completed"])
            total = int(snap["total"])
            rate = (done / elapsed) if elapsed > 0 else 0.0
            remaining = max(0, total - done)
            eta_s = (remaining / rate) if rate > 0 else 0.0
            log(
                f"[progress] model={model} mode={mode} selector={selector} "
                f"{done}/{total} errors={snap['errors']} "
                f"elapsed={elapsed:.1f}s eta={eta_s/60.0:.1f}m"
            )

    def run_tasks(task_list: List[Tuple[Dict[str, Any], int, str, str, Path]], label: str) -> None:
        if not task_list:
            return
        log(f"[phase] start {label} count={len(task_list)}")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = max(1, llm_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_one, *t) for t in task_list]
            for fut in as_completed(futures):
                _ = fut.result()
        log(f"[phase] done {label} count={len(task_list)}")

    run_tasks(invalid_tasks, "resume_invalid")
    run_tasks(new_tasks, "new")

    log(f"[model] done model={model} completed={stats.completed} errors={stats.errors}")


NOVITA_MODELS = (
    "qwen/qwen3-235b-a22b-fp8",
    "deepseek/deepseek-v3.1",
    "moonshotai/kimi-k2-instruct",
)


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-jsonl", "--input", dest="input_jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["matched"], default="matched")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--selector", "--label", dest="selector", default=None, help="Optional output label; defaults to the input filename stem.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--rate-limit-per-min", type=int, default=30)
    parser.add_argument("--avg-latency", type=float, default=35.0)
    parser.add_argument("--llm-workers", type=int, default=None)
    parser.add_argument("--resume-fix-invalid", dest="resume_fix_invalid", action="store_true", default=True)
    parser.add_argument("--no-resume-fix-invalid", dest="resume_fix_invalid", action="store_false")
    parser.add_argument("--record-input-messages", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without API calls or file writes.")


def resolve_selectors(args: argparse.Namespace) -> List[str]:
    if args.samples <= 0 or args.top_k <= 0 or args.rate_limit_per_min <= 0:
        raise SystemExit("samples, top-k and rate-limit-per-min must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    args.input_jsonl = args.input_jsonl.expanduser().resolve()
    if not args.input_jsonl.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input_jsonl}")
    label = args.selector or args.input_jsonl.stem
    if not label or label in {".", ".."} or "/" in label or "\\" in label:
        raise SystemExit("selector must be a single output label")
    return [label]


def preview_inputs(args: argparse.Namespace, models: List[str], selectors: List[str],
                   payload_builder: Optional[Any] = None,
                   item_iterator: Optional[Any] = None) -> None:
    iterator = item_iterator or iter_jsonl_limited
    for model in models:
        for selector in selectors:
            found = False
            for item in iterator(args.input_jsonl, args.limit):
                context, names, _, _ = pb.matched_context(item, args.top_k)
                messages = pb.build_answer_messages(extract_problem_text_local(item), context)
                payload = {"model": model, "messages": messages}
                for key in ("temperature", "top_p", "max_tokens"):
                    value = getattr(args, key)
                    if value is not None:
                        payload[key] = value
                if payload_builder is not None:
                    payload = payload_builder(messages, args.temperature, args.max_tokens, args.top_p)
                print(json.dumps({"model": model, "selector": selector,
                                  "input_jsonl": str(args.input_jsonl),
                                  "problem_id": pb.get_problem_id(item), "samples": args.samples,
                                  "premise_names": names, "request_payload": payload},
                                 ensure_ascii=False, indent=2), flush=True)
                found = True
                break
            if not found:
                raise SystemExit(f"No matching input records in {args.input_jsonl}")


def run_models(args: argparse.Namespace, models: List[str], selectors: List[str],
               call_api: Optional[Any] = None, item_iterator: Optional[Any] = None,
               run_metadata: Optional[Dict[str, Any]] = None) -> None:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")
    run_date = args.run_date or now_ts()
    limiter = FixedWindowLimiter(max_per_minute=args.rate_limit_per_min)
    workers = args.llm_workers or compute_default_workers(args.avg_latency, args.rate_limit_per_min)
    for model in models:
        collect_for_model(
            model=model, input_jsonl=args.input_jsonl, output_root=args.output_root,
            selectors=selectors, modes=["matched"], top_k=args.top_k, samples=args.samples,
            limit=args.limit, base_url=args.base_url, api_key=api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            max_completion_tokens=None, reasoning_effort=None, verbosity=None, top_p=args.top_p,
            request_timeout=args.request_timeout, llm_workers=workers, limiter=limiter,
            run_date=run_date, overwrite=args.overwrite, resume_fix_invalid=args.resume_fix_invalid,
            record_input_messages=args.record_input_messages, call_api=call_api, item_iterator=item_iterator,
            run_metadata=run_metadata,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_collection_arguments(parser)
    parser.add_argument("--model", action="append", choices=NOVITA_MODELS, required=True)
    parser.add_argument("--base-url", default="https://api.novita.ai/openai")
    parser.add_argument("--api-key-env", default="NOVITA_API_KEY")
    args = parser.parse_args()
    selectors = resolve_selectors(args)
    if args.dry_run:
        preview_inputs(args, args.model, selectors)
        return
    run_models(args, args.model, selectors)


if __name__ == "__main__":
    main()
