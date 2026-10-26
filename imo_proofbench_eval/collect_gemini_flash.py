#!/usr/bin/env python3
"""Collect Gemini-3.6-Flash answers using the experiment's v2-defaults payload."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import collect_novita as base


GEMINI_CANDIDATE_COUNT: Optional[int] = None
GEMINI_TOP_K: Optional[int] = None
GEMINI_THINKING_BUDGET: Optional[int] = None
GEMINI_THINKING_LEVEL: Optional[str] = None
PROBLEM_ID_FILTER: Optional[Set[str]] = None


def filtered_iter_jsonl_limited(path: Path, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    count = 0
    for obj in base.pb.iter_jsonl(path):
        pid = base.pb.get_problem_id(obj)
        if PROBLEM_ID_FILTER is not None and str(pid) not in PROBLEM_ID_FILTER:
            continue
        yield obj
        count += 1
        if limit is not None and count >= limit:
            break


def extract_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    meta = resp.get("usageMetadata")
    if not isinstance(meta, dict):
        return out
    mapping = {
        "promptTokenCount": "prompt_tokens",
        "candidatesTokenCount": "completion_tokens",
        "totalTokenCount": "total_tokens",
    }
    for src, dst in mapping.items():
        val = meta.get(src)
        if isinstance(val, int):
            out[dst] = val
    return out


def _messages_to_gemini_payload(
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
) -> Dict[str, Any]:
    system_parts: List[str] = []
    user_parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "\n\n".join(user_parts)}],
            }
        ]
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}],
        }
    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    if top_p is not None:
        generation_config["topP"] = float(top_p)
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = int(max_tokens)
    if GEMINI_CANDIDATE_COUNT is not None:
        generation_config["candidateCount"] = int(GEMINI_CANDIDATE_COUNT)
    if GEMINI_TOP_K is not None:
        generation_config["topK"] = int(GEMINI_TOP_K)
    thinking_config: Dict[str, Any] = {}
    if GEMINI_THINKING_BUDGET is not None:
        thinking_config["thinkingBudget"] = int(GEMINI_THINKING_BUDGET)
    if GEMINI_THINKING_LEVEL is not None:
        thinking_config["thinkingLevel"] = str(GEMINI_THINKING_LEVEL)
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config
    if generation_config:
        payload["generationConfig"] = generation_config
    return payload


def _extract_text(resp: Dict[str, Any]) -> str:
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except Exception:
        return ""
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "".join(texts)


def call_llm(
    pool: base.ClientPool,
    limiter: base.FixedWindowLimiter,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    top_p: Optional[float],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    del max_completion_tokens, reasoning_effort, verbosity
    limiter.acquire()
    start = base.time.time()
    try:
        payload = _messages_to_gemini_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        url = f"{pool.base_url.rstrip('/')}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": pool.api_key,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=pool.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        resp_obj = json.loads(raw)
        dur = base.time.time() - start
        meta = {
            "duration": dur,
            "base_url": pool.base_url,
            "response_id": resp_obj.get("responseId"),
            "model": resp_obj.get("modelVersion") or model,
            "usage": extract_usage(resp_obj),
        }
        return _extract_text(resp_obj), meta, None
    except Exception as exc:
        dur = base.time.time() - start
        meta = {"duration": dur, "base_url": pool.base_url, "error": str(exc)[:400]}
        return "", meta, str(exc)



def main() -> None:
    global GEMINI_CANDIDATE_COUNT, GEMINI_TOP_K, GEMINI_THINKING_BUDGET, GEMINI_THINKING_LEVEL, PROBLEM_ID_FILTER
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_collection_arguments(parser)
    parser.add_argument("--model", choices=["gemini-3.6-flash"], default="gemini-3.6-flash")
    parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--problem-id", action="append", default=None)
    parser.add_argument("--top-k-generation", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--thinking-level", default=None)
    args = parser.parse_args()
    GEMINI_CANDIDATE_COUNT = args.candidate_count
    GEMINI_TOP_K = args.top_k_generation
    GEMINI_THINKING_BUDGET = args.thinking_budget
    GEMINI_THINKING_LEVEL = args.thinking_level
    PROBLEM_ID_FILTER = set(args.problem_id or []) or None
    selectors = base.resolve_selectors(args)
    if args.dry_run:
        base.preview_inputs(args, [args.model], selectors, _messages_to_gemini_payload, filtered_iter_jsonl_limited)
        return
    metadata = {
        "gemini_generation_overrides": {
            "candidate_count": args.candidate_count,
            "top_k_generation": args.top_k_generation,
            "thinking_budget": args.thinking_budget,
            "thinking_level": args.thinking_level,
        },
        "problem_id_filter": sorted(PROBLEM_ID_FILTER) if PROBLEM_ID_FILTER is not None else None,
    }
    base.run_models(args, [args.model], selectors, call_llm, filtered_iter_jsonl_limited, metadata)


if __name__ == "__main__":
    main()
