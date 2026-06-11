#!/usr/bin/env python3
"""
tests/run_tests.py

Concurrent end-to-end harness. Simulates N simultaneous users (each running
a sequential session of queries against the live API) and writes every
turn's full debug trace to a single JSON file under `debug_results/`.

What it does:

  1. Loads queries from `user_query_examples.md` (1-based indices).
  2. Loads expected categories from `tests/query_categories.yml`.
  3. Shuffles queries with a seeded RNG (reproducible).
  4. Splits into N sessions of ~M queries each. Each session has a unique
     `user_id` so its conversation history accumulates server-side.
  5. Runs sessions in parallel (asyncio.gather), sequential within a session.
  6. For each query, posts to /chat/stream with X-Debug-Key and collects all
     NDJSON events. Extracts token-usage, retrieval, rerank, citation info,
     and runs the shared classifier (tests/_classifier.py) for pass/fail.
  7. Writes a single JSON file per query under:
        debug_results/<run_ts>/session_<n>/q<seq>_<safehash>.json
     Plus _session_summary.json per session and _run_summary.md at the top.
  8. Exit code: 0 iff no CRITICAL-category failures, else 1.

Usage:
  ADMIN_DEBUG_SECRET=... PYTHONPATH=. python3 tests/run_tests.py \\
      --api-url http://localhost:9000 \\
      --concurrent 8 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tests._classifier import (  # noqa: E402
    CRITICAL,
    classify_outcome,
    extract_citation_tags,
)


# ============================================================
# Paths
# ============================================================

QUERIES_PATH = REPO_ROOT / "user_query_examples.md"
CATEGORIES_PATH = THIS_DIR / "query_categories.yml"
OUTPUT_ROOT = REPO_ROOT / "debug_results"


# ============================================================
# Loaders
# ============================================================

def load_queries() -> List[str]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_categories() -> Dict[int, str]:
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("queries", {}) or {}
    return {int(k): str(v).strip() for k, v in raw.items()}


# ============================================================
# Streaming + event extraction
# ============================================================

async def stream_query(
    http: httpx.AsyncClient,
    api_url: str,
    user_id: str,
    query: str,
    debug_key: str,
    timeout: float,
) -> Dict[str, Any]:
    """POST /chat/stream, collect all NDJSON events. Never raises — returns
    an error string in `error` field if anything goes wrong."""
    t0 = time.time()
    events: List[Dict[str, Any]] = []
    error: Optional[str] = None
    headers = {"X-Debug-Key": debug_key} if debug_key else {}

    try:
        async with http.stream(
            "POST",
            f"{api_url}/chat/stream",
            json={"user_id": user_id, "query": query},
            headers=headers,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                error = f"HTTP {resp.status_code}: {body[:200].decode('utf-8', errors='replace')}"
            else:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "events": events,
        "error": error,
        "elapsed_ms": elapsed_ms,
    }


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the headline fields out of the event stream."""
    intent: Optional[str] = None
    sub_queries: List[str] = []
    history_messages_loaded = 0
    passages_returned: Optional[int] = None
    rerank: Dict[str, Any] = {}
    rerank_degraded = False
    disambiguate_fired = False
    composer_chars: Optional[int] = None
    final_answer = ""
    final_reason: Optional[str] = None

    token_usage: Dict[str, Optional[Dict[str, int]]] = {
        "router": None, "rerank": None, "composer": None, "nli": None,
    }

    streamed_chars = 0

    for ev in events:
        t = ev.get("type")
        if t == "router_done":
            intent = ev.get("intent")
            sub_queries = ev.get("sub_queries", []) or []
            token_usage["router"] = ev.get("token_usage")
        elif t == "history_loaded":
            history_messages_loaded = int(ev.get("turns", 0) or 0)
        elif t == "retrieval_done":
            passages_returned = ev.get("passages_returned")
            rerank = ev.get("rerank", {}) or {}
            rerank_degraded = bool(ev.get("degraded"))
            token_usage["rerank"] = ev.get("token_usage")
        elif t == "disambiguate_required":
            disambiguate_fired = True
        elif t == "composer_done":
            composer_chars = ev.get("chars")
            token_usage["composer"] = ev.get("token_usage")
        elif t == "verification_result":
            token_usage["nli"] = ev.get("token_usage")
        elif t == "answer_chunk":
            streamed_chars += len(ev.get("content", "") or "")
        elif t == "final_answer":
            final_answer = ev.get("content", "") or final_answer
            final_reason = ev.get("reason") or final_reason

    yes_total = sum(
        1 for entries in rerank.values()
        for entry in (entries or [])
        if isinstance(entry, list) and len(entry) >= 2 and entry[1] == 0
    )
    weak_total = sum(
        1 for entries in rerank.values()
        for entry in (entries or [])
        if isinstance(entry, list) and len(entry) >= 2 and entry[1] == 1
    )

    return {
        "intent": intent,
        "sub_queries": sub_queries,
        "history_messages_loaded": history_messages_loaded,
        "passages_returned": passages_returned,
        "rerank_yes": yes_total,
        "rerank_weak": weak_total,
        "rerank_degraded": rerank_degraded,
        "disambiguate_fired": disambiguate_fired,
        "composer_chars": composer_chars,
        "streamed_chars": streamed_chars,
        "final_answer": final_answer,
        "final_reason": final_reason,
        "citation_tags": extract_citation_tags(final_answer),
        "token_usage": token_usage,
    }


# ============================================================
# Session worker
# ============================================================

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


async def run_session(
    session_index: int,
    user_id: str,
    queries: List[Tuple[int, str, str]],   # [(corpus_idx, category, query), ...]
    http: httpx.AsyncClient,
    api_url: str,
    debug_key: str,
    per_query_timeout: float,
    out_dir: Path,
    run_id: str,
    progress_q: "asyncio.Queue[Dict[str, Any]]",
) -> Dict[str, Any]:
    """Run one session: sequential queries, accumulating history server-side."""
    session_dir = out_dir / f"session_{session_index:02d}"
    session_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for seq, (corpus_idx, category, query) in enumerate(queries, start=1):
        t_start = datetime.now(timezone.utc)
        result = await stream_query(
            http=http,
            api_url=api_url,
            user_id=user_id,
            query=query,
            debug_key=debug_key,
            timeout=per_query_timeout,
        )
        t_end = datetime.now(timezone.utc)

        summary = summarize_events(result["events"])
        classifier_input = {
            "final_answer": summary["final_answer"],
            "intent": summary["intent"],
            "final_reason": summary["final_reason"],
            "error": result["error"],
        }
        outcome = classify_outcome(category, classifier_input)

        record = {
            "run_id": run_id,
            "session_index": session_index,
            "user_id": user_id,
            "query_index_in_session": seq,
            "original_corpus_index": corpus_idx,
            "category": category,
            "query": query,
            "timestamp_start": t_start.isoformat(),
            "timestamp_end": t_end.isoformat(),
            "elapsed_ms": result["elapsed_ms"],
            "outcome": outcome,
            "summary": summary,
            "events": result["events"],
            "error": result["error"],
        }
        rows.append(record)

        fname = f"q{seq:02d}_{_safe_hash(query)}.json"
        with open(session_dir / fname, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        await progress_q.put({
            "session_index": session_index,
            "seq": seq,
            "total": len(queries),
            "corpus_idx": corpus_idx,
            "category": category,
            "outcome": outcome,
            "elapsed_ms": result["elapsed_ms"],
            "query": query,
        })

    # Per-session summary
    session_summary = {
        "session_index": session_index,
        "user_id": user_id,
        "queries": len(rows),
        "passed": sum(1 for r in rows if r["outcome"]["pass"]),
        "failed": sum(1 for r in rows if not r["outcome"]["pass"]),
        "categories": _by_category(rows),
        "total_token_usage": _total_tokens(rows),
    }
    with open(session_dir / "_session_summary.json", "w", encoding="utf-8") as f:
        json.dump(session_summary, f, ensure_ascii=False, indent=2)

    return {"session_index": session_index, "rows": rows, "summary": session_summary}


def _by_category(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c = r["category"]
        out.setdefault(c, {"pass": 0, "fail": 0})
        if r["outcome"]["pass"]:
            out[c]["pass"] += 1
        else:
            out[c]["fail"] += 1
    return out


def _total_tokens(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {"prompt": 0, "completion": 0}
    for r in rows:
        for u in (r["summary"]["token_usage"] or {}).values():
            if not u:
                continue
            totals["prompt"] += int(u.get("prompt", 0) or 0)
            totals["completion"] += int(u.get("completion", 0) or 0)
    return totals


# ============================================================
# Progress reporter
# ============================================================

async def progress_reporter(
    progress_q: "asyncio.Queue[Dict[str, Any]]",
    expected: int,
) -> None:
    """Print one line per completed query as sessions stream them in."""
    seen = 0
    while seen < expected:
        ev = await progress_q.get()
        seen += 1
        mark = "✓" if ev["outcome"]["pass"] else "✗"
        print(
            f"  [{seen:>4}/{expected}] s{ev['session_index']:02d}#{ev['seq']:02d} "
            f"Q{ev['corpus_idx']:<3} {ev['category']:<18} "
            f"{mark}  {ev['outcome']['verdict']:<28} "
            f"({ev['elapsed_ms']:>5}ms)  {ev['query'][:55]}"
        )


# ============================================================
# Reporting
# ============================================================

def write_run_summary(path: Path, run_id: str, all_rows: List[Dict[str, Any]],
                      sessions: List[Dict[str, Any]]) -> None:
    by_cat = _by_category(all_rows)
    lats = sorted(r["elapsed_ms"] for r in all_rows if r["elapsed_ms"])
    p50 = lats[len(lats) // 2] if lats else 0
    p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)] if lats else 0

    totals = {"prompt": 0, "completion": 0}
    for s in sessions:
        t = s["summary"]["total_token_usage"]
        totals["prompt"] += t["prompt"]
        totals["completion"] += t["completion"]

    lines = [
        f"# run_tests — {run_id}",
        "",
        f"Total queries: {len(all_rows)}",
        f"Passed: {sum(1 for r in all_rows if r['outcome']['pass'])}",
        f"Failed: {sum(1 for r in all_rows if not r['outcome']['pass'])}",
        f"Sessions: {len(sessions)}",
        "",
        f"Total tokens — prompt: {totals['prompt']}, completion: {totals['completion']}",
        "",
        "## Per-category",
        "",
    ]
    for cat in sorted(by_cat):
        p = by_cat[cat]["pass"]
        f_ = by_cat[cat]["fail"]
        critical = " (CRITICAL)" if cat in CRITICAL else ""
        lines.append(f"- `{cat}`{critical}: {p}/{p + f_} pass")

    lines.append("")
    lines.append("## Latency")
    lines.append(f"- count: {len(lats)}")
    lines.append(f"- p50:   {p50}ms")
    lines.append(f"- p95:   {p95}ms")
    lines.append(f"- max:   {max(lats) if lats else 0}ms")
    lines.append("")

    # First 5 failures
    fails = [r for r in all_rows if not r["outcome"]["pass"]][:5]
    if fails:
        lines.append("## First 5 failures")
        lines.append("")
        for r in fails:
            lines.append(f"- **Q{r['original_corpus_index']}** ({r['category']}) "
                         f"verdict={r['outcome']['verdict']}: `{r['query'][:80]}`")
            final = (r["summary"]["final_answer"] or "")[:200].replace("\n", " ")
            lines.append(f"  - final[:200]: `{final}`")
            lines.append(f"  - detail: {r['outcome']['detail']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

async def main_async(args: argparse.Namespace) -> int:
    debug_key = os.environ.get("ADMIN_DEBUG_SECRET", "")
    if not debug_key:
        print("WARN: ADMIN_DEBUG_SECRET not set — debug events will be filtered "
              "out and token-usage data will be missing.", file=sys.stderr)

    queries = load_queries()
    categories = load_categories()
    if not queries:
        print("ERROR: no queries loaded", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    ordered = [(i + 1, categories.get(i + 1, "gov_factual"), q) for i, q in enumerate(queries)]
    rng.shuffle(ordered)

    if args.limit:
        ordered = ordered[: args.limit]

    # Distribute round-robin into N sessions so each session is a mix.
    n_sessions = max(1, args.concurrent)
    buckets: List[List[Tuple[int, str, str]]] = [[] for _ in range(n_sessions)]
    for idx, item in enumerate(ordered):
        buckets[idx % n_sessions].append(item)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    total_queries = sum(len(b) for b in buckets)
    print(f"== run_tests {run_id} ==")
    print(f"  API:        {args.api_url}")
    print(f"  queries:    {total_queries}  (corpus={len(queries)}, limit={args.limit or '-'})")
    print(f"  sessions:   {n_sessions} (sizes={[len(b) for b in buckets]})")
    print(f"  out:        {out_dir}")
    print()

    progress_q: asyncio.Queue = asyncio.Queue(maxsize=0)
    reporter = asyncio.create_task(progress_reporter(progress_q, total_queries))

    async with httpx.AsyncClient(timeout=None) as http:
        session_tasks = [
            run_session(
                session_index=i + 1,
                user_id=f"test_{run_id}_{i + 1:02d}",
                queries=bucket,
                http=http,
                api_url=args.api_url,
                debug_key=debug_key,
                per_query_timeout=args.timeout,
                out_dir=out_dir,
                run_id=run_id,
                progress_q=progress_q,
            )
            for i, bucket in enumerate(buckets) if bucket
        ]
        sessions = await asyncio.gather(*session_tasks)

    await reporter
    all_rows = [r for s in sessions for r in s["rows"]]

    run_summary_path = out_dir / "_run_summary.md"
    write_run_summary(run_summary_path, run_id, all_rows, sessions)

    print()
    print(f"== complete. summary → {run_summary_path} ==")
    by_cat = _by_category(all_rows)
    for cat in sorted(by_cat):
        p, f_ = by_cat[cat]["pass"], by_cat[cat]["fail"]
        crit = " CRITICAL" if cat in CRITICAL else ""
        print(f"  {cat:<20} {p:>3} pass · {f_:>3} fail{crit}")
    print()

    critical_failed = any(
        (not r["outcome"]["pass"]) and r["category"] in CRITICAL
        for r in all_rows
    )
    if critical_failed:
        print("FAIL: at least one CRITICAL-category query failed.")
        return 1
    print("PASS: no CRITICAL-category failures.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",
                        default=os.environ.get("GOVOPS_API_URL", "http://localhost:9000"))
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Number of simultaneous sessions (~simulated users)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries (after shuffle)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible shuffles")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-query timeout in seconds")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
