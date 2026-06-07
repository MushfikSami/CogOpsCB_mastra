#!/usr/bin/env python3
"""
Replay user_query_examples.md through the orchestrator and save results.

Usage:
    python tests/run_queries.py                       # run a representative subset (~20 queries)
    python tests/run_queries.py --all                 # run every line of the examples file
    python tests/run_queries.py --indices 1,5,80,116  # cherry-pick by 1-based line number
    python tests/run_queries.py --no-classifier       # disable intent classifier (factual-only path)
    python tests/run_queries.py --no-verifier         # disable NLI verifier
    python tests/run_queries.py --out PATH            # custom output file (default: data/test_runs/run_<ts>.jsonl)

Output:
    JSONL where each line is one query result:
      {
        "idx":            int (1-based line number in examples file),
        "query":          str (the question),
        "intent":         "factual" | "chitchat" | "refuse" | null,
        "final_answer":   str (the post-verify, post-Sources-block answer),
        "raw_streamed":   str (the LLM's raw streamed text before post-process),
        "source_tags":    [str, ...] (S# tags that appear in final answer),
        "sources":        [dict, ...] (full source_map at end of turn),
        "events":         [dict, ...] (full event trace minus answer_chunks),
        "duration_ms":    int,
        "error":          str | null,
      }

A summary file is also written alongside the JSONL with one-line per query
showing intent, final-answer-preview, and whether citations were produced.

The orchestrator's normal config defaults are used, EXCEPT the intent classifier
and verifier are forced ON for the test run (override --no-classifier / --no-verifier
to disable). The orchestrator instance is reused across queries so the LLM
clients and tool registry are loaded once.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO = Path(__file__).resolve().parent.parent

# A curated representative subset — covers grounded gov-services, refusal,
# chitchat, political refusal, mixed-language, identity & opinion edge cases.
DEFAULT_SUBSET = [
    1,    # factual: SSC certificate board correction
    3,    # factual: NID status check
    16,   # factual: HSC certificate name correction
    36,   # factual: MRT Pass purchase location
    49,   # factual: disability ID acquisition
    50,   # factual: metro fare payment
    52,   # factual: disability helpline number
    64,   # factual: police clearance use cases
    76,   # English mixed: cyber-bullying punishment
    80,   # off-topic factual: army chief (likely no Jiggasha match)
    82,   # off-topic factual: Fazle Hasan Abed
    84,   # political: Jamaat party symbol
    87,   # political/genealogical: Tareq Rahman father's name
    93,   # English mixed: who's the foreign minister now
    94,   # political opinion: PM is Tareq or Salauddin?
    109,  # political/factual: current PM
    113,  # factual: today's date
    116,  # chitchat: hi bro
    117,  # factual: PM of bangladesh
    102,  # chitchat-or-political-name: "hasina"
]


def load_queries() -> List[str]:
    """Load user_query_examples.md as one-query-per-non-blank-line."""
    p = REPO / "data" / "user_query_examples.md"
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]  # drop blanks


def parse_indices(arg: str) -> List[int]:
    out: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


_SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]")


async def run_one(orchestrator, idx: int, query: str, timeout_sec: float = 90.0) -> Dict[str, Any]:
    """Run a single query through the orchestrator, capture results."""
    started = time.time()
    events: List[Dict[str, Any]] = []
    raw_streamed_parts: List[str] = []
    final_answer: str = ""
    intent: Optional[str] = None
    sources_meta: List[Dict[str, Any]] = []
    error: Optional[str] = None

    try:
        async def consume():
            nonlocal final_answer, intent, sources_meta
            async for evt in orchestrator.process_query(query, user_id=f"test_run_{idx}"):
                et = evt.get("type")
                if et == "answer_chunk":
                    raw_streamed_parts.append(evt.get("content", ""))
                    # don't store raw answer chunks in events (noisy)
                    continue
                if et == "final_answer":
                    final_answer = evt.get("content", "")
                    sources_meta = evt.get("sources", []) or []
                if et == "intent_classified":
                    intent = evt.get("intent")
                events.append(evt)

        await asyncio.wait_for(consume(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        error = f"timeout after {timeout_sec}s"
    except Exception as e:
        error = f"{e.__class__.__name__}: {e}"

    duration_ms = int((time.time() - started) * 1000)
    raw_streamed = "".join(raw_streamed_parts)
    source_tags = sorted(set(_SOURCE_TAG_RE.findall(final_answer or "")))

    return {
        "idx": idx,
        "query": query,
        "intent": intent,
        "final_answer": final_answer,
        "raw_streamed": raw_streamed,
        "source_tags": [f"S{t}" for t in source_tags],
        "sources": sources_meta,
        "events": events,
        "duration_ms": duration_ms,
        "error": error,
    }


def format_summary_line(rec: Dict[str, Any]) -> str:
    """One short readable line per query."""
    idx = rec["idx"]
    q = rec["query"]
    if len(q) > 60:
        q_disp = q[:57] + "…"
    else:
        q_disp = q
    intent = rec.get("intent") or "—"
    tags = ",".join(rec.get("source_tags", []) or []) or "none"
    err = rec.get("error")
    ans = rec.get("final_answer", "") or ""
    ans_one = re.sub(r"\s+", " ", ans).strip()
    if len(ans_one) > 80:
        ans_one = ans_one[:77] + "…"
    err_part = f"  ERROR={err}" if err else ""
    return (
        f"[{idx:>3}] intent={intent:<8} cites={tags:<14} "
        f"{rec.get('duration_ms', 0):>5}ms  Q: {q_disp}\n"
        f"     A: {ans_one}{err_part}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run every query (else representative subset)")
    ap.add_argument("--indices", type=str, default=None,
                    help="Comma-separated 1-based line indices to run")
    ap.add_argument("--no-classifier", action="store_true", help="Disable intent classifier")
    ap.add_argument("--no-verifier", action="store_true", help="Disable NLI verifier")
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path")
    ap.add_argument("--timeout", type=float, default=90.0, help="Per-query timeout seconds")
    args = ap.parse_args()

    # Load environment from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    queries = load_queries()
    print(f"Loaded {len(queries)} queries from user_query_examples.md")

    if args.indices:
        wanted = parse_indices(args.indices)
    elif args.all:
        wanted = list(range(1, len(queries) + 1))
    else:
        wanted = [i for i in DEFAULT_SUBSET if 1 <= i <= len(queries)]

    print(f"Will run {len(wanted)} queries: {wanted}")

    # Late import so any orchestrator-side env errors surface here
    from cogops.agents.orchestrator import Orchestrator

    orch = Orchestrator()
    # Force-enable both grounding layers for the test run (defaults are off
    # for safety until thresholds are tuned, but we want to exercise them).
    orch.intent_classifier_enabled = not args.no_classifier
    orch.verifier_enabled = not args.no_verifier

    print(
        f"Orchestrator ready: agent={orch.agent_name} "
        f"primary={orch.llm_service.model} "
        f"secondary={orch.secondary_service.model}"
    )

    out_path = Path(args.out) if args.out else (
        REPO / "data" / "test_runs" / f"run_{int(time.time())}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.txt")

    print(f"Writing JSONL to {out_path}")
    print(f"Writing summary to {summary_path}\n")

    results: List[Dict[str, Any]] = []

    async def go():
        with open(out_path, "w", encoding="utf-8") as f_jsonl, \
             open(summary_path, "w", encoding="utf-8") as f_sum:
            f_sum.write(f"Test run @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_sum.write(
                f"agent={orch.agent_name} "
                f"primary={orch.llm_service.model} "
                f"secondary={orch.secondary_service.model}\n\n"
            )
            for idx in wanted:
                if idx < 1 or idx > len(queries):
                    continue
                query = queries[idx - 1]
                print(f"[{idx:>3}/{len(queries)}] {query[:70]}{'…' if len(query) > 70 else ''}")
                rec = await run_one(orch, idx, query, timeout_sec=args.timeout)
                results.append(rec)
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_jsonl.flush()
                line = format_summary_line(rec)
                f_sum.write(line + "\n\n")
                f_sum.flush()
                print("    " + line.split("\n")[1].strip())

        # Bucket summary
        by_intent: Dict[str, int] = {}
        cited = 0
        errored = 0
        for r in results:
            by_intent[r.get("intent") or "—"] = by_intent.get(r.get("intent") or "—", 0) + 1
            if r.get("source_tags"):
                cited += 1
            if r.get("error"):
                errored += 1

        with open(summary_path, "a", encoding="utf-8") as f_sum:
            f_sum.write("\n--- TOTALS ---\n")
            f_sum.write(f"Total queries: {len(results)}\n")
            f_sum.write(f"By intent:     {by_intent}\n")
            f_sum.write(f"Answers with citations: {cited}\n")
            f_sum.write(f"Errors:        {errored}\n")

        print()
        print(f"Done. {len(results)} queries processed.")
        print(f"  By intent: {by_intent}")
        print(f"  With citations: {cited}")
        print(f"  Errors:    {errored}")
        print(f"  JSONL:     {out_path}")
        print(f"  Summary:   {summary_path}")

    asyncio.run(go())


if __name__ == "__main__":
    main()
