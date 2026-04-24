"""
test_multi_user.py — Concurrent multi-user session test with token cost analysis.

Spawns N concurrent users, each sending M queries to the live /chat/stream endpoint.
Measures:
- Per-user: input tokens, output tokens, total tokens, latency, query count
- Aggregate: total tokens, total latency, concurrent throughput (queries/sec)
- Success/failure rate per user
- Token distribution across users

Usage:
    cd /home/vpa/CogOpsCB && python tests/test_multi_user.py --users 5 --queries 3
    cd /home/vpa/CogOpsCB && python tests/test_multi_user.py --users 2 5 10 --queries 2
"""

import asyncio
import json
import time
import argparse
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_API_URL = "http://localhost:9000"
DEFAULT_DEBUG_KEY = "SuperDebugCoTCB"

# Bangla queries covering different government service topics
QUERIES = [
    "পাসপোর্ট এর ফি কত",
    "জন্মসনদ এর ফি কত",
    "এনআইডি ডুপ্লিকেট সনদ",
    "মৃত্যু সনদ এর ফি কত",
    "ড্রাইভিং লাইসেন্স এর নিয়ম",
    "পাসপোর্ট নবায়ন প্রক্রিয়া",
    "জন্মসনদ এর প্রয়োজনীয় কাগজপত্র",
    "এনআইডি সংশোধন পদ্ধতি",
    "মোটর ভেহিকেল রিজিস্ট্রেশন ফি",
    "পাসপোর্ট অফিস এর ঠিকানা",
]


@dataclass
class UserResult:
    user_id: str
    queries_sent: int = 0
    queries_completed: int = 0
    queries_failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    token_records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


async def send_query(
    session: aiohttp.ClientSession,
    api_url: str,
    user_id: str,
    query: str,
    debug_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a single query and return (tokens, latency_ms, chunks_count, raw_response)."""
    headers = {"Content-Type": "application/json"}
    # Always use debug mode so we can capture usage events on the stream
    headers["X-Debug-Key"] = debug_key

    start = time.perf_counter()
    total_input = 0
    total_output = 0
    total_tokens = 0
    chunks = 0
    raw_response = ""

    try:
        async with session.post(
            f"{api_url}/chat/stream",
            json={"user_id": user_id, "query": query},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            raw_response = ""
            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunks += 1

                # Capture usage emitted by the reasoning_loop as a "usage" event
                if evt.get("type") == "usage" and "tokens" in evt:
                    u = evt["tokens"]
                    total_input = u.get("prompt_tokens", 0)
                    total_output = u.get("completion_tokens", 0)
                    total_tokens = u.get("total_tokens", 0)

                raw_response += line

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "chunks": chunks,
            "raw_response_len": len(raw_response),
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": latency_ms,
            "chunks": 0,
            "raw_response_len": 0,
            "error": str(e),
        }


async def run_user(
    session: aiohttp.ClientSession,
    api_url: str,
    user_id: str,
    queries: List[str],
    debug_key: Optional[str],
) -> UserResult:
    """Run all queries for a single user, returning aggregated results."""
    result = UserResult(user_id=user_id)
    for i, query in enumerate(queries):
        result.queries_sent += 1
        res = await send_query(session, api_url, user_id, query, debug_key)

        if "error" in res:
            result.queries_failed += 1
            result.errors.append(f"Q{i+1}: {res['error']}")
        else:
            result.queries_completed += 1
            result.total_input_tokens += res["input_tokens"]
            result.total_output_tokens += res["output_tokens"]
            result.total_tokens += res["total_tokens"]
            result.total_latency_ms += res["latency_ms"]
            result.latencies.append(res["latency_ms"])
            result.token_records.append({
                "query": query[:50],
                "input_tokens": res["input_tokens"],
                "output_tokens": res["output_tokens"],
                "total_tokens": res["total_tokens"],
                "latency_ms": round(res["latency_ms"], 1),
                "chunks": res["chunks"],
            })
        # Small stagger per query within a user to avoid self-collision
        await asyncio.sleep(0.05)
    return result


async def run_concurrent_test(
    api_url: str,
    num_users: int,
    num_queries: int,
    debug_key: Optional[str],
) -> List[UserResult]:
    """Run the full concurrent test: all users start simultaneously."""
    user_queries = [QUERIES[:num_queries] for _ in range(num_users)]
    user_ids = [f"user_{i+1:02d}" for i in range(num_users)]

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Launch all users concurrently
        tasks = [
            run_user(session, api_url, uid, uq, debug_key)
            for uid, uq in zip(user_ids, user_queries)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


def print_report(results: List[UserResult]):
    """Print a comprehensive token cost and performance report."""
    n = len(results)
    total_queries = sum(r.queries_sent for r in results)
    completed = sum(r.queries_completed for r in results)
    failed = sum(r.queries_failed for r in results)

    print(f"\n{'='*120}")
    print(f"MULTI-USER CONCURRENT TEST REPORT")
    print(f"{'='*120}")
    print(f"Users: {n} | Queries per user: {results[0].queries_sent if n else 0} | "
          f"Total queries: {total_queries} | Completed: {completed} | Failed: {failed}")

    # Per-user breakdown
    print(f"\n{'='*120}")
    print(f"PER-USER BREAKDOWN")
    print(f"{'='*120}")
    print(f"{'User':<10} {'Queries':<10} {'Done':<6} {'Fail':<6} {'In Toks':<10} {'Out Toks':<10} {'Total Toks':<11} {'Avg Lat(ms)':<12} {'Q/s':<6}")
    print("-" * 120)

    user_stats = []
    for r in results:
        avg_lat = r.total_latency_ms / r.queries_completed if r.queries_completed > 0 else 0
        throughput = r.queries_completed / (r.total_latency_ms / 1000) if r.total_latency_ms > 0 else 0
        user_stats.append({
            "user": r.user_id,
            "queries": r.queries_sent,
            "done": r.queries_completed,
            "fail": r.queries_failed,
            "input": r.total_input_tokens,
            "output": r.total_output_tokens,
            "total": r.total_tokens,
            "avg_lat": avg_lat,
            "throughput": throughput,
        })
        print(f"{r.user_id:<10} {r.queries_sent:<10} {r.queries_completed:<6} {r.queries_failed:<6} "
              f"{r.total_input_tokens:<10} {r.total_output_tokens:<10} {r.total_tokens:<11} "
              f"{avg_lat:<12.1f} {throughput:<6.2f}")

    # Aggregate
    agg_input = sum(u["input"] for u in user_stats)
    agg_output = sum(u["output"] for u in user_stats)
    agg_total = sum(u["total"] for u in user_stats)
    all_lats = [l for r in results for l in r.latencies]
    total_time = max(sum(r.total_latency_ms for r in results) / 1000, 0.001)

    print(f"\n{'='*120}")
    print(f"AGGREGATE METRICS")
    print(f"{'='*120}")
    print(f"Total input tokens:   {agg_input:>10,}")
    print(f"Total output tokens:  {agg_output:>10,}")
    print(f"Total tokens:         {agg_total:>10,}")
    print(f"Per-query avg tokens: {agg_total // total_queries if total_queries else 0:>10,}")
    print(f"Per-query avg input:  {agg_input // total_queries if total_queries else 0:>10,}")
    print(f"Per-query avg output: {agg_output // total_queries if total_queries else 0:>10,}")
    print(f"Total wall-clock time: {total_time:>10.1f}s")
    print(f"Overall throughput:   {total_queries / total_time:>10.2f} queries/sec")
    print(f"Avg latency/query:    {sum(all_lats)/len(all_lats):>10.1f} ms" if all_lats else "N/A")
    print(f"Min latency/query:    {min(all_lats):>10.1f} ms" if all_lats else "N/A")
    print(f"Max latency/query:    {max(all_lats):>10.1f} ms" if all_lats else "N/A")
    print(f"Median latency/query: {sorted(all_lats)[len(all_lats)//2]:>10.1f} ms" if all_lats else "N/A")

    # Token cost analysis
    print(f"\n{'='*120}")
    print(f"TOKEN COST DISTRIBUTION (per query)")
    print(f"{'='*120}")
    print(f"{'User':<10} {'Query':<40} {'In Toks':<10} {'Out Toks':<10} {'Total Toks':<11} {'Lat(ms)':<10}")
    print("-" * 120)
    for r in results:
        for rec in r.token_records:
            print(f"{r.user_id:<10} {rec['query']:<40} {rec['input_tokens']:<10} "
                  f"{rec['output_tokens']:<10} {rec['total_tokens']:<11} {rec['latency_ms']:<10.1f}")

    # Errors
    all_errors = []
    for r in results:
        all_errors.extend(r.errors)
    if all_errors:
        print(f"\n{'='*120}")
        print(f"ERRORS ({len(all_errors)} total)")
        print(f"{'='*120}")
        for e in all_errors:
            print(f"  {e}")

    # Scaling analysis placeholder
    print(f"\n{'='*120}")
    print(f"COST INSIGHTS")
    print(f"{'='*120}")
    print(f"Each query consumes ~{agg_total // total_queries} tokens (input: {agg_input // total_queries}, output: {agg_output // total_queries})")
    print(f"With {n} concurrent users, the LLM processes {agg_total} tokens total in {total_time:.1f}s wall-clock time.")
    print(f"Concurrent users do NOT multiply token count — each user has independent conversations.")
    print(f"Token count scales with NUMBER OF QUERIES, not concurrency level.")
    print(f"The main impact of concurrency is LLM API load (queries/sec throughput) and latency p99.")

    print(f"\n{'='*120}\n")


async def main():
    parser = argparse.ArgumentParser(description="Multi-user concurrent session test")
    parser.add_argument("--users", nargs="+", type=int, default=[2, 5, 10],
                        help="Number of concurrent users to test (default: 2 5 10)")
    parser.add_argument("--queries", type=int, default=3,
                        help="Queries per user (default: 3)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--debug-key", default=DEFAULT_DEBUG_KEY, help="Debug key header value")
    parser.add_argument("--single", action="store_true",
                        help="Run only the first --users value")
    args = parser.parse_args()

    api_url = args.api_url
    debug_key = args.debug_key
    user_counts = [args.users[0]] if args.single else args.users

    all_reports = []
    for n in user_counts:
        print(f"\n{'#'*60}")
        print(f"  Running test with {n} concurrent users, {args.queries} queries each")
        print(f"{'#'*60}")

        start_wall = time.perf_counter()
        results = await run_concurrent_test(api_url, n, args.queries, debug_key)
        wall_time = time.perf_counter() - start_wall

        print(f"Test completed in {wall_time:.1f}s\n")
        print_report(results)
        all_reports.append((n, results, wall_time))

    # Scaling summary
    if len(all_reports) > 1:
        print(f"\n{'='*120}")
        print(f"SCALING SUMMARY")
        print(f"{'='*120}")
        print(f"{'Users':<8} {'Wall(s)':<10} {'Q/s':<10} {'Total Toks':<13} {'Avg Lat(ms)':<13}")
        print("-" * 65)
        for n_users, res, wt in all_reports:
            total_q = sum(r.queries_completed for r in res)
            qps = total_q / wt if wt > 0 else 0
            total_t = sum(r.total_tokens for r in res)
            avg_l = (sum(r.total_latency_ms for r in res) / max(sum(r.queries_completed for r in res), 1))
            print(f"{n_users:<8} {wt:<10.1f} {qps:<10.2f} {total_t:<13,} {avg_l:<13.1f}")
        print(f"\n{'='*120}\n")


if __name__ == "__main__":
    asyncio.run(main())
