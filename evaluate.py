"""evaluate.py — CLI for evaluating queries against the CogOpsCB Orchestrator.

Usage:
    python evaluate.py --mode interactive
    python evaluate.py --mode batch --csv evaluation/query.csv
    python evaluate.py --mode batch --csv evaluation/query.csv --start 0 --end 50
    python evaluate.py --mode batch --csv evaluation/query.csv --output reports/
"""
import argparse
import asyncio
import csv
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_BDT = timezone(timedelta(hours=6))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _now_bdt() -> str:
    return datetime.now(_BDT).isoformat()


# ---------------------------------------------------------------------------
# Evaluation helper (inlined from deleted cogops/evaluation/evaluator.py)
# ---------------------------------------------------------------------------

async def evaluate_single(
    query: str,
    config_path: str = "configs/config.yml",
) -> Dict[str, Any]:
    """Run a single query through the Orchestrator and collect every event."""
    logger.info(f"Evaluating: {query[:80]}...")
    from cogops.agents.orchestrator import Orchestrator

    query_id = f"eval_{uuid.uuid4().hex[:8]}"
    record: Dict[str, Any] = {
        "query_id": query_id,
        "user_query": query,
        "started_at": _now_bdt(),
        "finished_at": None,
        "events": [],
        "reasoning": [],
        "tool_calls": [],
        "tool_results": [],
        "final_response": "",
        "clarification": None,
        "error": None,
    }

    o = Orchestrator(config_path=config_path)
    user_id = query_id
    full_answer_accumulator: List[str] = []
    clarification_data: Optional[Dict] = None

    async for event in o.process_query(query, debug_mode=True, user_id=user_id):
        event_type = event.get("type", "unknown")
        record["events"].append(event)

        if event_type == "reasoning_chunk":
            record["reasoning"].append(event.get("data", ""))
        elif event_type == "tool_call":
            for tc in event.get("tool_calls", []):
                record["tool_calls"].append({
                    "tool_name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                    "call_id": tc.get("id", ""),
                    "turn": event.get("turn"),
                })
        elif event_type == "tool_result":
            record["tool_results"].append({
                "call_id": event.get("call_id", ""),
                "content": event.get("content", ""),
                "duration_ms": event.get("duration_ms"),
                "status": event.get("status"),
                "turn": event.get("turn"),
            })
        elif event_type == "answer_chunk":
            full_answer_accumulator.append(event.get("content", ""))
        elif event_type == "clarification_needed":
            clarification_data = {
                "question": event.get("question", ""),
                "options": event.get("options", []),
                "reason": event.get("reason", ""),
                "turn_id": event.get("turn_id", ""),
            }

    record["finished_at"] = _now_bdt()
    record["final_response"] = "".join(full_answer_accumulator).strip()
    record["clarification"] = clarification_data
    tool_names_used = list({tc["tool_name"] for tc in record["tool_calls"]})
    record["tool_names_used"] = tool_names_used
    record["num_turns"] = sum(
        1 for e in record["events"] if e.get("type") == "turn_start"
    )

    logger.info(
        f"Done: {len(record['tool_calls'])} tool calls, "
        f"{record['num_turns']} turns, "
        f"final response length={len(record['final_response'])}"
    )
    return record


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

async def run_interactive(config_path: str):
    print("\n=== CogOpsCB Evaluation — Interactive Mode ===")
    print("Type a query and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        print(f"\n--- Evaluating: {query[:100]}...\n")
        t0 = time.time()
        record = await evaluate_single(query, config_path=config_path)
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"Query:    {record['user_query']}")
        print(f"Turns:    {record['num_turns']}")
        print(f"Tools:    {', '.join(record.get('tool_names_used', [])) or 'none'}")
        print(f"Elapsed:  {elapsed:.1f}s")
        if record.get("clarification"):
            print(f"Clarify:  {record['clarification']['question']}")
            print(f"Options:  {record['clarification']['options']}")
        print(f"\nResponse:\n{record['final_response']}")
        print(f"{'='*60}\n")

        out = input("Save to JSON? (y/n) ").strip().lower()
        if out == "y":
            ts = datetime.now(_BDT).strftime("%Y%m%d_%H%M%S")
            path = Path(f"evaluation/output_{ts}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved to {path}\n")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _read_csv_queries(csv_path: str) -> List[str]:
    queries: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col_idx = 0
        for i, h in enumerate(header):
            if h.strip().lower() == "user_query":
                col_idx = i
                break
        for row in reader:
            if row and row[col_idx].strip():
                queries.append(row[col_idx].strip())
    return queries


async def run_batch(
    csv_path: str,
    output_dir: str,
    start: int = 0,
    end: Optional[int] = None,
    config_path: str = "configs/config.yml",
):
    queries = _read_csv_queries(csv_path)
    if not queries:
        logger.error("No queries found in CSV.")
        return

    queries = queries[start:end]
    total = len(queries)
    logger.info(f"Batch mode: {total} queries (start={start}, end={end or total}), output={output_dir}")

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    started_at = _now_bdt()
    completed = 0
    failed = 0
    errors: List[Dict] = []

    for idx, query in enumerate(queries):
        qnum = start + idx + 1
        t0 = time.time()
        try:
            record = await evaluate_single(query, config_path=config_path)
            elapsed = time.time() - t0
            completed += 1
            logger.info(
                f"[{qnum}/{total}] Done in {elapsed:.1f}s — "
                f"tools={record['tool_names_used']}, turns={record['num_turns']}, "
                f"response_len={len(record['final_response'])}"
            )
        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            logger.error(f"[{qnum}/{total}] Failed in {elapsed:.1f}s: {e}")
            errors.append({"query_index": qnum, "query": query[:200], "error": str(e)})
            record = {
                "query_id": f"eval_{qnum:04d}",
                "user_query": query,
                "started_at": _now_bdt(),
                "finished_at": _now_bdt(),
                "events": [],
                "reasoning": [],
                "tool_calls": [],
                "tool_results": [],
                "final_response": "",
                "clarification": None,
                "error": str(e),
                "tool_names_used": [],
                "num_turns": 0,
                "elapsed_s": round(elapsed, 2),
            }

        qid = record.get("query_id", f"q{qnum:04d}")
        safe_query = "".join(c if c.isalnum() or c in " _-।. " else "" for c in query)[:60]
        report_file = report_dir / f"{qid}_{safe_query}.json"
        record["elapsed_s"] = round(time.time() - t0, 2)
        record["query_index"] = qnum
        report_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "batch_started_at": started_at,
        "batch_finished_at": _now_bdt(),
        "csv_file": csv_path,
        "total_queries": total,
        "completed": completed,
        "failed": failed,
        "errors": errors,
        "config_path": config_path,
        "output_dir": str(report_dir),
    }
    summary_file = report_dir / "batch_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {completed}/{total} succeeded, {failed} failed")
    logger.info(f"Reports in: {report_dir}")
    logger.info(f"Summary: {summary_file}")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CogOpsCB Evaluation Suite — run queries through the Orchestrator.",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Run mode: interactive (one query) or batch (from CSV).",
    )
    parser.add_argument(
        "--csv",
        default="evaluation/query.csv",
        help="Path to CSV file with user_query column (batch mode).",
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports",
        help="Directory to save report JSON files (batch mode, default: evaluation/reports).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) for batch mode. Useful for resuming.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) for batch mode. Default: run to end.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yml",
        help="Path to config.yml.",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        asyncio.run(run_interactive(config_path=args.config))
    else:
        asyncio.run(run_batch(
            csv_path=args.csv,
            output_dir=args.output,
            start=args.start,
            end=args.end,
            config_path=args.config,
        ))


if __name__ == "__main__":
    main()
