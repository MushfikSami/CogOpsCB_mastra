"""logger.py — Combined query log and session trace (JSONL fallback).

Replaces the old QueryLog + SessionLogger pair.
If PostgreSQL is available (DATABASE_URL env var), writes to Postgres.
Otherwise falls back to rolling JSONL files.
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    _PSYCOPG2_OK = True
except Exception:
    _PSYCOPG2_OK = False

_BDT = timezone(timedelta(hours=6))
logger = logging.getLogger(__name__)


def _now_bdt_iso() -> str:
    return datetime.now(_BDT).isoformat()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QuerySessionLogger:
    """Unified logger: query log + session traces.

    Uses Postgres if DATABASE_URL is set and psycopg2 is installed,
    otherwise writes to JSONL files under data/.
    """

    def __init__(
        self,
        query_log_path: str = "data/query_log.jsonl",
        session_trace_path: str = "data/session_traces.jsonl",
        max_age_days: int = 10,
    ):
        self.db_url = os.getenv("DATABASE_URL")
        self.use_pg = bool(self.db_url and _PSYCOPG2_OK)

        self.query_log_path = Path(query_log_path)
        self.session_trace_path = Path(session_trace_path)
        self.max_age_days = max_age_days

        # Ensure parent dirs exist
        self.query_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_trace_path.parent.mkdir(parents=True, exist_ok=True)
        for p in (self.query_log_path, self.session_trace_path):
            if not p.exists():
                p.touch()

        self._session_buffers: Dict[str, Dict[str, Any]] = {}

        if self.use_pg:
            self._ensure_pg_tables()

    # ── Postgres helpers ──────────────────────────────────────────────

    def _pg_conn(self):
        return psycopg2.connect(self.db_url)

    def _ensure_pg_tables(self):
        try:
            with self._pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS query_log (
                            id SERIAL PRIMARY KEY,
                            query TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS session_traces (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT UNIQUE NOT NULL,
                            user_id TEXT NOT NULL,
                            query TEXT NOT NULL,
                            trace JSONB NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                    """)
                conn.commit()
        except Exception as e:
            logger.warning("Postgres init failed (%s), falling back to JSONL", e)
            self.use_pg = False

    # ── Query log ─────────────────────────────────────────────────────

    def append_query(self, query: str) -> None:
        if self.use_pg:
            try:
                with self._pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO query_log (query, timestamp) VALUES (%s, NOW())",
                            (query,),
                        )
                    conn.commit()
                return
            except Exception as e:
                logger.warning("PG query log failed: %s", e)

        entry = json.dumps({"query": query, "timestamp": _now_bdt_iso()}, ensure_ascii=False)
        with open(self.query_log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        self._prune(self.query_log_path)

    def get_queries(self, limit: int = 500) -> List[Dict[str, Any]]:
        if self.use_pg:
            try:
                with self._pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT query, timestamp FROM query_log ORDER BY timestamp DESC LIMIT %s",
                            (limit,),
                        )
                        return [
                            {"query": row[0], "timestamp": row[1].isoformat()}
                            for row in cur.fetchall()
                        ]
            except Exception as e:
                logger.warning("PG query read failed: %s", e)

        cutoff = datetime.now(_BDT) - timedelta(days=self.max_age_days)
        results: List[Dict[str, Any]] = []
        if not self.query_log_path.exists():
            return results
        with open(self.query_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts >= cutoff:
                        results.append(entry)
                except (json.JSONDecodeError, KeyError):
                    pass
        return results[-limit:]

    # ── Session traces ────────────────────────────────────────────────

    def start_session(self, user_id: str, query: str) -> str:
        session_id = f"{user_id}_{int(time.time())}"
        self._session_buffers[user_id] = {
            "user_id": user_id,
            "query": query,
            "session_id": session_id,
            "events": [],
            "tool_calls": [],
            "tool_results": [],
            "reasoning_chunks": [],
            "answer_chunks": [],
            "start_time": _now_utc_iso(),
        }
        return session_id

    def ingest_event(self, event: Dict[str, Any]) -> None:
        if not self._session_buffers:
            return
        buf = list(self._session_buffers.values())[-1]
        buf["events"].append(event)

        etype = event.get("type", "")
        if etype == "tool_call":
            for s in event.get("tool_call_summaries", []):
                buf["tool_calls"].append({
                    "call_id": s.get("call_id", ""),
                    "tool_name": s.get("name", "?"),
                    "arguments": s.get("arguments", {}),
                })
        elif etype == "tool_result":
            buf["tool_results"].append({
                "name": event.get("name", "?"),
                "status": event.get("status", "?"),
                "preview": str(event.get("preview", ""))[:200],
            })
        elif etype == "reasoning_chunk":
            buf["reasoning_chunks"].append(event.get("content", ""))
        elif etype == "answer_chunk":
            buf["answer_chunks"].append(event.get("content", ""))

    def finalize_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        buf = self._session_buffers.pop(user_id, None)
        if not buf:
            return None

        buf["end_time"] = _now_utc_iso()
        buf["total_answer"] = "".join(buf["answer_chunks"])
        buf["total_reasoning"] = "".join(buf["reasoning_chunks"])
        buf["event_count"] = len(buf["events"])
        buf["tool_call_count"] = len(buf["tool_calls"])
        buf["tool_result_count"] = len(buf["tool_results"])

        if self.use_pg:
            try:
                with self._pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO session_traces
                               (session_id, user_id, query, trace, created_at)
                               VALUES (%s, %s, %s, %s, NOW())
                               ON CONFLICT (session_id) DO UPDATE
                               SET trace = EXCLUDED.trace, created_at = NOW()
                            """,
                            (buf["session_id"], buf["user_id"], buf["query"], json.dumps(buf)),
                        )
                    conn.commit()
                return buf
            except Exception as e:
                logger.warning("PG session trace write failed: %s", e)

        try:
            with open(self.session_trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(buf, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Session trace write failed: %s", e)
        return buf

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self.use_pg:
            try:
                with self._pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT trace FROM session_traces ORDER BY created_at DESC LIMIT %s",
                            (limit,),
                        )
                        return [row[0] for row in cur.fetchall()]
            except Exception as e:
                logger.warning("PG trace read failed: %s", e)

        if not self.session_trace_path.exists():
            return []
        traces: List[Dict[str, Any]] = []
        with open(self.session_trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return traces[-limit:]

    def clear(self) -> None:
        self._session_buffers.clear()

    # ── Internal helpers ──────────────────────────────────────────────

    def _prune(self, path: Path) -> None:
        cutoff = datetime.now(_BDT) - timedelta(days=self.max_age_days)
        kept: List[str] = []
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts >= cutoff:
                        kept.append(line)
                except (json.JSONDecodeError, KeyError):
                    pass
        with open(path, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
