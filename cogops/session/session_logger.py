"""session_logger.py — Persistent audit log of full agent interactions.

Stores the complete interaction trace per session:
- incoming user query
- all streaming events (tool calls, reasoning chunks, answers)
- final answer text
- total duration in milliseconds

Written as JSONL per user_id so the file can be read by scripts that
generate reports. Each session is a single JSONL entry containing all
collected data.
"""
import asyncio
import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from cogops.config.loader import load_config

_BDT = timezone(timedelta(hours=6))
_DEFAULT_PATH = "data/session_traces.jsonl"
logger = logging.getLogger(__name__)


def _now_bdt() -> datetime:
    return datetime.now(_BDT)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionLogger:
    """Collects streaming events per user_id and writes a structured trace."""

    def __init__(self, path: Optional[str] = None):
        cfg = load_config()
        env_path = os.getenv("SESSION_TRACE_PATH")
        self.path = Path(path or env_path or cfg.get("session", {})
                           .get("query_log_path", _DEFAULT_PATH).replace(
                               "query_log.jsonl", "session_traces.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffers: Dict[str, Dict[str, Any]] = {}

    def start_session(self, user_id: str, query: str) -> str:
        """Mark the beginning of a session. Returns a session_id."""
        session_id = f"{user_id}_{int(time.time())}"
        self._buffers[user_id] = {
            "user_id": user_id,
            "query": query,
            "session_id": session_id,
            "events": [],
            "tool_calls": [],
            "reasoning_chunks": [],
            "answer_chunks": [],
            "tool_results": [],
            "start_time": _now_utc_iso(),
            "turn_ids": [],
        }
        logger.info(f"SessionLogger: started session {session_id} for user {user_id}")
        return session_id

    def ingest_event(self, event: Dict[str, Any]) -> None:
        """Ingest a single streaming event from the generator."""
        user_id = None
        # Walk through all open buffers to find the one to update
        for uid, buf in self._buffers.items():
            # We only track the last started session per user
            pass

        # Find the most recent buffer (the one currently being processed)
        if not self._buffers:
            return

        buf = list(self._buffers.values())[-1]
        etype = event.get("type", "")
        channel = event.get("channel", "user")

        buf["events"].append(event)

        if etype == "tool_call":
            buf["tool_calls"].append({
                "tool_name": event.get("tool_calls", [{}])[0].get("function", {}).get("name", "?"),
                "arguments": event.get("tool_calls", [{}])[0].get("function", {}).get("arguments", ""),
                "turn": event.get("turn", "?"),
            })
        elif etype == "tool_result":
            buf["tool_results"].append({
                "name": event.get("name", "?"),
                "status": event.get("status", "?"),
                "preview": event.get("preview", "")[:200] if event.get("preview") else "",
                "sources": event.get("sources", []),
            })
        elif etype == "reasoning_chunk":
            buf["reasoning_chunks"].append(event.get("content", ""))
        elif etype == "answer_chunk":
            buf["answer_chunks"].append(event.get("content", ""))
        elif etype == "answer_complete":
            buf["turn_ids"].append(event.get("turn_id", "?"))

    def finalize_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Close a session buffer and write it to the JSONL file. Returns the trace."""
        buf = self._buffers.pop(user_id, None)
        if not buf:
            return None

        buf["end_time"] = _now_utc_iso()
        buf["total_answer"] = "".join(buf["answer_chunks"])
        buf["total_reasoning"] = "".join(buf["reasoning_chunks"])
        buf["event_count"] = len(buf["events"])
        buf["tool_call_count"] = len(buf["tool_calls"])
        buf["tool_result_count"] = len(buf["tool_results"])

        # Write to JSONL
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(buf, ensure_ascii=False) + "\n")
            logger.info(
                f"SessionLogger: finalized {buf['session_id']} — "
                f"{buf['tool_call_count']} tool calls, "
                f"{len(buf['answer_chunks'])} answer chunks, "
                f"{len(buf['reasoning_chunks'])} reasoning chunks"
            )
            return buf
        except Exception as e:
            logger.error(f"SessionLogger: failed to write trace for {user_id}: {e}")
            return None

    def get_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent traces from the JSONL file."""
        if not self.path.exists():
            return []
        traces: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
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
        """Clear the buffer (debug only)."""
        self._buffers.clear()
