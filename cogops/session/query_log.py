"""query_log.py — Persistent log of incoming queries with timestamps.

Stores only: the query text and the ISO-8601 timestamp of when it arrived.
Rolling 10-day window: entries older than 10 days are pruned on every write.
Data is written to a single JSONL file so appends are cheap and truncation
is safe.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from cogops.config.loader import load_config

# Default: CogOpsCB/data/query_log.jsonl (created if missing)
_DEFAULT_PATH = "data/query_log.jsonl"
_DEFAULT_MAX_AGE_DAYS = 10

# Bangladesh Standard Time (UTC+6)
_BDT = timezone(timedelta(hours=6))


def _now_bdt() -> datetime:
    """Current time in Bangladesh timezone."""
    return datetime.now(_BDT)


class QueryLog:
    """Append-only query log with automatic 10-day pruning."""

    def __init__(self, path: Optional[str] = None):
        cfg = load_config()
        session_cfg = cfg.get("session", {})
        env_path = os.getenv("QUERY_LOG_PATH")
        self.path = Path(path or env_path or session_cfg.get("query_log_path", _DEFAULT_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._max_age_days = (
            session_cfg.get("query_log_retention_days", _DEFAULT_MAX_AGE_DAYS)
            if path is None and env_path is None
            else _DEFAULT_MAX_AGE_DAYS
        )

    def append(self, query: str) -> None:
        """Record a query with the current Bangladesh timestamp."""
        entry = json.dumps({
            "query": query,
            "timestamp": _now_bdt().isoformat(),
        }, ensure_ascii=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        # Prune after every append so stale data never accumulates.
        self._prune()

    @property
    def entries(self) -> list[dict]:
        """Return all entries within the 10-day window (prunes silently)."""
        cutoff = _now_bdt() - timedelta(days=self._max_age_days)
        results: list[dict] = []
        if not self.path.exists():
            return results
        with open(self.path, "r", encoding="utf-8") as f:
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
        return results

    def count(self) -> int:
        """Number of entries within the retention window."""
        return len(self.entries)

    def _prune(self) -> None:
        """Rewrite the file keeping only entries within the retention window."""
        cutoff = _now_bdt() - timedelta(days=self._max_age_days)
        kept: list[str] = []
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
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
        with open(self.path, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
