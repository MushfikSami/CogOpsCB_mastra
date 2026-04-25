"""
cogops/session/redis_store.py

Redis wrapper for session: raw turns, rolling summary, pending_clarification.
"""

import json
import logging
import os
from typing import Optional, Any, List
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisSessionStore:
    """Redis-backed session store for turns, summaries, and pending clarifications."""

    def __init__(self, url: Optional[str] = None, ttl_seconds: Optional[int] = None):
        if not _REDIS_AVAILABLE:
            logger.warning("redis package not installed — RedisSessionStore is disabled.")
            self._client = None
            return

        cfg = {"session": {}}
        try:
            from cogops.config.loader import load_config
            cfg = load_config()
        except Exception:
            pass
        session_cfg = cfg.get("session", {})

        default_url = session_cfg.get("redis_url_default", "redis://localhost:6379/0")
        default_ttl = session_cfg.get("ttl_default", 86400)

        url = url or os.getenv(session_cfg.get("redis_url_env", "REDIS_URL"), default_url)
        self.ttl = ttl_seconds if ttl_seconds is not None else int(
            os.getenv(session_cfg.get("ttl_seconds_env", "REDIS_SESSION_TTL_SECONDS"), str(default_ttl))
        )
        try:
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info(f"Redis connected: {url}")
        except Exception as e:
            logger.warning(f"Redis connection failed ({e}) — falling back to in-memory store.")
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, user_id: str, suffix: str) -> str:
        return f"session:{user_id}:{suffix}"

    # --- Turns ---
    def store_turn(self, user_id: str, turn: dict) -> None:
        """Append a turn (dict) to the turns list."""
        if not self.available:
            return
        key = self._key(user_id, "turns")
        self._client.lpush(key, json.dumps(turn))
        self._client.expire(key, self.ttl)

    def get_recent_turns(self, user_id: str, n: int = 5) -> List[dict]:
        """Get the last N turns (most recent first in list)."""
        if not self.available:
            return []
        key = self._key(user_id, "turns")
        raw = self._client.lrange(key, 0, n - 1)
        return [json.loads(r) for r in raw]

    def clear_turns(self, user_id: str) -> None:
        if not self.available:
            return
        key = self._key(user_id, "turns")
        self._client.delete(key)

    # --- Rolling Summary ---
    def set_summary(self, user_id: str, summary: str) -> None:
        if not self.available:
            return
        key = self._key(user_id, "summary")
        self._client.set(key, summary)
        self._client.expire(key, self.ttl)

    def get_summary(self, user_id: str) -> str:
        if not self.available:
            return ""
        key = self._key(user_id, "summary")
        return self._client.get(key) or ""

    def clear_summary(self, user_id: str) -> None:
        if not self.available:
            return
        key = self._key(user_id, "summary")
        self._client.delete(key)

    # --- Pending Clarification ---
    def set_clarification(self, user_id: str, q: dict) -> None:
        if not self.available:
            return
        key = self._key(user_id, "clarification")
        self._client.set(key, json.dumps(q))
        self._client.expire(key, self.ttl)

    def get_clarification(self, user_id: str) -> Optional[dict]:
        if not self.available:
            return None
        key = self._key(user_id, "clarification")
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def clear_clarification(self, user_id: str) -> None:
        if not self.available:
            return
        key = self._key(user_id, "clarification")
        self._client.delete(key)

    # --- Last assistant reply (for resolving short follow-ups like "3") ---
    def set_last_assistant_meta(self, user_id: str, meta: dict) -> None:
        """Store the most recent assistant reply + any enumerated options it offered.

        meta = {"assistant_text": str, "options": list[str], "turn_id": str}
        """
        if not self.available:
            return
        key = self._key(user_id, "last_assistant")
        self._client.set(key, json.dumps(meta))
        self._client.expire(key, self.ttl)

    def get_last_assistant_meta(self, user_id: str) -> Optional[dict]:
        if not self.available:
            return None
        key = self._key(user_id, "last_assistant")
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    # --- Cleanup ---
    def clear_all(self, user_id: str) -> None:
        if not self.available:
            return
        pattern = self._key(user_id, "*")
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)
