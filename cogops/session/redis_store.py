"""
cogops/session/redis_store.py

Redis-backed session store for turns and rolling summaries.
Falls back to in-memory store when Redis is unavailable.
"""

import json
import logging
import os
from typing import Optional, List

load_dotenv_imported = False
try:
    from dotenv import load_dotenv
    load_dotenv_imported = True
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class InMemoryStore:
    """Simple in-memory fallback when Redis is unavailable."""

    def __init__(self):
        self._turns: dict[str, List[dict]] = {}
        self._meta: dict[str, dict] = {}
        self._memory: dict[str, str] = {}  # key -> value for memory tools
        self._redis_available = False

    def store_turn(self, user_id: str, turn: dict) -> None:
        self._turns.setdefault(user_id, []).insert(0, turn)

    def get_recent_turns(self, user_id: str, n: int = 5) -> List[dict]:
        return self._turns.get(user_id, [])[:n]

    def clear_turns(self, user_id: str) -> None:
        self._turns.pop(user_id, None)

    def set_last_assistant_meta(self, user_id: str, meta: dict) -> None:
        self._meta[user_id] = meta

    def get_last_assistant_meta(self, user_id: str) -> Optional[dict]:
        return self._meta.get(user_id)

    def clear_all(self, user_id: str) -> None:
        self._turns.pop(user_id, None)
        self._meta.pop(user_id, None)
        # Clean up memory keys for this user
        keys_to_remove = [k for k in self._memory if k.startswith(f"session:{user_id}:memory:")]
        for k in keys_to_remove:
            self._memory.pop(k, None)

    @property
    def available(self) -> bool:
        """InMemoryStore is always 'available' for local operations."""
        return True

    # --- Redis-compatible convenience methods for memory tools ---
    def keys(self, pattern: str) -> List[str]:
        """Return keys matching a glob-style pattern (only * supported)."""
        import fnmatch
        return [k for k in self._memory if fnmatch.fnmatch(k, pattern)]

    def get(self, key: str) -> Optional[str]:
        """Get value by key, or None."""
        return self._memory.get(key)

    def set(self, key: str, value: str) -> None:
        """Set key-value pair."""
        self._memory[key] = value

    def expire(self, key: str, ttl: int) -> None:
        """No-op for in-memory store (no TTL in fallback)."""
        pass


class RedisSessionStore:
    """Redis-backed session store with in-memory fallback."""

    def __init__(self, url: Optional[str] = None, ttl_seconds: Optional[int] = None):
        try:
            import redis as _redis
        except ImportError:
            logger.warning("redis package not installed — using in-memory fallback.")
            self._client = InMemoryStore()
            self._redis_available = False
            return

        self._redis_available = False
        self.ttl = ttl_seconds if ttl_seconds is not None else 86400

        session_cfg = {}
        try:
            from cogops.config.loader import load_config
            session_cfg = load_config().get("session", {})
        except Exception:
            pass

        default_url = session_cfg.get("redis_url_default", "redis://localhost:6379/0")
        url = url or os.getenv(session_cfg.get("redis_url_env", "REDIS_URL"), default_url)

        try:
            self._client = _redis.from_url(url, decode_responses=True)
            self._client.ping()
            self._redis_available = True
            logger.info("Redis connected: %s", url)
        except Exception as e:
            logger.warning("Redis connection failed (%s) — using in-memory fallback.", e)
            self._client = InMemoryStore()

    @property
    def available(self) -> bool:
        return self._redis_available

    def _key(self, user_id: str, suffix: str) -> str:
        return f"session:{user_id}:{suffix}"

    def store_turn(self, user_id: str, turn: dict) -> None:
        if isinstance(self._client, InMemoryStore):
            self._client.store_turn(user_id, turn)
            return
        key = self._key(user_id, "turns")
        self._client.lpush(key, json.dumps(turn))
        self._client.expire(key, self.ttl)

    def get_recent_turns(self, user_id: str, n: int = 5) -> List[dict]:
        if isinstance(self._client, InMemoryStore):
            return self._client.get_recent_turns(user_id, n)
        key = self._key(user_id, "turns")
        raw = self._client.lrange(key, 0, n - 1)
        return [json.loads(r) for r in raw]

    def clear_turns(self, user_id: str) -> None:
        if isinstance(self._client, InMemoryStore):
            self._client.clear_turns(user_id)
            return
        key = self._key(user_id, "turns")
        self._client.delete(key)

    def set_last_assistant_meta(self, user_id: str, meta: dict) -> None:
        if isinstance(self._client, InMemoryStore):
            self._client.set_last_assistant_meta(user_id, meta)
            return
        key = self._key(user_id, "last_assistant_meta")
        self._client.set(key, json.dumps(meta))
        self._client.expire(key, self.ttl)

    def get_last_assistant_meta(self, user_id: str) -> Optional[dict]:
        if isinstance(self._client, InMemoryStore):
            return self._client.get_last_assistant_meta(user_id)
        key = self._key(user_id, "last_assistant_meta")
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def clear_all(self, user_id: str) -> None:
        if isinstance(self._client, InMemoryStore):
            self._client.clear_all(user_id)
            return
        pattern = self._key(user_id, "*")
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)
