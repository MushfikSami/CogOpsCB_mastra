"""test_session.py — Phase 2: Redis session store (mocked)."""
import sys
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.session"):
            del sys.modules[mod]
    yield


class TestRedisSessionStore:
    """RedisSessionStore roundtrip and edge cases."""

    def test_store_and_get_turn(self):
        mock_redis = MagicMock()
        mock_redis.lpush = MagicMock()
        mock_redis.expire = MagicMock()
        turn = {"turn_id": "t1", "user": "hello", "assistant": "hi"}
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store._client = mock_redis
            store.ttl = 100
            store.store_turn("u1", turn)
            mock_redis.lpush.assert_called_once()
            mock_redis.expire.assert_called_once()

    def test_get_recent_turns_roundtrip(self):
        """lpush + lrange roundtrip."""
        turn = {"turn_id": "t1", "user": "hello", "assistant": "hi"}
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100

            # Simulate lpush behavior: most recent at index 0
            mock_redis = MagicMock()
            mock_redis.lpush = MagicMock()
            mock_redis.lrange = MagicMock(return_value=[json.dumps(turn)])
            mock_redis.expire = MagicMock()
            store._client = mock_redis

            result = store.get_recent_turns("u1")
            assert len(result) == 1
            assert result[0]["turn_id"] == "t1"

    def test_set_get_summary(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100
            mock_redis = MagicMock()
            mock_redis.set = MagicMock()
            mock_redis.get = MagicMock(return_value="User asked about passport.")
            mock_redis.expire = MagicMock()
            store._client = mock_redis

            store.set_summary("u1", "test summary")
            mock_redis.set.assert_called_once()
            assert store.get_summary("u1") == "User asked about passport."

    def test_get_summary_returns_empty_when_none(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100
            mock_redis = MagicMock()
            mock_redis.get = MagicMock(return_value=None)
            store._client = mock_redis
            assert store.get_summary("u1") == ""

    def test_clarification_roundtrip(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100
            mock_redis = MagicMock()
            mock_redis.set = MagicMock()
            mock_redis.get = MagicMock(return_value=json.dumps({"question": "What?"}))
            mock_redis.delete = MagicMock()
            store._client = mock_redis

            store.set_clarification("u1", {"question": "What?"})
            result = store.get_clarification("u1")
            assert result["question"] == "What?"

    def test_clear_clarification(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100
            mock_redis = MagicMock()
            mock_redis.get = MagicMock(return_value=None)
            mock_redis.delete = MagicMock()
            store._client = mock_redis

            store.clear_clarification("u1")
            mock_redis.delete.assert_called_once()
            assert store.get_clarification("u1") is None

    def test_clear_all(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store.ttl = 100
            mock_redis = MagicMock()
            mock_redis.keys = MagicMock(return_value=["key1", "key2"])
            mock_redis.delete = MagicMock()
            store._client = mock_redis

            store.clear_all("u1")
            mock_redis.delete.assert_called_once()

    def test_unavailable_is_noop(self):
        with patch.dict(sys.modules, {"redis": MagicMock()}):
            from cogops.session.redis_store import RedisSessionStore
            store = RedisSessionStore.__new__(RedisSessionStore)
            store._client = None
            store.ttl = 100

            # Should not raise
            store.store_turn("u1", {"turn_id": "t1", "user": "q", "assistant": "a"})
            assert store.get_recent_turns("u1") == []
            store.set_summary("u1", "s")
            assert store.get_summary("u1") == ""
            store.set_clarification("u1", {"question": "q"})
            assert store.get_clarification("u1") is None
            store.clear_all("u1")
