"""test_api_integration.py - Phase 3: FastAPI endpoint integration tests (mocked)."""
import sys
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from contextlib import ExitStack

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def _async_gen(events_list):
    """Wrap a list of event dicts into an async generator."""
    async def gen():
        for e in events_list:
            yield e
            await asyncio.sleep(0)
    return gen()


def _async_run(coro):
    """Run a coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        return loop.run_until_complete(coro)
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        for prefix in ("cogops.agents", "cogops.session", "cogops.llm", "cogops.tools",
                       "cogops.prompts", "cogops.config", "cogops.events", "api"):
            if mod.startswith(prefix):
                del sys.modules[mod]
    yield
    try:
        import api
        api.active_sessions.clear()
    except Exception:
        pass


@pytest.fixture
def config_path(tmp_path):
    cfg = {
        "agent_name": "TestAgent", "agent_story": "A test agent.",
        "llm": {"api_key_env": "LLM_API_KEY", "model_name_env": "LLM_MODEL_NAME",
                "base_url_env": "LLM_BASE_URL", "max_context_tokens": 32000, "thinking": True},
        "reranker": {"api_key_env": "RERANKER_API_KEY", "model_name_env": "RERANKER_MODEL_NAME",
                     "base_url_env": "RERANKER_BASE_URL", "max_context_tokens": 32000},
        "secondary": {"api_key_env": "SECONDARY_API_KEY", "model_name_env": "SECONDARY_MODEL_NAME",
                      "base_url_env": "SECONDARY_BASE_URL", "max_context_tokens": 32000},
        "embedder": {"url_env": "TRITON_URL", "model_name_env": "TRITON_MODEL_NAME",
                     "tokenizer_env": "TRITON_TOKENIZER", "max_batch_size": 8},
        "reasoning": {"max_turns": 10}, "conversation": {"history_window": 10},
        "graphiti": {
            "search": {"limit": 5, "min_score": 0.8}, "entity_search": {"max_results": 10},
            "entity_detail": {}, "node_explore": {"max_results": 100},
            "relation_browse": {"filter_prefix": None, "top_n": 100},
            "relation_filter": {"max_results": 50},
            "similar_entities": {"max_results": 10, "min_score": 0.5},
            "path_find": {"max_hops": 3, "max_paths": 5},
            "episodic_search": {"max_results": 10}, "graph_stats": {"detail_level": "basic"},
        },
        "session": {"redis_url_env": "REDIS_URL", "ttl_seconds_env": "REDIS_SESSION_TTL_SECONDS"},
        "summarizer": {"max_tokens_env": "SUMMARIZER_MAX_TOKENS"},
        "token_management": {"tokenizer_model_env": "TOKENIZER_MODEL_NAME",
                             "system_prompt_reservation": 3500, "history_budget": 0.3},
        "llm_call_parameters": {
            "thinking_general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
            "instruct_general": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
            "instruct_reasoning": {"temperature": 1.0, "top_p": 1.0, "top_k": 40,
                                   "min_p": 0.0, "presence_penalty": 2.0, "repetition_penalty": 1.0},
            "max_tokens": 2048,
        },
        "response_templates": {"error_fallback": "Error fallback.", "tool_failure": "Tool failure.",
                               "no_info_found": "No info found.", "safety_violation": "Safety violation."},
    }
    p = tmp_path / "config.yml"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


class TestHealthEndpoint:
    def test_health_returns_online(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get("/health")
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "online"
            assert data["service"] == "GovOps Graphiti"
            assert "active_sessions" in data

    def test_health_returns_session_count(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.get("/health")
                return resp
            resp =  _async_run(run())
            assert resp.json()["active_sessions"] >= 0


class TestChatStreamEndpoint:
    """POST /chat/stream - channel filtering."""

    @staticmethod
    def _mock_init(config_path, events):
        """Return a ContextManager that patches Orchestrator init + process_query."""
        from cogops.agents.orchestrator import Orchestrator
        mock_svc = MagicMock()
        mock_svc.client_llm = AsyncMock()
        mock_svc.client_secondary = None
        mock_svc.llm_config = None
        mock_svc.model = ""
        mock_tok = MagicMock()
        mock_tok.count = MagicMock(return_value=10)

        async def _process(query, debug_mode):
            for e in events:
                yield e
                await asyncio.sleep(0)

        stack = ExitStack()
        stack.enter_context(patch.multiple(
            "cogops.agents.orchestrator",
            AsyncLLMService=MagicMock(return_value=mock_svc),
            get_graph_prompt=MagicMock(return_value="mock"),
            Tokenizer=MagicMock(return_value=mock_tok),
            build_tool_registry=MagicMock(return_value=([], {})),
            RedisSessionStore=MagicMock(),
        ))
        stack.enter_context(patch.object(Orchestrator, "process_query", side_effect=_process))
        return stack

    def test_user_channel_only_without_debug_key(self, config_path):
        events = [
            {"type": "turn_start", "turn_number": 1, "channel": "debug"},
            {"type": "reasoning_chunk", "data": "thinking...", "channel": "debug"},
            {"type": "tool_call", "name": "graph_search", "channel": "debug"},
            {"type": "tool_result", "call_id": "1", "channel": "debug"},
            {"type": "answer_chunk", "content": "Fee is 3000 Taka.", "channel": "both"},
        ]
        with (
            patch("api.AGENT_CONFIG_PATH", config_path),
            self._mock_init(config_path, events),
        ):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/chat/stream", json={"user_id": "u1", "query": "passport fee"})
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 200
            lines = resp.text.strip().split("\n")
            parsed = [json.loads(l) for l in lines]
            types = [e["type"] for e in parsed]
            assert "reasoning_chunk" not in types
            assert "tool_call" not in types
            assert "tool_result" not in types
            assert "answer_chunk" in types

    def test_debug_channel_includes_debug_events(self, config_path):
        events = [
            {"type": "turn_start", "turn_number": 1, "channel": "debug"},
            {"type": "reasoning_chunk", "data": "thinking...", "channel": "debug"},
            {"type": "tool_call", "name": "graph_search", "channel": "debug"},
            {"type": "tool_result", "call_id": "1", "channel": "debug"},
            {"type": "answer_chunk", "content": "Fee is 3000 Taka.", "channel": "both"},
        ]
        with (
            patch("api.AGENT_CONFIG_PATH", config_path),
            self._mock_init(config_path, events),
        ):
            import api
            api.os.environ["ADMIN_DEBUG_SECRET"] = "test-secret"
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/chat/stream",
                        json={"user_id": "u1", "query": "passport fee"},
                        headers={"X-Debug-Key": "test-secret"},
                    )
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 200
            lines = resp.text.strip().split("\n")
            parsed = [json.loads(l) for l in lines]
            types = [e["type"] for e in parsed]
            assert "reasoning_chunk" in types
            assert "tool_call" in types
            assert "turn_start" in types
            assert "answer_chunk" in types


class TestChatEndpoint:
    def test_empty_query_returns_400(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/chat/stream", json={"user_id": "u1", "query": "  "})
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 400

    def test_query_with_content(self, config_path):
        events = [{"type": "answer_chunk", "content": "OK", "channel": "both"}]
        with (
            patch("api.AGENT_CONFIG_PATH", config_path),
            TestChatStreamEndpoint._mock_init(config_path, events),
        ):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/chat/stream", json={"user_id": "u1", "query": "hello"})
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 200


class TestFeedbackEndpoint:
    def test_feedback_success_with_active_session(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            # Create a session first
            with (
                patch("cogops.agents.orchestrator.AsyncLLMService") as mock_svc,
                patch("cogops.agents.orchestrator.get_graph_prompt", return_value="mock"),
                patch("cogops.agents.orchestrator.Tokenizer"),
                patch("cogops.agents.orchestrator.build_tool_registry", return_value=([], {})),
                patch("cogops.agents.orchestrator.RedisSessionStore"),
            ):
                mock_svc.return_value.client_llm = AsyncMock()
                mock_svc.return_value.client_secondary = None
                mock_svc.return_value.llm_config = None
                mock_svc.return_value.model = ""
                # Trigger session creation
                transport = ASGITransport(app=api.app)
                async def run():
                    async with AsyncClient(transport=transport, base_url="http://test") as ac:
                        await ac.post("/chat/stream", json={"user_id": "u1", "query": "hi"})
                    return ac
                _async_run(run())

            transport = ASGITransport(app=api.app)
            async def run_fb():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/feedback", json={
                        "user_id": "u1", "turn_id": "t1", "rating": "bad", "comment": "too long"
                    })
                return resp
            resp =  _async_run(run_fb())
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"

    def test_feedback_no_session(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            transport = ASGITransport(app=api.app)
            async def run_fb():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/feedback", json={
                        "user_id": "nouser", "turn_id": "t1", "rating": "good"
                    })
                return resp
            resp =  _async_run(run_fb())
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ignored"


class TestSessionClearEndpoint:
    def test_clear_existing_session(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            with (
                patch("cogops.agents.orchestrator.AsyncLLMService") as mock_svc,
                patch("cogops.agents.orchestrator.get_graph_prompt", return_value="mock"),
                patch("cogops.agents.orchestrator.Tokenizer"),
                patch("cogops.agents.orchestrator.build_tool_registry", return_value=([], {})),
                patch("cogops.agents.orchestrator.RedisSessionStore"),
            ):
                mock_svc.return_value.client_llm = AsyncMock()
                mock_svc.return_value.client_secondary = None
                mock_svc.return_value.llm_config = None
                mock_svc.return_value.model = ""

            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp1 = await ac.post("/chat/stream", json={"user_id": "u1", "query": "hi"})
                    resp2 = await ac.post("/session/clear", json={"user_id": "u1"})
                return resp2
            resp =  _async_run(run())
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"

    def test_clear_nonexistent_session(self, config_path):
        with patch("api.AGENT_CONFIG_PATH", config_path):
            import api
            transport = ASGITransport(app=api.app)
            async def run():
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/session/clear", json={"user_id": "ghost"})
                return resp
            resp =  _async_run(run())
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ignored"
