"""test_orchestrator.py — Phase 2: orchestrator (mocked LLM + Redis)."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.agents"):
            del sys.modules[mod]
        if mod.startswith("cogops.session"):
            del sys.modules[mod]
        if mod.startswith("cogops.llm"):
            del sys.modules[mod]
        if mod.startswith("cogops.tools"):
            del sys.modules[mod]
        if mod.startswith("cogops.prompts"):
            del sys.modules[mod]
    yield
    # Clear the class-level cache so subsequent tests get a fresh one
    try:
        from cogops.agents.orchestrator import Orchestrator
        Orchestrator._cached_system_prompt = None
    except Exception:
        pass


@pytest.fixture
def config_path(tmp_path):
    """Write a minimal config.yml to a temp file."""
    cfg = {
        "agent_name": "TestAgent",
        "agent_story": "A test agent.",
        "llm": {
            "api_key_env": "LLM_API_KEY",
            "model_name_env": "LLM_MODEL_NAME",
            "base_url_env": "LLM_BASE_URL",
            "max_context_tokens": 32000,
            "thinking": True,
        },
        "reranker": {
            "api_key_env": "RERANKER_API_KEY",
            "model_name_env": "RERANKER_MODEL_NAME",
            "base_url_env": "RERANKER_BASE_URL",
            "max_context_tokens": 32000,
        },
        "secondary": {
            "api_key_env": "SECONDARY_API_KEY",
            "model_name_env": "SECONDARY_MODEL_NAME",
            "base_url_env": "SECONDARY_BASE_URL",
            "max_context_tokens": 32000,
        },
        "embedder": {
            "url_env": "TRITON_URL",
            "model_name_env": "TRITON_MODEL_NAME",
            "tokenizer_env": "TRITON_TOKENIZER",
            "max_batch_size": 8,
        },
        "reasoning": {"max_turns": 10},
        "conversation": {"history_window": 10},
        "graphiti": {
            "search": {"limit": 5, "min_score": 0.8},
            "entity_search": {"max_results": 10},
            "entity_detail": {},
            "node_explore": {"max_results": 100},
            "relation_browse": {"filter_prefix": None, "top_n": 100},
            "relation_filter": {"max_results": 50},
            "similar_entities": {"max_results": 10, "min_score": 0.5},
            "path_find": {"max_hops": 3, "max_paths": 5},
            "episodic_search": {"max_results": 10},
            "graph_stats": {"detail_level": "basic"},
        },
        "session": {
            "redis_url_env": "REDIS_URL",
            "ttl_seconds_env": "REDIS_SESSION_TTL_SECONDS",
        },
        "summarizer": {"max_tokens_env": "SUMMARIZER_MAX_TOKENS"},
        "token_management": {
            "tokenizer_model_env": "TOKENIZER_MODEL_NAME",
            "system_prompt_reservation": 3500,
            "history_budget": 0.3,
        },
        "llm_call_parameters": {
            "thinking_general": {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
            },
            "instruct_general": {
                "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
            },
            "instruct_reasoning": {
                "temperature": 1.0, "top_p": 1.0, "top_k": 40,
                "min_p": 0.0, "presence_penalty": 2.0, "repetition_penalty": 1.0,
            },
            "max_tokens": 2048,
        },
        "response_templates": {
            "error_fallback": "Error fallback.",
            "tool_failure": "Tool failure.",
            "no_info_found": "No info found.",
            "safety_violation": "Safety violation.",
        },
    }
    p = tmp_path / "config.yml"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def _make_mock_orchestrator(config_path):
    """Create an Orchestrator with all heavy dependencies mocked."""
    from cogops.agents.orchestrator import Orchestrator

    mock_svc = MagicMock()
    mock_svc.client_llm = AsyncMock()
    mock_svc.client_secondary = None
    mock_svc.llm_config = None
    mock_svc.model = ""

    mock_tokenizer = MagicMock()
    mock_tokenizer.count = MagicMock(return_value=10)

    with (
        patch("cogops.agents.orchestrator.AsyncLLMService", return_value=mock_svc),
        patch("cogops.agents.orchestrator.get_graph_prompt", return_value="mock prompt"),
        patch("cogops.agents.orchestrator.Tokenizer", return_value=mock_tokenizer),
        patch("cogops.agents.orchestrator.build_tool_registry", return_value=([], {})),
        patch("cogops.agents.orchestrator.RedisSessionStore") as mock_redis_cls,
    ):
        mock_redis = MagicMock()
        mock_redis.available = True
        mock_redis_cls.return_value = mock_redis
        o = Orchestrator(config_path=config_path)
        return o


class TestOrchestratorInit:
    def test_init_loads_config(self, config_path):
        o = _make_mock_orchestrator(config_path)
        assert o.agent_name == "TestAgent"
        assert isinstance(o.system_prompt, str)
        assert len(o.system_prompt) > 0

    def test_system_prompt_shared_across_instances(self, config_path):
        o1 = _make_mock_orchestrator(config_path)
        o2 = _make_mock_orchestrator(config_path)
        # Should be the SAME object (class-level cache)
        assert o1.system_prompt is o2.system_prompt


class TestOrchestratorMethods:
    def test_clear_session(self, config_path):
        o = _make_mock_orchestrator(config_path)
        o.history.append(("q", "a"))
        o.feedback_history.append({"turn_id": "t1"})
        o.clear_session()
        assert o.history == []
        assert o.feedback_history == []

    def test_add_feedback(self, config_path):
        o = _make_mock_orchestrator(config_path)
        o.add_feedback("u1", "t1", "bad", "too long")
        assert len(o.feedback_history) == 1
        assert o.feedback_history[0]["rating"] == "bad"

    def test_get_negative_feedback(self, config_path):
        o = _make_mock_orchestrator(config_path)
        o.add_feedback("u1", "t1", "bad", "wrong answer")
        o.add_feedback("u1", "t2", "good", "")
        result = o.get_negative_feedback()
        assert "bad" in result
        assert "wrong answer" in result
        assert "good" not in result

    def test_get_negative_feedback_empty(self, config_path):
        o = _make_mock_orchestrator(config_path)
        o.add_feedback("u1", "t1", "good", "")
        assert o.get_negative_feedback() == ""
