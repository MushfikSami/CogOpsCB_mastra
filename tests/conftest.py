import sys
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

# Ensure the project root is on the path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        "secondary": {
            "grep_passage": {"context_lines": 2},
            "extract_from_document": {"max_tokens": 2048},
            "delegate_task": {"max_tokens": 2048},
            "spawn_subagent": {"max_turns": 5},
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


@pytest.fixture
def mock_openai_client():
    """Create a mock AsyncOpenAI client with minimal structure."""
    mock = AsyncMock()
    mock.api_key = "test-key"
    mock.base_url = "http://localhost:9001/v1"
    return mock


@pytest.fixture
def mock_llm_service(mock_openai_client):
    """Create a mocked AsyncLLMService."""
    from cogops.llm.clients import AsyncLLMService

    service = AsyncLLMService(client_llm=mock_openai_client, config_llm=None)
    return service


@pytest.fixture
def sample_turns():
    """Sample conversation turns for history tests."""
    return [
        {"turn_id": "a1", "user": "What is passport fee?", "assistant": "The passport fee is 3000 Taka."},
        {"turn_id": "a2", "user": "What about renewal?", "assistant": "Renewal fee is 1500 Taka."},
        {"turn_id": "a3", "user": "What documents are needed?", "assistant": "NID and old passport."},
    ]


@pytest.fixture
def mock_redis():
    """Mock the redis module for RedisSessionStore tests."""
    with patch.dict(sys.modules, {"redis": MagicMock()}):
        yield
