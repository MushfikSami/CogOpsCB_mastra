"""test_config_loader.py — Phase 1: config loader tests."""
import os
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _reset_modules():
    """Clear cached modules so each test gets fresh imports."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.config"):
            del sys.modules[mod]
    yield


class TestLoadConfig:
    """load_config() basic functionality."""

    def test_loads_valid_yaml(self, config_path):
        from cogops.config.loader import load_config
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert cfg["agent_name"] == "TestAgent"

    def test_loads_graphiti_key(self, config_path):
        from cogops.config.loader import load_config
        cfg = load_config(config_path)
        assert "graphiti" in cfg
        assert cfg["graphiti"]["search"]["limit"] == 5

    def test_loads_all_sections(self, config_path):
        from cogops.config.loader import load_config
        cfg = load_config(config_path)
        for key in ("llm", "reranker", "secondary", "reasoning",
                     "graphiti", "llm_call_parameters", "response_templates"):
            assert key in cfg, f"Missing section: {key}"


class TestEndpointConfig:
    """EndpointConfig env-var loading."""

    def test_reads_env_vars(self, config_path, monkeypatch):
        """Test EndpointConfig reads env vars at construction time."""
        # Must delete module to get fresh dotenv import
        for mod in list(sys.modules.keys()):
            if mod.startswith("cogops.config"):
                del sys.modules[mod]
        monkeypatch.setenv("TEST_LLM_KEY", "key123")
        monkeypatch.setenv("TEST_LLM_MODEL", "model-abc")
        monkeypatch.setenv("TEST_LLM_URL", "http://test-url")
        from cogops.config.loader import EndpointConfig
        ep = EndpointConfig(
            api_key_env="TEST_LLM_KEY", model_name_env="TEST_LLM_MODEL",
            base_url_env="TEST_LLM_URL", max_context_tokens=1000,
        )
        assert ep.api_key == "key123"
        assert ep.model == "model-abc"
        assert ep.base_url == "http://test-url"
        assert ep.max_context_tokens == 1000

    def test_defaults_to_empty_on_missing_env(self, config_path, monkeypatch):
        # Ensure vars DON'T exist
        monkeypatch.delenv("NONEXISTENT_KEY_A", raising=False)
        monkeypatch.delenv("NONEXISTENT_M", raising=False)
        monkeypatch.delenv("NONEXISTENT_U", raising=False)
        from cogops.config.loader import _load_endpoint_config
        cfg = {"api_key_env": "NONEXISTENT_KEY_A", "model_name_env": "NONEXISTENT_M",
               "base_url_env": "NONEXISTENT_U"}
        ep = _load_endpoint_config(cfg, "_unused")
        assert ep.api_key == ""
        assert ep.model == ""
        assert ep.base_url == ""


class TestGetToolConfig:
    """Unified graphiti key helper."""

    def test_graphiti_key_first(self, config_path):
        from cogops.config.loader import load_config, get_tool_config
        cfg = load_config(config_path)
        assert get_tool_config(cfg, "entity_search") == {"max_results": 10}
        assert get_tool_config(cfg, "node_explore") == {"max_results": 100}

    def test_graph_search_reads_from_graphiti_search(self, config_path):
        from cogops.config.loader import load_config, get_tool_config
        cfg = load_config(config_path)
        assert get_tool_config(cfg, "graph_search") == {"limit": 5, "min_score": 0.8}

    def test_fallback_to_flat_key(self, config_path, tmp_path):
        """Old flat key still works via secondary section."""
        flat_cfg = dict(yaml.safe_load(Path(config_path).read_text()))
        flat_cfg["secondary"]["old_tool"] = {"limit": 42}
        p = tmp_path / "flat.yml"
        p.write_text(yaml.dump(flat_cfg))
        from cogops.config.loader import load_config, get_tool_config
        cfg = load_config(str(p))
        assert get_tool_config(cfg, "old_tool") == {"limit": 42}

    def test_returns_empty_dict_for_missing(self, config_path):
        from cogops.config.loader import load_config, get_tool_config
        cfg = load_config(config_path)
        assert get_tool_config(cfg, "nonexistent_tool") == {}

    def test_secondary_tools(self, config_path):
        from cogops.config.loader import load_config, get_tool_config
        cfg = load_config(config_path)
        assert get_tool_config(cfg, "grep_passage") == {"context_lines": 2}
        assert get_tool_config(cfg, "spawn_subagent") == {"max_turns": 5}
