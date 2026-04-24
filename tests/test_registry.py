"""test_registry.py — Phase 1: tool registry tests."""
import sys
import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.tools"):
            del sys.modules[mod]
    yield


class TestBuildToolRegistry:
    """build_tool_registry() output structure."""

    def test_returns_16_tools(self):
        from cogops.tools.registry import build_tool_registry
        schema, m = build_tool_registry()
        assert len(schema) == 16
        assert len(m) == 16

    def test_all_schemas_are_valid_openai(self):
        from cogops.tools.registry import build_tool_registry
        schema, _ = build_tool_registry()
        for t in schema:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]
            assert "description" in t["function"]
            assert "parameters" in t["function"]
            assert "properties" in t["function"]["parameters"]

    def test_tool_map_keys_match_schema_names(self):
        from cogops.tools.registry import build_tool_registry
        schema, m = build_tool_registry()
        schema_names = {t["function"]["name"] for t in schema}
        assert set(m.keys()) == schema_names

    def test_no_duplicate_names(self):
        from cogops.tools.registry import build_tool_registry
        schema, _ = build_tool_registry()
        names = [t["function"]["name"] for t in schema]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self):
        from cogops.tools.registry import build_tool_registry
        schema, m = build_tool_registry()
        expected = {
            "graph_search", "entity_search", "entity_detail", "node_explore",
            "relation_browse", "relation_filter", "similar_entities", "path_find",
            "episodic_search", "graph_stats",
            "grep_passage", "extract_from_document", "delegate_task",
            "spawn_subagent",
            "ask_user",
            "history_query",
        }
        assert set(m.keys()) == expected


class TestGetToolNames:
    def test_returns_all_names(self):
        from cogops.tools.registry import build_tool_registry, get_tool_names
        schema, _ = build_tool_registry()
        names = get_tool_names(schema)
        assert len(names) == 16

    def test_names_match_schema(self):
        from cogops.tools.registry import build_tool_registry, get_tool_names
        schema, _ = build_tool_registry()
        names = get_tool_names(schema)
        expected = {t["function"]["name"] for t in schema}
        assert set(names) == expected
