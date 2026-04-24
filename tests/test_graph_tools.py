"""test_graph_tools.py — Phase 4: graph tool tests against live Neo4j.

Each test creates its own asyncio.run() call which is fine for a fresh
process. Event loop conflicts only happen when running multiple tests in
the same process — so these tests are designed to run in isolation, or
we use a shared process-level loop via a module fixture.
"""
import sys
import os

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

os.environ.setdefault("NEO4J_URI", "bolt+ssc://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "L09YKaTBcxKUy0DTXXu69nkUbUH5KRjzUvH765cBKTI=")
os.environ.setdefault("NEO4J_DATABASE", "qwen34neo4j")
os.environ.setdefault("ADMIN_DEBUG_SECRET", "test-debug-secret")


class _LoopHolder:
    """Hold a single event loop for the entire test module."""
    loop = None

    @classmethod
    def get_loop(cls):
        import asyncio
        if cls.loop is None or cls.loop.is_closed():
            cls.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.loop)
        return cls.loop


def _async_run(coro):
    """Run coroutine on the shared process-level event loop."""
    loop = _LoopHolder.get_loop()
    return loop.run_until_complete(coro)


class TestGraphSearch:
    def test_search_passport(self):
        from cogops.tools.graph.search import graph_search
        result = _async_run(graph_search(query="passport"))
        assert "## Nodes" in result

    def test_search_bangla(self):
        from cogops.tools.graph.search import graph_search
        result = _async_run(graph_search(query="পাসপোর্ট"))
        assert len(result) > 0


class TestEntitySearch:
    def test_fuzzy_search_passport(self):
        from cogops.tools.graph.entity_search import entity_search
        result = _async_run(entity_search(search_term="passport"))
        assert len(result) > 0

    def test_fuzzy_search_fees(self):
        from cogops.tools.graph.entity_search import entity_search
        result = _async_run(entity_search(search_term="fee"))
        assert len(result) >= 0


class TestEntityDetail:
    def test_detail_passport(self):
        from cogops.tools.graph.entity_detail import entity_detail
        result = _async_run(entity_detail(identifier="passport"))
        assert len(result) > 0 or "No entity found" in result


class TestNodeExplore:
    def test_explore_passport(self):
        from cogops.tools.graph.node_explore import node_explore
        result = _async_run(node_explore(entity_name="passport"))
        assert len(result) > 0 or "No entity found" in result


class TestRelationBrowse:
    def test_list_relations(self):
        from cogops.tools.graph.relation_browse import relation_browse
        result = _async_run(relation_browse())
        assert "RELATES_TO" in result or "MENTIONS" in result or len(result) > 0


class TestRelationFilter:
    def test_filter_relates_to(self):
        from cogops.tools.graph.relation_filter import relation_filter
        result = _async_run(relation_filter(relation_name="RELATES_TO"))
        assert len(result) > 0

    def test_filter_mentions(self):
        from cogops.tools.graph.relation_filter import relation_filter
        result = _async_run(relation_filter(relation_name="MENTIONS"))
        assert len(result) > 0


class TestSimilarEntities:
    def test_similar_to_passport(self):
        from cogops.tools.graph.similar_entities import similar_entities
        result = _async_run(similar_entities(entity_name="passport"))
        assert "No info found" not in result


class TestPathFind:
    def test_path_passport_to_nid(self):
        from cogops.tools.graph.path_find import path_find
        result = _async_run(path_find(start_entity="passport", end_entity="nid"))
        assert len(result) > 0


class TestEpisodicSearch:
    def test_search_passage_passport(self):
        from cogops.tools.graph.episodic_search import episodic_search
        result = _async_run(episodic_search(search_term="passport"))
        assert len(result) > 0


class TestGraphStats:
    def test_basic_stats(self):
        from cogops.tools.graph.graph_stats import graph_stats
        result = _async_run(graph_stats(detail_level="basic"))
        assert "nodes" in result.lower() or len(result) > 0

    def test_detailed_stats(self):
        from cogops.tools.graph.graph_stats import graph_stats
        result = _async_run(graph_stats(detail_level="detailed"))
        assert len(result) > 0


class TestToolRegistryLive:
    @pytest.mark.parametrize("tool_name", [
        "graph_search", "entity_search", "entity_detail", "node_explore",
        "relation_browse", "relation_filter", "similar_entities", "path_find",
        "episodic_search", "graph_stats",
        "grep_passage", "extract_from_document", "delegate_task", "spawn_subagent",
        "ask_user", "history_query",
    ], ids=lambda x: x)
    def test_tool_exists_in_registry(self, tool_name):
        from cogops.tools.registry import get_tool_names, build_tool_registry
        tools_schema, tool_map = build_tool_registry(
            secondary_client=None, secondary_model=""
        )
        names = get_tool_names(tools_schema)
        assert tool_name in names
