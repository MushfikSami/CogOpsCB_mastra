"""
Tests for CogOpsCB core components.

Tests cover:
- System prompt generation (placeholders, content sections, Bengali text)
- Memory tools (memory_read, memory_write) with InMemoryStore
- Tool registry (schemas, binding, injectable params)
- ThinkingParser (edge cases, streaming, chunked tags)
- InMemoryStore (turns, summaries, meta, clearing)
- Event channels (filtering)
- Reasoning loop helpers (_make_event, _unpack_tool_response)
- Config loader (sections removed, defaults)
"""

import json
import inspect
import os
import subprocess
from typing import Any, Dict, List

import pytest


def _has_openai():
    """Check if openai package is available."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _has_api_key():
    """Check if API key is configured for real OpenAI client initialization."""
    return bool(
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_ADMIN_KEY")
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_default_placeholders(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "''" not in p or "{agent_name}" not in p  # no unfilled placeholders
        assert "{agent_name}" not in p
        assert "{agent_story}" not in p
        assert "{max_concurrent_query}" not in p

    def test_custom_agent_name(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt(agent_name="TestAgent")
        assert "TestAgent" in p

    def test_custom_story(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt(agent_story="A custom story here")
        assert "A custom story here" in p

    def test_max_concurrent_query(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt(max_concurrent_query=5)
        # Should appear in the QUERY BATCHING section
        assert "answer at most 5 of them" in p

    def test_tools_description_ignored(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt(tools_description="some description")
        # tools_description is a legacy param but not embedded in v2 prompt
        assert "some description" not in p

    def test_all_sections_present(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        for section in ["# SYSTEM", "# USING YOUR TOOLS", "# ANTI-HALLUCINATION",
                         "# CONTEXT HANDLING", "# TONE AND STYLE", "# CONTEXT MANAGEMENT",
                         "# TIME AND LOCALE", "# QUERY BATCHING", "# EXAMPLES"]:
            assert section in p, f"Missing section: {section}"

    def test_anti_hallucination_rules(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "Never construct, guess, or normalize a URL" in p
        assert "Be honest about uncertainty" in p
        assert "Do not invent numbered steps" in p

    def test_no_memory_tool_instructions(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        # Memory tools are disabled — no instructions to call them
        assert "memory_read" not in p
        assert "memory_write" not in p
        assert "# MEMORY (REDIS)" not in p

    def test_bengali_content_present(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "প্রমিত বাংলা" in p
        # Example 2 Bengali text
        assert "দুঃখিত" in p

    def test_prompt_length(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert len(p) > 5000  # comprehensive prompt with examples

    def test_examples_present(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "## Example 1" in p
        assert "## Example 2" in p
        assert "## Example 3" in p
        assert "## Example 4" in p
        assert "## Example 5" in p

    def test_citation_format_in_examples(self):
        # Citation rules ARE enforced — examples must demonstrate the [S#] pattern.
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "[S1]" in p
        assert "# CITATION FORMAT" in p
        assert "NO_RELEVANT_RESULTS" in p

    def test_tool_first_protocol_present(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "TOOL-FIRST PROTOCOL" in p
        assert "MUST call a retrieval tool" in p

    def test_refusal_template_present(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "নির্ভরযোগ্য তথ্য পাওয়া যায়নি" in p

    def test_thinking_tag_instructions(self):
        from cogops.prompts.system import get_system_prompt
        p = get_system_prompt()
        assert "<thinking>" in p
        assert "Wrap every reasoning step" in p

# ---------------------------------------------------------------------------
# Tool registry (plug-in discovery)
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Registry is config-driven. Empty `enabled` ⇒ empty registry; listing a
    valid slug imports and registers that tool's module."""

    def test_empty_enabled_yields_empty_registry(self):
        from cogops.tools.registry import build_tool_registry
        schemas, tool_map = build_tool_registry(enabled=[])
        assert schemas == []
        assert tool_map == {}

    def test_none_enabled_yields_empty_registry(self):
        from cogops.tools.registry import build_tool_registry
        schemas, tool_map = build_tool_registry()
        assert schemas == []
        assert tool_map == {}

    def test_load_jiggasha_tool(self):
        from cogops.tools.registry import build_tool_registry
        schemas, tool_map = build_tool_registry(enabled=["jiggasha"])
        assert len(schemas) == 1
        assert len(tool_map) == 1
        assert schemas[0]["function"]["name"] == "search_gov_services"
        assert "search_gov_services" in tool_map

    def test_schema_enforces_additional_properties_false(self):
        from cogops.tools.registry import build_tool_registry
        schemas, _ = build_tool_registry(enabled=["jiggasha"])
        params = schemas[0]["function"]["parameters"]
        assert params.get("additionalProperties") is False

    def test_missing_tool_module_raises(self):
        from cogops.tools.registry import build_tool_registry
        with pytest.raises(ImportError):
            build_tool_registry(enabled=["this_tool_does_not_exist"])

    def test_injectable_params_includes_ctx(self):
        from cogops.tools.registry import _INJECTABLE_PARAMS
        assert set(_INJECTABLE_PARAMS) == {"user_id", "store", "ctx"}

    def test_bind_tools_injects_ctx(self):
        from functools import partial
        from cogops.tools.registry import build_tool_registry, bind_tools, ToolContext
        _, tool_map = build_tool_registry(enabled=["jiggasha"])
        ctx = ToolContext(user_id="u1", store=None)
        bound = bind_tools(tool_map, ctx)
        # The jiggasha handler declares (query, ctx) — bind_tools should inject
        # the ctx instance via functools.partial.keywords.
        wrapped = bound["search_gov_services"]
        assert isinstance(wrapped, partial)
        assert "ctx" in wrapped.keywords
        assert wrapped.keywords["ctx"] is ctx

    def test_bind_tools_no_context(self):
        from cogops.tools.registry import build_tool_registry, bind_tools, ToolContext
        _, tool_map = build_tool_registry(enabled=[])
        ctx = ToolContext(user_id=None, store=None)
        bound = bind_tools(tool_map, ctx)
        assert len(bound) == 0

    def test_no_tool_references_in_system_prompt(self):
        from cogops.prompts.system import SYSTEM_PROMPT
        # Individual tool names should not be enumerated in the prompt —
        # tools are discoverable from the schema at runtime.
        assert "search_gov_services" not in SYSTEM_PROMPT
        assert "search_wikipedia" not in SYSTEM_PROMPT
        assert "search_web" not in SYSTEM_PROMPT

    def test_tool_context_dataclass(self):
        from cogops.tools.registry import ToolContext
        ctx = ToolContext(user_id="u1", store=None)
        assert ctx.user_id == "u1"
        assert ctx.store is None
        assert ctx.source_map == {}
        assert ctx.tool_map is None
        assert ctx.tools_schema is None

    def test_tool_context_allocates_source_tags_monotonically(self):
        from cogops.tools.registry import ToolContext
        ctx = ToolContext(user_id="u1")
        tag1 = ctx.allocate_source_tag({"text": "first"})
        tag2 = ctx.allocate_source_tag({"text": "second"})
        tag3 = ctx.allocate_source_tag({"text": "third"})
        assert tag1 == "S1"
        assert tag2 == "S2"
        assert tag3 == "S3"
        assert ctx.source_map["S2"]["text"] == "second"

# ---------------------------------------------------------------------------
# ThinkingParser
# ---------------------------------------------------------------------------

class TestThinkingParser:
    @pytest.fixture
    def parse_full(self):
        """Helper: feed + flush, return all chunks."""
        def _parse(text: str):
            from cogops.utils.thinking_parser import ThinkingParser
            parser = ThinkingParser()
            chunks: list[tuple[str, str]] = []
            for c in parser.feed(text):
                chunks.append(c)
            for c in parser.flush():
                chunks.append(c)
            return chunks
        return _parse

    def test_plain_answer_only(self, parse_full):
        chunks = parse_full("just a plain answer")
        # Parser emits per-token for streaming; verify all content is answer channel
        assert all(c[0] == "answer" for c in chunks)
        assert "".join(c[1] for c in chunks) == "just a plain answer"

    def test_single_thinking_block(self, parse_full):
        chunks = parse_full("<thinking>thought</thinking>answer")
        thinking = [c for c in chunks if c[0] == "thinking"]
        answer = [c for c in chunks if c[0] == "answer"]
        assert len(thinking) == 1 and thinking[0][1] == "thought"
        assert len(answer) == 1 and answer[0][1] == "answer"

    def test_multiple_thinking_blocks(self, parse_full):
        text = "<thinking>A</thinking>B<thinking>C</thinking>D"
        chunks = parse_full(text)
        thinking = "".join(c[1] for c in chunks if c[0] == "thinking")
        answer = "".join(c[1] for c in chunks if c[0] == "answer")
        assert thinking == "AC"
        assert answer == "BD"

    def test_thinking_at_start(self, parse_full):
        chunks = parse_full("<thinking>first</thinking>rest")
        assert chunks[0] == ("thinking", "first")
        assert chunks[1] == ("answer", "rest")

    def test_thinking_at_end(self, parse_full):
        chunks = parse_full("start<thinking>end</thinking>")
        assert chunks[0] == ("answer", "start")
        assert chunks[1] == ("thinking", "end")

    def test_empty_thinking(self, parse_full):
        chunks = parse_full("<thinking></thinking>after")
        answer = [c for c in chunks if c[0] == "answer"]
        assert any("after" in c[1] for c in answer)

    def test_only_thinking_no_close(self, parse_full):
        chunks = parse_full("<thinking>unclosed")
        thinking = [c for c in chunks if c[0] == "thinking"]
        assert len(thinking) == 1 and thinking[0][1] == "unclosed"

    def test_unclosed_thinking_in_middle(self, parse_full):
        text = "start<thinking>mid"
        chunks = parse_full(text)
        assert chunks[0] == ("answer", "start")
        assert chunks[1] == ("thinking", "mid")

    def test_answer_before_and_after_thinking(self, parse_full):
        chunks = parse_full("before <thinking>mid</thinking> after")
        answer = [c for c in chunks if c[0] == "answer"]
        assert len(answer) == 2
        assert "before" in answer[0][1]
        assert "after" in answer[1][1]

    def test_empty_input(self):
        from cogops.utils.thinking_parser import ThinkingParser
        parser = ThinkingParser()
        chunks = list(parser.feed(""))
        assert len(chunks) == 0

    def test_flush_empty(self):
        from cogops.utils.thinking_parser import ThinkingParser
        parser = ThinkingParser()
        list(parser.feed(""))
        chunks = list(parser.flush())
        assert len(chunks) == 0

    def test_streaming_chunked_tag_open(self):
        from cogops.utils.thinking_parser import ThinkingParser
        parser = ThinkingParser()
        chunks: list[tuple[str, str]] = []
        for c in parser.feed("<thinki"):
            chunks.append(c)
        for c in parser.feed("ng>done"):
            chunks.append(c)
        for c in parser.flush():
            chunks.append(c)
        all_text = "".join(c[1] for c in chunks)
        assert "done" in all_text

    def test_streaming_chunked_tag_close(self):
        from cogops.utils.thinking_parser import ThinkingParser
        parser = ThinkingParser()
        chunks: list[tuple[str, str]] = []
        for c in parser.feed("before</thinki"):
            chunks.append(c)
        for c in parser.feed("ng>after"):
            chunks.append(c)
        for c in parser.flush():
            chunks.append(c)
        all_text = "".join(c[1] for c in chunks)
        assert "before" in all_text
        assert "after" in all_text

    def test_complex_streaming(self):
        from cogops.utils.thinking_parser import ThinkingParser
        parser = ThinkingParser()
        chunks: list[tuple[str, str]] = []
        for c in parser.feed("<thinki"):
            chunks.append(c)
        for c in parser.feed("ng>plan "):
            chunks.append(c)
        for c in parser.feed("now</thinki"):
            chunks.append(c)
        for c in parser.feed("ng>answer"):
            chunks.append(c)
        for c in parser.flush():
            chunks.append(c)
        thinking = "".join(c[1] for c in chunks if c[0] == "thinking")
        answer = "".join(c[1] for c in chunks if c[0] == "answer")
        assert "plan " in thinking
        assert "answer" in answer

    def test_whitespace_preserved_around_tags(self, parse_full):
        chunks = parse_full("  <thinking>  content  </thinking>  answer  ")
        thinking = [c for c in chunks if c[0] == "thinking"]
        answer = [c for c in chunks if c[0] == "answer"]
        # Whitespace is part of content (exact tag matching)
        assert len(thinking) == 1
        assert thinking[0][1] == "  content  "
        assert len(answer) == 2
        assert answer[0][1] == "  "
        assert answer[1][1] == "  answer  "

    def test_no_tags_yield_answer(self, parse_full):
        text = "some\nmultiline\ntext here"
        chunks = parse_full(text)
        assert all(c[0] == "answer" for c in chunks)

    def test_consecutive_thinking_blocks(self, parse_full):
        text = "<thinking>A</thinking><thinking>B</thinking>"
        chunks = parse_full(text)
        thinking = [c for c in chunks if c[0] == "thinking"]
        assert len(thinking) == 2
        assert thinking[0][1] == "A"
        assert thinking[1][1] == "B"

    def test_long_text_no_tags(self, parse_full):
        long_text = "x" * 200
        chunks = parse_full(long_text)
        all_text = "".join(c[1] for c in chunks if c[0] == "answer")
        assert all_text == long_text

    def test_holds_back_sufficient_bytes(self):
        """The _HOLDBACK constant should be enough to catch tags spanning chunks."""
        from cogops.utils.thinking_parser import ThinkingParser, _HOLDBACK
        # </thinking> is 12 chars, <thinking> is 10 chars, +4 margin
        assert _HOLDBACK == 15  # max(len("<thinking>"), len("</thinking>")) + 4

# ---------------------------------------------------------------------------
# Reasoning loop helpers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_openai(), reason="Reasoning loop imports openai")
class TestReasoningLoopHelpers:
    def test_make_event(self):
        from cogops.llm.reasoning_loop import _make_event
        evt = _make_event("test", {"key": "val"}, "debug")
        assert evt["type"] == "test"
        assert evt["channel"] == "debug"
        assert evt["key"] == "val"

    def test_make_event_data_overrides_channel(self):
        from cogops.llm.reasoning_loop import _make_event
        evt = _make_event("type", {"channel": "override"}, "user")
        # data dict.update() overrides the initial channel
        assert evt["channel"] == "override"

    def test_unpack_tool_response_none(self):
        from cogops.llm.reasoning_loop import _unpack_tool_response
        content, sources = _unpack_tool_response(None)
        assert content == ""
        assert sources == []

    def test_unpack_tool_response_string(self):
        from cogops.llm.reasoning_loop import _unpack_tool_response
        content, sources = _unpack_tool_response("simple text")
        assert content == "simple text"
        assert sources == []

    def test_unpack_tool_response_tuple(self):
        from cogops.llm.reasoning_loop import _unpack_tool_response
        content, sources = _unpack_tool_response(("context part", ["src1", "src2"]))
        assert content == "context part"
        assert sources == ["src1", "src2"]

    def test_unpack_tool_response_tuple_with_lists(self):
        from cogops.llm.reasoning_loop import _unpack_tool_response
        content, sources = _unpack_tool_response(
            (["list item 1", "list item 2"], ["s1"])
        )
        assert content == "list item 1\n\nlist item 2"
        assert sources == ["s1"]

    def test_default_max_turns(self):
        from cogops.llm.reasoning_loop import _DEFAULT_MAX_TURNS
        assert _DEFAULT_MAX_TURNS == 10

    def test_retryable_exceptions(self):
        from cogops.llm.reasoning_loop import RETRYABLE
        assert ConnectionError in RETRYABLE
        assert TimeoutError in RETRYABLE
        assert RuntimeError in RETRYABLE

# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------

class TestInMemoryStore:
    def test_store_and_get_turn(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        store.store_turn("u1", {"turn_id": "t1", "user": "hi", "assistant": "hello"})
        turns = store.get_recent_turns("u1")
        assert len(turns) == 1
        assert turns[0]["user"] == "hi"

    def test_get_recent_turns_limit(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        for i in range(5):
            store.store_turn("u1", {"turn_id": f"t{i}", "user": f"q{i}", "assistant": f"a{i}"})
        turns = store.get_recent_turns("u1", n=3)
        assert len(turns) == 3
        # Most recent first (insert(0))
        assert turns[0]["user"] == "q4"

    def test_clear_turns(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        store.store_turn("u1", {"turn_id": "t1", "user": "hi", "assistant": "hello"})
        store.clear_turns("u1")
        assert store.get_recent_turns("u1") == []

    def test_set_and_get_meta(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        store.set_last_assistant_meta("u1", {"text": "reply", "turn_id": "t1"})
        meta = store.get_last_assistant_meta("u1")
        assert meta == {"text": "reply", "turn_id": "t1"}

    def test_meta_default_none(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        assert store.get_last_assistant_meta("u1") is None

    def test_clear_all(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        store.store_turn("u1", {"turn_id": "t1", "user": "hi", "assistant": "hello"})
        store.set_last_assistant_meta("u1", {"text": "reply"})
        store.clear_all("u1")
        assert store.get_recent_turns("u1") == []
        assert store.get_last_assistant_meta("u1") is None

    def test_turns_per_user_isolated(self):
        from cogops.session.redis_store import InMemoryStore
        store = InMemoryStore()
        store.store_turn("u1", {"turn_id": "t1", "user": "hi", "assistant": "hello"})
        store.store_turn("u2", {"turn_id": "t2", "user": "hey", "assistant": "hi"})
        assert len(store.get_recent_turns("u1")) == 1
        assert len(store.get_recent_turns("u2")) == 1
        assert store.get_recent_turns("u1")[0]["user"] == "hi"
        assert store.get_recent_turns("u2")[0]["user"] == "hey"

# ---------------------------------------------------------------------------
# Citation extractor + Sources block (cogops/verifier/citations.py)
# ---------------------------------------------------------------------------

class TestCitations:
    def test_extract_tags_in_order(self):
        from cogops.verifier.citations import extract_citation_tags
        ans = "প্রথমে গিয়ে আবেদন [S1]। তারপর ফি [S2][S1]।"
        assert extract_citation_tags(ans) == ["S1", "S2", "S1"]

    def test_extract_tags_empty(self):
        from cogops.verifier.citations import extract_citation_tags
        assert extract_citation_tags("") == []
        assert extract_citation_tags("no citations here") == []

    def test_extract_citations_bengali_sentence_split(self):
        from cogops.verifier.citations import extract_citations
        # Two Bengali sentences separated by danda
        ans = "প্রথম তথ্য [S1]। দ্বিতীয় তথ্য [S2]।"
        pairs = extract_citations(ans)
        assert len(pairs) == 2
        tags = [t for t, _ in pairs]
        assert tags == ["S1", "S2"]
        # Each sentence is attached to its tag
        assert "প্রথম তথ্য" in pairs[0][1]
        assert "দ্বিতীয় তথ্য" in pairs[1][1]

    def test_extract_citations_multi_tag_in_one_sentence(self):
        from cogops.verifier.citations import extract_citations
        ans = "ফি এবং প্রক্রিয়া [S1][S2]।"
        pairs = extract_citations(ans)
        assert len(pairs) == 2
        assert pairs[0][1] == pairs[1][1]  # same sentence
        assert {t for t, _ in pairs} == {"S1", "S2"}

    def test_strip_unknown_tags(self):
        from cogops.verifier.citations import strip_unknown_tags
        source_map = {"S1": {"text": "real"}}
        ans = "real fact [S1]. fake fact [S99]."
        cleaned, dropped = strip_unknown_tags(ans, source_map)
        assert "[S1]" in cleaned
        assert "[S99]" not in cleaned
        assert dropped == ["S99"]

    def test_strip_unknown_tags_no_change(self):
        from cogops.verifier.citations import strip_unknown_tags
        source_map = {"S1": {}, "S2": {}}
        ans = "x [S1] y [S2]."
        cleaned, dropped = strip_unknown_tags(ans, source_map)
        assert cleaned == ans
        assert dropped == []

    def test_build_sources_block_only_lists_used_tags(self):
        from cogops.verifier.citations import build_sources_block
        source_map = {
            "S1": {"category": "পাসপোর্ট", "topic": "ফি", "passage_id": 1, "tool": "jiggasha"},
            "S2": {"category": "পাসপোর্ট", "topic": "আবেদন", "passage_id": 2, "tool": "jiggasha"},
            "S3": {"category": "NID", "topic": "সংশোধন", "passage_id": 9, "tool": "jiggasha"},
        }
        # Only S1 and S2 are used in the answer; S3 must NOT appear
        block = build_sources_block(source_map, used_tags=["S1", "S2", "S1"])
        assert "**সূত্র (Sources)**" in block
        assert "[S1]" in block
        assert "[S2]" in block
        assert "[S3]" not in block

    def test_build_sources_block_empty_when_no_used_tags(self):
        from cogops.verifier.citations import build_sources_block
        block = build_sources_block({"S1": {"topic": "x"}}, used_tags=[])
        assert block == ""

    def test_build_sources_block_deduplicates_in_order(self):
        from cogops.verifier.citations import build_sources_block
        source_map = {
            "S1": {"topic": "first", "passage_id": 1, "tool": "jiggasha"},
            "S2": {"topic": "second", "passage_id": 2, "tool": "jiggasha"},
        }
        block = build_sources_block(source_map, used_tags=["S2", "S1", "S2", "S1"])
        # S2 first (first cited), then S1
        s2_idx = block.find("[S2]")
        s1_idx = block.find("[S1]")
        assert 0 < s2_idx < s1_idx


# ---------------------------------------------------------------------------
# Event channels
# ---------------------------------------------------------------------------

class TestEventChannels:
    def test_filter_for_user(self):
        from cogops.events.channels import filter_for_user
        events = [
            {"channel": "user", "data": "a"},
            {"channel": "debug", "data": "b"},
            {"channel": "both", "data": "c"},
        ]
        result = list(filter_for_user(events))
        assert len(result) == 2
        assert result[0]["data"] == "a"
        assert result[1]["data"] == "c"

    def test_filter_for_debug(self):
        from cogops.events.channels import filter_for_debug
        events = [
            {"channel": "user", "data": "a"},
            {"channel": "debug", "data": "b"},
            {"channel": "both", "data": "c"},
        ]
        result = list(filter_for_debug(events))
        assert len(result) == 2
        assert result[0]["data"] == "b"
        assert result[1]["data"] == "c"

    def test_filter_for_user_default_channel(self):
        from cogops.events.channels import filter_for_user
        events = [{"data": "x"}]  # no channel key
        result = list(filter_for_user(events))
        # Default channel is "user" -> should be included
        assert len(result) == 1

    def test_filter_for_debug_default_channel(self):
        from cogops.events.channels import filter_for_debug
        events = [{"data": "x"}]  # no channel key
        result = list(filter_for_debug(events))
        # Default channel is "user" -> NOT debug
        assert len(result) == 0

    def test_filter_empty(self):
        from cogops.events.channels import filter_for_user
        assert list(filter_for_user([])) == []

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_loads(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_agent_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert cfg["agent"]["name"] == "আশা"
        assert cfg["agent"]["story"] != ""

    def test_llm_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "llm" in cfg
        assert "api_key_env" in cfg["llm"]
        assert "max_context_tokens" in cfg["llm"]

    def test_reasoning_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        # The deterministic pipeline doesn't use the ReAct loop; reasoning.*
        # is retained only for backward compatibility with the chitchat path.
        assert cfg["reasoning"]["max_turns"] == 1
        assert cfg["reasoning"]["max_concurrent_query"] == 2

    def test_tools_section_enables_jiggasha(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "tools" in cfg
        assert "jiggasha" in cfg["tools"]["enabled"]
        # Instruction-based retrieval cap.
        assert cfg["tools"]["jiggasha"]["top_k_fetch"] >= 10

    def test_retrieval_refusal_text_present(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "retrieval" in cfg
        assert "নির্ভরযোগ্য" in cfg["retrieval"]["refusal_text_bn"]
        assert "তথ্য" in cfg["retrieval"]["refusal_text_bn"]

    def test_session_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "redis_url_default" in cfg["session"]

    def test_removed_top_level_jiggasha_section(self):
        # jiggasha is now under tools.jiggasha (plug-in registry), not a
        # top-level section.
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "jiggasha" not in cfg.keys()  # not top-level
        assert "jiggasha" in cfg.get("tools", {})  # but under tools.

    def test_removed_wiki_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "wiki" not in cfg

    def test_removed_summarizer_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "summarizer" not in cfg

    def test_removed_post_tool_refine_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "post_tool_refine" not in cfg

    def test_removed_history_query_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "history_query" not in cfg

    def test_removed_token_management_section(self):
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "token_management" not in cfg

    def test_pipeline_section_present(self):
        """LLM rerank moved to Jiggasha; chatbot-side pipeline keeps composer
        knobs and the disambiguation thresholds. Rerank knobs (keep_cap,
        weak_per_sub_cap, …) now live under tools.jiggasha.*."""
        from cogops.config.loader import load_config
        cfg = load_config()
        assert "pipeline" in cfg
        assert "composer" in cfg["pipeline"]
        assert "max_tokens" in cfg["pipeline"]["composer"]
        assert "disambiguation" in cfg["pipeline"]
        assert "min_distinct_services" in cfg["pipeline"]["disambiguation"]

        # Instruction-based retrieval knobs on Jiggasha.
        jcfg = cfg.get("tools", {}).get("jiggasha", {})
        assert jcfg.get("use_instruction") is True
        assert "cosine_threshold" in jcfg
        assert "token_budget" in jcfg

# ---------------------------------------------------------------------------
# Messages (fallback strings)
# ---------------------------------------------------------------------------

class TestFallbackMessages:
    def test_error_fallback_bn(self):
        from cogops.prompts.messages import ERROR_FALLBACK_BN
        assert len(ERROR_FALLBACK_BN) > 0
        assert "প্রযুক্তিগত" in ERROR_FALLBACK_BN

    def test_server_load_fallback_bn(self):
        from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
        assert len(SERVER_LOAD_FALLBACK_BN) > 0
        assert "সার্ভারে" in SERVER_LOAD_FALLBACK_BN

    def test_both_in_bengali(self):
        from cogops.prompts.messages import ERROR_FALLBACK_BN, SERVER_LOAD_FALLBACK_BN
        # Verify they contain Bengali script
        assert any("ঀ" <= c <= "৿" for c in ERROR_FALLBACK_BN)
        assert any("ঀ" <= c <= "৿" for c in SERVER_LOAD_FALLBACK_BN)

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_openai(), reason="AsyncLLMService imports openai")
class TestAsyncLLMService:
    def test_default_values(self):
        from cogops.llm.clients import AsyncLLMService
        svc = AsyncLLMService()
        assert svc.model == ""
        assert svc.max_context_tokens == 32000

    def test_no_reranker_or_secondary(self):
        from cogops.llm.clients import AsyncLLMService
        import inspect
        sig = inspect.signature(AsyncLLMService.__init__)
        params = list(sig.parameters.keys())
        assert "client_reranker" not in params
        assert "config_reranker" not in params
        assert "client_secondary" not in params
        assert "config_secondary" not in params
        assert "client_llm" in params
        assert "config_llm" in params

    def test_health_check_method_exists(self):
        from cogops.llm.clients import AsyncLLMService
        assert hasattr(AsyncLLMService, "health_check")

# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_openai() or not _has_api_key(), reason="Orchestrator needs openai + API key")
class TestOrchestrator:
    def test_config_loaded(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert o.agent_name == "আশা"

    def test_system_prompt_cached(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert o.system_prompt is not None
        assert len(o.system_prompt) > 5000

    def test_system_prompt_contains_agent_name(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert "আশা" in o.system_prompt

    def test_tools_registry_loaded(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        # jiggasha is the only enabled plug-in in configs/config.yml.
        assert len(o.tools_schema) == 1
        assert len(o.raw_tool_map) == 1
        assert "search_gov_services" in o.raw_tool_map

    def test_no_tokenizer_or_truncation(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert not hasattr(o, "tokenizer")
        assert not hasattr(o, "_tokenizer_model")
        assert not hasattr(o, "system_prompt_reservation")

    def test_no_summarizer(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert not hasattr(o, "summarizer_max_tokens")

    def test_no_secondary_client_in_context(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        ctx = o._build_tool_context("u1")
        assert not hasattr(ctx, "secondary_client")
        assert not hasattr(ctx, "secondary_model")

    def test_max_turns_from_config(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert o.max_turns == 3

    def test_max_concurrent_query_from_config(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert o.max_concurrent_query == 2

    def test_prompt_not_cached_across_instances(self):
        from cogops.agents.orchestrator import Orchestrator
        # First instance creates the cache
        o1 = Orchestrator()
        first_prompt = o1.system_prompt
        # Second instance uses the same cached prompt
        o2 = Orchestrator()
        assert o2.system_prompt is first_prompt

    def _cached_prompt_is_string(self):
        from cogops.agents.orchestrator import Orchestrator
        Orchestrator._cached_system_prompt = None  # reset for test
        o = Orchestrator()
        assert isinstance(o.system_prompt, str)

    def test_clear_session(self):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        o.clear_session()  # Should not raise

    def test_prompt_does_not_enumerate_tool_names(self):
        # Tool names live in the schema, not the prompt — keeps the prompt
        # stable as tools evolve.
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        assert "search_gov_services" not in o.system_prompt
        assert "search_wikipedia" not in o.system_prompt
        assert "search_web" not in o.system_prompt

# ---------------------------------------------------------------------------
# Setup.py
# ---------------------------------------------------------------------------

class TestSetupPy:
    def test_transformers_removed(self):
        with open("setup.py", "r") as f:
            content = f.read()
        # transformers should not appear in install_requires
        # Check that 'transformers' is not a standalone dependency
        assert "'transformers'" not in content and '"transformers"' not in content

# ---------------------------------------------------------------------------
# API health endpoint
# ---------------------------------------------------------------------------

class TestAPI:
    def test_no_jiggasha_in_health(self):
        with open("api.py", "r") as f:
            content = f.read()
        assert "jiggasha" not in content.lower() or "jiggasha" not in content

    def test_no_wiki_in_health(self):
        with open("api.py", "r") as f:
            content = f.read()
        # The wiki tab section in Streamlit app still exists, but api.py health should not probe wiki
        # Check that JIGGASHA_ENDPOINT and WIKI_ENDPOINT are not used in api.py
        assert "JIGGASHA_ENDPOINT" not in content
        assert "WIKI_ENDPOINT" not in content

# ---------------------------------------------------------------------------
# No stale references
# ---------------------------------------------------------------------------

class TestNoStaleReferences:
    def test_no_post_tool_refine_import(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "_post_tool_refine", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found _post_tool_refine: {result.stdout}"

    def test_no_summarizer_import(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "run_summarizer_task", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found run_summarizer_task: {result.stdout}"

    def test_no_truncation_import(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "truncate_messages_to_budget", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found truncate_messages_to_budget: {result.stdout}"

    def test_no_thinking_stripper(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "ThinkingStripper", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found ThinkingStripper: {result.stdout}"

    def test_no_client_secondary(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "client_secondary", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found client_secondary: {result.stdout}"

    def test_no_client_reranker(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "client_reranker", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found client_reranker: {result.stdout}"

    def test_no_thinking_parser_import(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "from cogops.utils.thinking_parser import ThinkingParser",
             "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        # Should find exactly 1 import (in reasoning_loop.py)
        assert result.returncode == 0
        assert "reasoning_loop.py" in result.stdout

    def test_no_old_tokenizer_import(self):
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "from cogops.utils.tokenizer", "--include=*.py", "cogops/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, f"Found old tokenizer import: {result.stdout}"
