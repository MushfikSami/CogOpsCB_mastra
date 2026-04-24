"""test_reasoning_loop.py — Phase 2: reasoning loop (mocked LLM)."""
import sys
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def asyncio_run(coro):
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


class _AsyncIterator:
    """Wrap a sync iterable into an async iterator for testing."""
    def __init__(self, items):
        self._iter = iter(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.llm"):
            del sys.modules[mod]
    yield


def _delta(content=None, reasoning=None, tool_calls=None):
    d = MagicMock()
    if content is not None:
        d.content = content
    else:
        d.content = None
    if reasoning is not None:
        d.reasoning = reasoning
    else:
        d.reasoning = None
    if tool_calls is not None:
        d.tool_calls = tool_calls
    else:
        d.tool_calls = None
    return d


class TestReasoningLoopBasic:
    """stream_with_tool_calls basic flow."""

    def test_no_tool_calls_yields_answer(self):
        from cogops.llm.reasoning_loop import stream_with_tool_calls

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=_delta(content="Hello world"))]
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        mock_stream = _AsyncIterator([mock_chunk, empty_chunk])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        messages = []
        events = []

        def run():
            async def _collect():
                async for evt in stream_with_tool_calls(
                    client_llm=mock_client, model="test",
                    messages=messages, tools_schema=[],
                    available_tools={},
                ):
                    events.append(evt)
            return _collect()

        asyncio_run(run())

        types = [e["type"] for e in events]
        assert "answer_chunk" in types
        answer_events = [e for e in events if e["type"] == "answer_chunk"]
        assert len(answer_events) > 0
        for ae in answer_events:
            assert ae.get("channel") in ("both", "user")

    def test_tool_call_yields_tool_call_event(self):
        from cogops.llm.reasoning_loop import stream_with_tool_calls

        tc_id = "call_123"
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = tc_id
        tc_delta.function = MagicMock()
        tc_delta.function.name = "graph_search"
        tc_delta.function.arguments = '{"query": "test"}'

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=_delta(tool_calls=[tc_delta]))]
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        mock_stream = _AsyncIterator([mock_chunk, empty_chunk])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        async def dummy_tool(**kw):
            return "result"

        messages = []
        events = []

        def run():
            async def _collect():
                async for evt in stream_with_tool_calls(
                    client_llm=mock_client, model="test",
                    messages=messages,
                    tools_schema=[{"type": "function", "function": {"name": "graph_search", "description": "d", "parameters": {
                        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
                    }}}],
                    available_tools={"graph_search": dummy_tool},
                ):
                    events.append(evt)
            return _collect()

        asyncio_run(run())

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types

    def test_event_has_type_and_channel(self):
        from cogops.llm.reasoning_loop import stream_with_tool_calls

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=_delta(content="Hi"))]
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        mock_stream = _AsyncIterator([mock_chunk, empty_chunk])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        events = []

        def run():
            async def _collect():
                async for evt in stream_with_tool_calls(
                    client_llm=mock_client, model="test",
                    messages=[], tools_schema=[],
                    available_tools={},
                ):
                    events.append(evt)
            return _collect()

        asyncio_run(run())

        for evt in events:
            assert "type" in evt
            assert "channel" in evt

    def test_max_turns_enforced(self):
        """After 10 turns, break even with tool calls."""
        from cogops.llm.reasoning_loop import MAX_TURNS, stream_with_tool_calls

        assert MAX_TURNS == 10

        tc_id = "call_1"
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = tc_id
        tc_delta.function = MagicMock()
        tc_delta.function.name = "tool"
        tc_delta.function.arguments = "{}"

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=_delta(tool_calls=[tc_delta]))]
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        mock_stream = _AsyncIterator([mock_chunk, empty_chunk])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        call_count = [0]
        async def dummy(**kw):
            call_count[0] += 1
            return "ok"

        events = []

        def run():
            async def _collect():
                async for evt in stream_with_tool_calls(
                    client_llm=mock_client, model="test",
                    messages=[],
                    tools_schema=[{"type": "function", "function": {"name": "tool", "description": "d", "parameters": {}}}],
                    available_tools={"tool": dummy},
                    max_turns=10,
                ):
                    events.append(evt)
            return _collect()

        asyncio_run(run())

        turn_starts = [e for e in events if e["type"] == "turn_start"]
        assert len(turn_starts) <= 10
