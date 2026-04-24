"""test_tools_pure.py — Phase 1: pure logic tool tests."""
import sys
import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.tools.ask_user"):
            del sys.modules[mod]
        if mod.startswith("cogops.tools.history"):
            del sys.modules[mod]
        if mod.startswith("cogops.tools.secondary.grep"):
            del sys.modules[mod]
    yield


class TestGrepPassage:
    """grep_passage: pure regex, no LLM."""

    def test_valid_regex_finds_match(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        passage = "line 1\nline 2\nline 3\nline 4\nline 5"
        result = asyncio_run(grep_passage(passage, "line 3"))
        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result

    def test_no_matches(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        passage = "hello world"
        result = asyncio_run(grep_passage(passage, "xyz"))
        assert "No matches" in result

    def test_invalid_regex(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        passage = "hello"
        result = asyncio_run(grep_passage(passage, "[invalid"))
        assert "Invalid regex" in result

    def test_empty_passage(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        result = asyncio_run(grep_passage("", "test"))
        assert "No passage" in result

    def test_empty_pattern(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        result = asyncio_run(grep_passage("hello", ""))
        assert "No passage" in result or "No" in result

    def test_case_insensitive(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        passage = "Hello World\nHELLO again"
        result = asyncio_run(grep_passage(passage, "hello"))
        assert "Hello World" in result
        assert "HELLO again" in result

    def test_context_lines(self):
        from cogops.tools.secondary.grep_passage import grep_passage
        passage = "l1\nl2\nl3\nl4\nl5"
        result = asyncio_run(grep_passage(passage, "l3", context_lines=1))
        assert "l2" in result
        assert "l3" in result
        assert "l4" in result
        # Should NOT include l1 or l5
        assert "l1" not in result


class TestAskUser:
    """ask_user: raises ClarificationRequested."""

    def test_raises_clarification_requested(self):
        from cogops.tools.ask_user import ask_user, ClarificationRequested
        with pytest.raises(ClarificationRequested) as exc_info:
            asyncio_run(ask_user("What do you mean?"))
        assert exc_info.value.question == "What do you mean?"
        assert exc_info.value.options == []
        assert exc_info.value.reason is None
        assert exc_info.value.turn_id is not None

    def test_with_options(self):
        from cogops.tools.ask_user import ask_user, ClarificationRequested
        with pytest.raises(ClarificationRequested) as exc_info:
            asyncio_run(ask_user("Choose:", options=["A", "B"], reason="ambiguous"))
        assert exc_info.value.options == ["A", "B"]
        assert exc_info.value.reason == "ambiguous"


class TestHistoryQuery:
    """history_query: lookup/summarize/recent logic (no Redis)."""

    def test_lookup_finds_match(self):
        from cogops.tools.history.query import history_query_lookup
        turns = [
            {"turn_id": "t1", "user": "passport fee", "assistant": "3000 Taka"},
            {"turn_id": "t2", "user": "visa info", "assistant": "Go to embassy"},
        ]
        result = history_query_lookup(turns, "passport")
        assert "t1" in result
        assert "passport fee" in result

    def test_lookup_no_match(self):
        from cogops.tools.history.query import history_query_lookup
        turns = [{"turn_id": "t1", "user": "passport fee", "assistant": "3000"}]
        result = history_query_lookup(turns, "nonexistent")
        assert "No matches" in result

    def test_lookup_empty_turns(self):
        from cogops.tools.history.query import history_query_lookup
        result = history_query_lookup([], "anything")
        assert "No conversation history" in result

    def test_summarize_returns_summary(self):
        from cogops.tools.history.query import history_query_summarize
        result = history_query_summarize("User asked about passport.")
        assert "Recent conversation summary:" in result
        assert "User asked about passport" in result

    def test_summarize_empty(self):
        from cogops.tools.history.query import history_query_summarize
        result = history_query_summarize("")
        assert "No conversation summary available" in result

    def test_recent_returns_last_n(self):
        from cogops.tools.history.query import history_query_recent
        # Redis lpush means most recent is first in the list
        turns = [
            {"turn_id": "t3", "user": "q3", "assistant": "a3"},
            {"turn_id": "t2", "user": "q2", "assistant": "a2"},
            {"turn_id": "t1", "user": "q1", "assistant": "a1"},
        ]
        result = history_query_recent(turns, n=2)
        assert "t3" in result
        assert "t2" in result
        assert "t1" not in result

    def test_recent_empty_turns(self):
        from cogops.tools.history.query import history_query_recent
        result = history_query_recent([])
        assert "No conversation history" in result

    def test_invalid_mode(self):
        from cogops.tools.history.query import history_query
        result = asyncio_run(history_query(mode="invalid"))
        assert "Invalid mode" in result

    def test_missing_user_id(self):
        from cogops.tools.history.query import history_query
        result = asyncio_run(history_query(mode="summarize"))
        assert "Missing user_id" in result

    def test_summarize_mode(self):
        from cogops.tools.history.query import history_query
        # Without mock Redis, it should return store unavailable
        result = asyncio_run(history_query(mode="summarize", user_id="u1"))
        assert "not available" in result or "Missing" in result


class TestAsyncHelper:
    """Helper used by all async tests."""
    pass


def asyncio_run(coro):
    """Run a coroutine synchronously."""
    import asyncio
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
