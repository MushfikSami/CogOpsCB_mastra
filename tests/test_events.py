"""test_events.py — Phase 1: channel filtering tests."""
from cogops.events.channels import filter_for_user, filter_for_debug, strip_channel


class TestFilterForUser:
    def test_includes_user_events(self):
        events = [{"channel": "user", "type": "answer_chunk"}]
        result = list(filter_for_user(events))
        assert len(result) == 1
        assert result[0]["type"] == "answer_chunk"

    def test_includes_both_events(self):
        events = [{"channel": "both", "type": "answer_complete"}]
        result = list(filter_for_user(events))
        assert len(result) == 1

    def test_excludes_debug_events(self):
        events = [
            {"channel": "debug", "type": "reasoning_chunk"},
            {"channel": "debug", "type": "tool_call"},
        ]
        result = list(filter_for_user(events))
        assert len(result) == 0

    def test_mixed_events(self):
        events = [
            {"channel": "user", "type": "a"},
            {"channel": "debug", "type": "b"},
            {"channel": "both", "type": "c"},
        ]
        result = list(filter_for_user(events))
        assert len(result) == 2
        assert {r["type"] for r in result} == {"a", "c"}

    def test_empty_input(self):
        assert list(filter_for_user([])) == []


class TestFilterForDebug:
    def test_includes_debug_events(self):
        events = [{"channel": "debug", "type": "reasoning_chunk"}]
        result = list(filter_for_debug(events))
        assert len(result) == 1

    def test_includes_both_events(self):
        events = [{"channel": "both", "type": "answer_complete"}]
        result = list(filter_for_debug(events))
        assert len(result) == 1

    def test_excludes_user_events(self):
        events = [{"channel": "user", "type": "answer_chunk"}]
        result = list(filter_for_debug(events))
        assert len(result) == 0

    def test_mixed_events(self):
        events = [
            {"channel": "user", "type": "a"},
            {"channel": "debug", "type": "b"},
            {"channel": "both", "type": "c"},
        ]
        result = list(filter_for_debug(events))
        assert len(result) == 2
        assert {r["type"] for r in result} == {"b", "c"}

    def test_empty_input(self):
        assert list(filter_for_debug([])) == []


class TestStripChannel:
    def test_removes_channel(self):
        events = [{"channel": "user", "type": "answer_chunk"}]
        result = list(strip_channel(events))
        assert "channel" not in result[0]
        assert result[0]["type"] == "answer_chunk"

    def test_preserves_other_fields(self):
        events = [{"channel": "user", "type": "a", "data": 42}]
        result = list(strip_channel(events))
        assert result[0] == {"type": "a", "data": 42}

    def test_empty_input(self):
        assert list(strip_channel([])) == []
