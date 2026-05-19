"""
Tests for the bulletproof grounding pipeline:

- Tool plugin discovery and Jiggasha HTTP client (mocked).
- Intent classifier verdicts + domain-vocab override.
- NLI verifier batched call + fail-soft.
- Policy module redact/refuse/warn decisions.
- Orchestrator end-to-end with mocked tool + LLMs.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# =============================================================================
# Tool registry discovery (extends test_core.TestToolRegistry with negative cases)
# =============================================================================

class TestRegistryDiscovery:
    def test_wikipedia_and_websearch_loadable(self):
        from cogops.tools.registry import build_tool_registry
        schemas, tool_map = build_tool_registry(enabled=["wikipedia", "websearch"])
        names = [s["function"]["name"] for s in schemas]
        assert "search_wikipedia" in names
        assert "search_web" in names
        assert len(tool_map) == 2

    def test_per_tool_configure_hook_invoked(self):
        from cogops.tools.registry import build_tool_registry
        from cogops.tools import jiggasha
        original = dict(jiggasha._CONFIG)
        try:
            build_tool_registry(
                enabled=["jiggasha"],
                tool_configs={"jiggasha": {"min_score": 0.99}},
            )
            assert jiggasha._CONFIG["min_score"] == 0.99
        finally:
            jiggasha._CONFIG.clear()
            jiggasha._CONFIG.update(original)


# =============================================================================
# Jiggasha HTTP client (mocked)
# =============================================================================

class _MockHttpxResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _MockAsyncClient:
    def __init__(self, response_payload: Dict[str, Any], *, raises: Exception | None = None):
        self._payload = response_payload
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, *args, **kwargs):
        if self._raises is not None:
            raise self._raises
        return _MockHttpxResponse(self._payload)


class TestJiggashaHandler:
    @pytest.fixture
    def reset_config(self):
        """Ensure JIGGASHA_ENDPOINT and module config are set for tests."""
        import os
        from cogops.tools import jiggasha
        original_env = os.environ.get("JIGGASHA_ENDPOINT")
        original_cfg = dict(jiggasha._CONFIG)
        os.environ["JIGGASHA_ENDPOINT"] = "http://test.local/search"
        yield
        if original_env is None:
            os.environ.pop("JIGGASHA_ENDPOINT", None)
        else:
            os.environ["JIGGASHA_ENDPOINT"] = original_env
        jiggasha._CONFIG.clear()
        jiggasha._CONFIG.update(original_cfg)

    def test_returns_no_relevant_when_below_threshold(self, reset_config):
        from cogops.tools.jiggasha import handler, NO_RELEVANT_RESULTS
        from cogops.tools.registry import ToolContext

        payload = {"results": [
            {"passage_id": 1, "text": "low-score", "score": 0.05, "category": "x", "topic": "y"},
            {"passage_id": 2, "text": "also low", "score": 0.10, "category": "x", "topic": "y"},
        ]}
        ctx = ToolContext()

        with patch("cogops.tools.jiggasha.httpx.AsyncClient",
                   lambda *a, **k: _MockAsyncClient(payload)):
            content, sources = asyncio.run(handler(query="q", ctx=ctx))

        assert content == NO_RELEVANT_RESULTS
        assert sources == []
        assert ctx.source_map == {}

    def test_tags_passages_and_populates_source_map(self, reset_config):
        from cogops.tools.jiggasha import handler
        from cogops.tools.registry import ToolContext

        payload = {"results": [
            {"passage_id": 11, "text": "fee is 4025", "score": 0.8,
             "category": "পাসপোর্ট", "topic": "ফি"},
            {"passage_id": 22, "text": "apply online", "score": 0.7,
             "category": "পাসপোর্ট", "topic": "আবেদন"},
        ]}
        ctx = ToolContext()

        with patch("cogops.tools.jiggasha.httpx.AsyncClient",
                   lambda *a, **k: _MockAsyncClient(payload)):
            content, sources = asyncio.run(handler(query="passport fee", ctx=ctx))

        # Both passages tagged S1, S2
        assert "[S1]" in content
        assert "[S2]" in content
        assert "fee is 4025" in content
        assert "apply online" in content
        # Source map populated
        assert set(ctx.source_map.keys()) == {"S1", "S2"}
        assert ctx.source_map["S1"]["passage_id"] == 11
        assert ctx.source_map["S2"]["passage_id"] == 22
        # Telemetry sources mirror source_map
        assert len(sources) == 2

    def test_http_error_returns_error_sentinel(self, reset_config):
        from cogops.tools.jiggasha import handler
        from cogops.tools.registry import ToolContext

        err = httpx.ConnectError("conn refused")
        ctx = ToolContext()

        with patch("cogops.tools.jiggasha.httpx.AsyncClient",
                   lambda *a, **k: _MockAsyncClient({}, raises=err)):
            content, sources = asyncio.run(handler(query="q", ctx=ctx))

        assert content.startswith("ERROR:")
        assert sources == []
        assert ctx.source_map == {}

    def test_score_filter_below_threshold_drops(self, reset_config):
        from cogops.tools.jiggasha import handler
        from cogops.tools.registry import ToolContext

        payload = {"results": [
            {"passage_id": 1, "text": "high", "score": 0.9, "category": "a", "topic": "b"},
            {"passage_id": 2, "text": "low", "score": 0.1, "category": "c", "topic": "d"},
        ]}
        ctx = ToolContext()

        with patch("cogops.tools.jiggasha.httpx.AsyncClient",
                   lambda *a, **k: _MockAsyncClient(payload)):
            content, _ = asyncio.run(handler(query="q", ctx=ctx))

        assert "high" in content
        assert "low" not in content
        assert list(ctx.source_map.keys()) == ["S1"]


# =============================================================================
# Intent classifier
# =============================================================================

def _mock_secondary_returning(intent_str: str):
    """Build a mock AsyncOpenAI-like client whose chat.completions.create
    returns the given JSON-string intent."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps({"intent": intent_str})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


class TestIntentClassifier:
    def test_chitchat_passthrough(self):
        from cogops.verifier.intent import classify_intent
        client = _mock_secondary_returning("chitchat")
        result = asyncio.run(classify_intent("হ্যালো", client, "model"))
        assert result == "chitchat"

    def test_factual_passthrough(self):
        from cogops.verifier.intent import classify_intent
        client = _mock_secondary_returning("factual")
        result = asyncio.run(classify_intent("কিছু একটা প্রশ্ন", client, "model"))
        assert result == "factual"

    def test_political_keyword_short_circuits_to_refuse(self):
        from cogops.verifier.intent import classify_intent
        client = _mock_secondary_returning("chitchat")  # classifier output ignored
        # Must hit the refuse keyword shortcut before any LLM call.
        result = asyncio.run(classify_intent("আওয়ামী লীগ না বিএনপি?", client, "model"))
        assert result == "refuse"
        client.chat.completions.create.assert_not_called()

    def test_domain_vocab_override_factual(self):
        from cogops.verifier.intent import classify_intent
        # Classifier says chitchat, but message contains "পাসপোর্ট" →
        # belt-and-braces override should force "factual".
        client = _mock_secondary_returning("chitchat")
        result = asyncio.run(classify_intent(
            "পাসপোর্ট সম্পর্কে কিছু বলো", client, "model",
        ))
        assert result == "factual"

    def test_llm_error_defaults_to_factual(self):
        from cogops.verifier.intent import classify_intent
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = asyncio.run(classify_intent("কিছু একটা", client, "model"))
        assert result == "factual"

    def test_empty_text_yields_chitchat(self):
        from cogops.verifier.intent import classify_intent
        client = _mock_secondary_returning("factual")
        result = asyncio.run(classify_intent("", client, "model"))
        assert result == "chitchat"
        client.chat.completions.create.assert_not_called()


# =============================================================================
# NLI verifier (batched, fail-soft)
# =============================================================================

class TestNLIVerifier:
    def _client_returning(self, verdicts_payload: Dict[str, Any]):
        client = MagicMock()
        msg = MagicMock()
        msg.content = json.dumps(verdicts_payload)
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=resp)
        return client

    def test_single_batched_call_for_multiple_pairs(self):
        from cogops.verifier.nli import verify_claims
        pairs = [("S1", "claim A"), ("S2", "claim B"), ("S3", "claim C")]
        source_map = {
            "S1": {"text": "evidence for A"},
            "S2": {"text": "evidence for B"},
            "S3": {"text": "evidence for C"},
        }
        client = self._client_returning({
            "verdicts": [
                {"i": 0, "v": "entailed"},
                {"i": 1, "v": "not_entailed"},
                {"i": 2, "v": "partial"},
            ],
        })

        verdicts, usage = asyncio.run(verify_claims(pairs, source_map, client, "model"))

        # ONE LLM call total, regardless of pair count.
        assert client.chat.completions.create.call_count == 1
        assert verdicts == ["entailed", "not_entailed", "partial"]
        # MagicMock auto-vivifies `usage` so the extractor returns a dict with
        # ints. Either None (real-world no-usage) or a dict is acceptable here.
        assert usage is None or isinstance(usage, dict)

    def test_unknown_tag_short_circuits_not_entailed_without_llm(self):
        from cogops.verifier.nli import verify_claims
        pairs = [("S99", "bogus claim")]  # S99 not in source_map
        source_map = {"S1": {"text": "real"}}
        client = self._client_returning({"verdicts": []})

        verdicts, usage = asyncio.run(verify_claims(pairs, source_map, client, "model"))

        assert verdicts == ["not_entailed"]
        assert usage is None
        # No LLM call needed when ALL pairs are bogus-tag fast-path.
        client.chat.completions.create.assert_not_called()

    def test_timeout_defaults_to_entailed(self):
        from cogops.verifier.nli import verify_claims
        pairs = [("S1", "claim A")]
        source_map = {"S1": {"text": "evidence"}}
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

        verdicts, usage = asyncio.run(verify_claims(pairs, source_map, client, "model", timeout=0.1))

        assert verdicts == ["entailed"]   # fail-soft
        assert usage is None   # call never returned a usage payload

    def test_empty_pairs_returns_empty(self):
        from cogops.verifier.nli import verify_claims
        verdicts, usage = asyncio.run(verify_claims([], {}, MagicMock(), "model"))
        assert verdicts == []
        assert usage is None


# =============================================================================
# Policy module
# =============================================================================

class TestPolicy:
    def test_redact_replaces_not_entailed_sentence(self):
        from cogops.verifier.policy import apply_policy, UNSUPPORTED_REDACTION_BN
        answer = "fact one [S1]। fact two [S2]।"
        pairs = [("S1", "fact one [S1]।"), ("S2", "fact two [S2]।")]
        verdicts = ["entailed", "not_entailed"]
        final, events = apply_policy(answer, pairs, verdicts, policy="redact",
                                     refusal_text="REFUSED")

        assert "fact one" in final
        assert "fact two" not in final
        assert UNSUPPORTED_REDACTION_BN in final
        # An unsupported_claim event was emitted for the redacted sentence.
        actions = [e.get("action") for e in events if e.get("type") == "unsupported_claim"]
        assert "redacted" in actions

    def test_redact_escalates_to_refusal_when_majority_fail(self):
        from cogops.verifier.policy import apply_policy
        answer = "f1 [S1]। f2 [S2]। f3 [S3]।"
        pairs = [
            ("S1", "f1 [S1]।"),
            ("S2", "f2 [S2]।"),
            ("S3", "f3 [S3]।"),
        ]
        verdicts = ["not_entailed", "not_entailed", "entailed"]  # 2/3 = 66% fail
        final, events = apply_policy(answer, pairs, verdicts, policy="redact",
                                     refusal_text="REFUSED")
        assert final == "REFUSED"
        actions = [e.get("action") for e in events if e.get("type") == "verification_result"]
        assert actions == ["escalated_refusal"]

    def test_refuse_policy_any_failure_triggers_refusal(self):
        from cogops.verifier.policy import apply_policy
        pairs = [("S1", "claim a"), ("S2", "claim b")]
        verdicts = ["entailed", "not_entailed"]
        final, _ = apply_policy("a [S1]। b [S2]।", pairs, verdicts,
                                 policy="refuse", refusal_text="REFUSED")
        assert final == "REFUSED"

    def test_warn_policy_keeps_answer(self):
        from cogops.verifier.policy import apply_policy
        original = "fact [S1]।"
        pairs = [("S1", "fact [S1]।")]
        verdicts = ["not_entailed"]
        final, events = apply_policy(original, pairs, verdicts, policy="warn",
                                     refusal_text="REFUSED")
        assert final == original
        types_ = {e.get("type") for e in events}
        assert "unsupported_claim" in types_

    def test_all_entailed_no_modifications(self):
        from cogops.verifier.policy import apply_policy
        original = "a [S1]। b [S2]।"
        pairs = [("S1", "a [S1]।"), ("S2", "b [S2]।")]
        verdicts = ["entailed", "entailed"]
        final, events = apply_policy(original, pairs, verdicts, policy="redact",
                                     refusal_text="REFUSED")
        assert final == original
        # No unsupported_claim events for entailed claims.
        assert not any(e.get("type") == "unsupported_claim" for e in events)


# =============================================================================
# Orchestrator end-to-end with mocked dependencies
# =============================================================================

class _AsyncIter:
    """Helper: turn a list of items into an async iterator."""
    def __init__(self, items: List[Any]):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _streaming_chunk(content: str):
    """Construct an OpenAI-streaming-style chunk with text content."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _nonstream_response(content: str, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_call(call_id: str, name: str, args: Dict[str, Any]):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


class TestOrchestratorPipeline:
    """Integration tests that exercise the full process_query pipeline with
    mocked tool + primary + secondary LLMs."""

    def _build_orchestrator(self, *, intent="factual", verifier_enabled=False,
                            intent_classifier_enabled=False):
        from cogops.agents.orchestrator import Orchestrator
        o = Orchestrator()
        o.intent_classifier_enabled = intent_classifier_enabled
        o.verifier_enabled = verifier_enabled
        return o

    @pytest.mark.skipif(
        # Orchestrator needs an OpenAI client to instantiate; skip if env missing.
        not __import__("os").environ.get("LLM_API_KEY"),
        reason="Orchestrator integration needs LLM env vars",
    )
    def test_factual_query_emits_final_answer_with_sources_block(self):
        """LLM returns a tool call → tool returns S1/S2 → LLM finalizes with
        citations → orchestrator appends Sources block + emits final_answer."""
        o = self._build_orchestrator()

        # Primary LLM: 1st non-stream call returns a tool_call,
        #              2nd non-stream call returns the final text (no tool_calls),
        #              streaming call re-emits the same text token-wise.
        tool_call_resp = _nonstream_response("",
            tool_calls=[_tool_call("c1", "search_gov_services", {"query": "passport fee"})])
        nonstream_calls = {"count": 0}

        async def primary_create(*args, **kwargs):
            stream = kwargs.get("stream", False)
            if stream:
                return _AsyncIter([
                    _streaming_chunk("ফি ৪০২৫ টাকা [S1]। আবেদন অনলাইনে [S2]।"),
                ])
            nonstream_calls["count"] += 1
            if nonstream_calls["count"] == 1:
                return tool_call_resp
            return _nonstream_response("ফি ৪০২৫ টাকা [S1]। আবেদন অনলাইনে [S2]।")

        o.llm_service.client_llm = MagicMock()
        o.llm_service.client_llm.chat.completions.create = AsyncMock(side_effect=primary_create)
        o.llm_service.llm_config.model = "mock-model"

        # Mock the bound tool handler to populate source_map
        async def mock_handler(query: str, ctx):
            ctx.allocate_source_tag({
                "tool": "search_gov_services", "passage_id": 11, "text": "fee is 4025",
                "category": "পাসপোর্ট", "topic": "ফি", "score": 0.9,
            })
            ctx.allocate_source_tag({
                "tool": "search_gov_services", "passage_id": 22, "text": "apply online",
                "category": "পাসপোর্ট", "topic": "আবেদন", "score": 0.8,
            })
            return "[S1] fee is 4025\n\n[S2] apply online", []

        o.raw_tool_map = {"search_gov_services": mock_handler}

        async def run():
            events = []
            async for e in o.process_query("passport fee?", user_id=None):
                events.append(e)
            return events

        events = asyncio.run(run())
        finals = [e for e in events if e.get("type") == "final_answer"]
        assert len(finals) == 1, f"expected 1 final_answer, got events: {[e.get('type') for e in events]}"
        final = finals[0]["content"]
        assert "[S1]" in final
        assert "[S2]" in final
        assert "সূত্র (Sources)" in final
