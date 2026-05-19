"""
tests/test_pipeline.py

Integration tests for the deterministic pipeline (cogops/agents/pipeline.py).

The pipeline now makes ONE Jiggasha POST that returns LLM-reranked passages.
We mock:
  - Jiggasha HTTP via httpx.MockTransport (returns the new multi-query shape)
  - secondary LLM (NLI verifier ONLY now) via AsyncMock
  - primary LLM (composer streaming) via a custom async-iterable mock

These tests verify the WIRING — events flow correctly, refusals fire at the
right stages, disambiguation fires under the right conditions, and the
source_map/citations/Sources-block plumbing holds. Semantic quality of the
rerank and composer is exercised by spot-checks on the live stack.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import httpx

from cogops.agents.pipeline import PipelineConfig, run_factual_pipeline
from cogops.pipeline.router import RouterResult


# ============================================================
# Test fixtures / mock helpers
# ============================================================

def _factual_router(subs: List[str], raw: Optional[str] = None) -> RouterResult:
    return RouterResult(
        intent="factual_govt",
        sub_queries_bengali=subs,
        raw_query=raw or " / ".join(subs),
    )


def _passage(pid: int, score: float, text: str, **kw) -> Dict[str, Any]:
    return {
        "passage_id": pid,
        "text": text,
        "score": score,
        "category": kw.get("category", ""),
        "sub_category": kw.get("sub_category", ""),
        "service": kw.get("service", ""),
        "topic": kw.get("topic", ""),
    }


def _jiggasha_transport(
    passages: List[Dict[str, Any]],
    rerank: Dict[str, List[List[int]]],
    degraded: bool = False,
):
    """Mock /search transport. Returns the new multi-query shape.

    The single canned response is returned for any sub_queries payload.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except Exception:
            return httpx.Response(400, json={"error": "bad json"})
        subs = body.get("sub_queries") or []
        if not subs:
            # Legacy query path — not expected in pipeline tests.
            return httpx.Response(400, json={"error": "expected sub_queries"})
        return httpx.Response(200, json={
            "sub_queries": subs,
            "passages": passages,
            "rerank": rerank,
            "degraded": degraded,
        })
    return httpx.MockTransport(handler)


def _mock_secondary_nli(nli_json: str = '{"verdicts":[]}'):
    """Secondary client now only handles NLI verifier calls (rerank moved to
    Jiggasha). Returns whatever NLI JSON the caller specifies."""
    async def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=nli_json))],
        )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def _mock_primary_stream(text_chunks: List[str]):
    """Mock primary client that streams the given chunks as composer output."""
    async def make_stream():
        for piece in text_chunks:
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=piece)
            )])

    async def create(**kwargs):
        return make_stream()

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


async def _drain(gen):
    out: List[Dict[str, Any]] = []
    async for ev in gen:
        out.append(ev)
    return out


def _types(events):
    return [e.get("type") for e in events]


# ============================================================
# Happy path
# ============================================================

class TestHappyPath(unittest.TestCase):
    def test_single_sub_question_streams_cited_answer(self):
        passages = [
            _passage(41, 0.82, "এনআইডি হারিয়ে গেলে থানায় জিডি করুন।", category="NID"),
            _passage(47, 0.79, "স্লিপ হারালে দ্বিতীয়বার আবেদন।", category="NID"),
        ]
        rerank = {"1": [[41, 0], [47, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))

        nli = json.dumps({"verdicts": [{"i": 0, "v": "entailed"}]})
        secondary = _mock_secondary_nli(nli)

        primary = _mock_primary_stream([
            "এনআইডি হারিয়ে গেলে প্রথমে থানায় জিডি করুন ",
            "[S1]।",
        ])

        cfg = PipelineConfig(verifier_enabled=True)
        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="এনআইডি কার্ড হারিয়ে গেলে কী করব?",
            router_result=_factual_router(["এনআইডি কার্ড হারিয়ে গেলে কী করব?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=cfg,
            http_client=http,
        )))
        asyncio.run(http.aclose())

        types = _types(events)
        self.assertIn("pipeline_start", types)
        self.assertIn("retrieval_done", types)
        self.assertIn("source_map_allocated", types)
        self.assertIn("composer_start", types)
        self.assertIn("answer_chunk", types)
        self.assertIn("composer_done", types)
        self.assertIn("verification_start", types)
        self.assertIn("final_answer", types)
        self.assertIn("answer_complete", types)

        final = next(e for e in events if e["type"] == "final_answer")
        self.assertIn("[S1]", final["content"])
        self.assertIn("সূত্র", final["content"])
        for tag, meta in final["source_map"].items():
            self.assertNotIn("text", meta)
            self.assertIn("passage_id", meta)


# ============================================================
# Refusal paths
# ============================================================

class TestRefusalPaths(unittest.TestCase):
    def test_empty_jiggasha_emits_refusal_no_llm(self):
        http = httpx.AsyncClient(transport=_jiggasha_transport([], {"1": []}))
        secondary_create = AsyncMock(side_effect=AssertionError("secondary must not be called"))
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=secondary_create)),
        )
        primary_create = AsyncMock(side_effect=AssertionError("primary must not be called"))
        primary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=primary_create)),
        )

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="x?",
            router_result=_factual_router(["x?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        final = next(e for e in events if e["type"] == "final_answer")
        self.assertEqual(final["reason"], "no_passages")
        self.assertIn("নির্ভরযোগ্য", final["content"])
        secondary_create.assert_not_awaited()
        primary_create.assert_not_awaited()

    def test_jiggasha_http_failure_emits_refusal(self):
        def boom(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="service unavailable")
        http = httpx.AsyncClient(transport=httpx.MockTransport(boom))

        primary_create = AsyncMock(side_effect=AssertionError("primary must not be called"))
        primary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=primary_create)),
        )
        secondary_create = AsyncMock(side_effect=AssertionError("secondary must not be called"))
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=secondary_create)),
        )

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="x?",
            router_result=_factual_router(["x?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        final = next(e for e in events if e["type"] == "final_answer")
        self.assertEqual(final["reason"], "jiggasha_failed")

    def test_composer_with_no_real_citations_falls_back_to_refusal(self):
        passages = [_passage(5, 0.82, "real passage text", topic="real")]
        rerank = {"1": [[5, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["just some prose with [S99] only."])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="q?",
            router_result=_factual_router(["q?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())
        final = next(e for e in events if e["type"] == "final_answer")
        self.assertIn("নির্ভরযোগ্য", final["content"])
        self.assertTrue(any(
            e["type"] == "unsupported_claim" and e.get("tag") == "S99"
            for e in events
        ))


# ============================================================
# Multi-question
# ============================================================

class TestMultiQuestion(unittest.TestCase):
    def test_two_subs_returns_three_passages(self):
        passages = [
            _passage(10, 0.70, "passport-fee passage", category="পাসপোর্ট"),
            _passage(20, 0.65, "shared passage", category="অন্য"),
            _passage(30, 0.55, "nid-correct passage", category="NID"),
        ]
        rerank = {
            "1": [[10, 0], [20, 0]],
            "2": [[20, 0], [30, 0]],
        }
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream([
            "ফি ৪০২৫ [S1]। সংশোধন কেন্দ্রে [S3]।",
        ])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="পাসপোর্ট ফি কত? এনআইডি সংশোধন কোথায়?",
            router_result=_factual_router([
                "পাসপোর্ট ফি কত?",
                "এনআইডি সংশোধন কোথায়?",
            ]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        alloc = next(e for e in events if e["type"] == "source_map_allocated")
        self.assertEqual(alloc["n_sources"], 3)
        self.assertEqual(set(alloc["tags"]), {"S1", "S2", "S3"})

        final = next(e for e in events if e["type"] == "final_answer")
        self.assertIn("[S1]", final["content"])
        self.assertIn("[S3]", final["content"])

    def test_source_map_records_sub_indices_from_rerank(self):
        """A passage that's yes for both subs gets sub_indices=[0,1]."""
        passages = [_passage(99, 0.70, "shared", category="X")]
        rerank = {"1": [[99, 0]], "2": [[99, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["answer [S1]।"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="a? b?",
            router_result=_factual_router(["a?", "b?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        final = next(e for e in events if e["type"] == "final_answer")
        # source_map should record sub_indices=[0,1] for S1.
        meta = final["source_map"]["S1"]
        self.assertEqual(meta["sub_indices"], [0, 1])
        self.assertEqual(meta["verdict"], "yes")


# ============================================================
# Composer empty
# ============================================================

class TestComposerEmpty(unittest.TestCase):
    def test_composer_empty_emits_refusal(self):
        passages = [_passage(5, 0.82, "real text", category="X")]
        rerank = {"1": [[5, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["", "", ""])  # all empty

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="q?",
            router_result=_factual_router(["q?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())
        final = next(e for e in events if e["type"] == "final_answer")
        self.assertEqual(final["reason"], "composer_empty")


# ============================================================
# Channel discipline
# ============================================================

class TestChannelDiscipline(unittest.TestCase):
    """User-channel events must be only answer_chunk / final_answer / answer_complete / error."""

    def test_only_answer_events_are_visible(self):
        passages = [_passage(5, 0.82, "real text", category="X")]
        rerank = {"1": [[5, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["answer [S1]।"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="q?",
            router_result=_factual_router(["q?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        visible_types = {"answer_chunk", "final_answer", "answer_complete"}
        for ev in events:
            if ev.get("channel") in ("both", "answer") and ev["type"] not in visible_types:
                self.fail(f"Event {ev['type']} leaks to user channel (channel={ev['channel']})")


# ============================================================
# Disambiguation
# ============================================================

class TestDisambiguation(unittest.TestCase):
    def test_short_intent_with_multiple_services_triggers_disambiguate(self):
        # Short normalized intent ("সার্টিফিকেট লাগবে?") + 3 distinct services.
        passages = [
            _passage(101, 0.80, "জন্ম সনদ আবেদন...",
                     category="সনদ", sub_category="জন্ম সনদ", service="জন্ম সনদ"),
            _passage(102, 0.78, "চারিত্রিক সনদ পুলিশ ভেরিফিকেশন...",
                     category="সনদ", sub_category="চারিত্রিক সনদ", service="চারিত্রিক সনদ"),
            _passage(103, 0.75, "বিবাহ সনদ কাজী অফিস...",
                     category="সনদ", sub_category="বিবাহ সনদ", service="বিবাহ সনদ"),
        ]
        rerank = {"1": [[101, 0], [102, 0], [103, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["কোনটি জানতে চান? [S1] [S2] [S3]"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="সার্টিফিকেট লাগবে?",
            router_result=_factual_router(["সার্টিফিকেট লাগবে?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        disamb_events = [e for e in events if e["type"] == "disambiguate_required"]
        self.assertEqual(len(disamb_events), 1)
        self.assertEqual(disamb_events[0]["n_candidates"], 3)
        self.assertEqual(set(disamb_events[0]["tags"]), {"S1", "S2", "S3"})

    def test_long_raw_short_normalized_intent_triggers_disambiguate(self):
        """Verbose raw query, but the router boils it down to a short intent
        — disambiguation must still fire on the normalized intent."""
        passages = [
            _passage(201, 0.80, "ঢাকা বোর্ডে নাম সংশোধন ৫০০ টাকা",
                     category="শিক্ষা", sub_category="ঢাকা বোর্ড নাম সংশোধন",
                     service="ঢাকা বোর্ড নাম সংশোধন"),
            _passage(202, 0.78, "চট্টগ্রাম বোর্ডে নাম সংশোধন ৮০০ টাকা",
                     category="শিক্ষা", sub_category="চট্টগ্রাম বোর্ড নাম সংশোধন",
                     service="চট্টগ্রাম বোর্ড নাম সংশোধন"),
            _passage(203, 0.75, "কুমিল্লা বোর্ডে নাম সংশোধন",
                     category="শিক্ষা", sub_category="কুমিল্লা বোর্ড নাম সংশোধন",
                     service="কুমিল্লা বোর্ড নাম সংশোধন"),
        ]
        rerank = {"1": [[201, 0], [202, 0], [203, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["কোন বোর্ড? [S1] [S2] [S3]"])

        verbose_raw = (
            "ভাই আমার একটা ছোট্ট সমস্যা — আমি আমার এসএসসি সনদে "
            "আমার নাম পরিবর্তন করতে চাই, কীভাবে করবো বলতে পারবেন?"
        )
        normalized_intent = "এসএসসিতে নাম পরিবর্তন কীভাবে?"   # ~5 tokens

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query=verbose_raw,
            router_result=_factual_router([normalized_intent], raw=verbose_raw),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        disamb_events = [e for e in events if e["type"] == "disambiguate_required"]
        self.assertEqual(len(disamb_events), 1,
                         "long raw + short normalized intent must still disambiguate")
        self.assertEqual(disamb_events[0]["n_candidates"], 3)

    def test_long_normalized_intent_does_not_trigger_disambiguate(self):
        # When the router's normalized sub-question is itself long/specific,
        # disambiguation should NOT fire even if multiple services come back.
        passages = [
            _passage(301, 0.80, "জন্ম সনদ আবেদন",
                     category="সনদ", sub_category="জন্ম সনদ", service="জন্ম সনদ"),
            _passage(302, 0.78, "চারিত্রিক সনদ পুলিশ",
                     category="সনদ", sub_category="চারিত্রিক সনদ", service="চারিত্রিক সনদ"),
        ]
        rerank = {"1": [[301, 0], [302, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["specific answer [S1]।"])

        long_sub = (
            "আমার নতুন চাকরিতে যোগ দিতে চারিত্রিক সনদ "
            "অনলাইনে আবেদন করার ধাপগুলো কী?"
        )
        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query=long_sub,
            router_result=_factual_router([long_sub]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        self.assertFalse(
            any(e["type"] == "disambiguate_required" for e in events),
            "disambiguate must not fire when the normalized intent is long/specific",
        )

    def test_multi_sub_query_never_disambiguates(self):
        # User pre-split their question into 2 sub-questions: skip disambig.
        passages = [
            _passage(401, 0.80, "জন্ম সনদ ফি", category="সনদ",
                     sub_category="জন্ম সনদ", service="জন্ম সনদ"),
            _passage(402, 0.78, "এনআইডি সংশোধন", category="NID",
                     sub_category="সংশোধন", service="NID সংশোধন"),
        ]
        rerank = {"1": [[401, 0]], "2": [[402, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["sub1 [S1]।\n\nsub2 [S2]।"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="জন্ম সনদ ফি? এনআইডি সংশোধন?",
            router_result=_factual_router(["জন্ম সনদ ফি?", "এনআইডি সংশোধন?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        self.assertFalse(
            any(e["type"] == "disambiguate_required" for e in events),
            "multi-sub queries must skip disambiguation",
        )

    def test_single_service_does_not_trigger_disambiguate(self):
        # Short query, only one (cat, sub_cat) tuple → no disambig.
        passages = [
            _passage(501, 0.80, "এনআইডি স্ট্যাটাস",
                     category="NID", sub_category="স্ট্যাটাস", service="NID স্ট্যাটাস"),
            _passage(502, 0.78, "এনআইডি ডাউনলোড",
                     category="NID", sub_category="স্ট্যাটাস", service="NID স্ট্যাটাস"),
        ]
        rerank = {"1": [[501, 0], [502, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["এনআইডি যেভাবে... [S1]।"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="NID কীভাবে?",
            router_result=_factual_router(["NID কীভাবে?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        self.assertFalse(
            any(e["type"] == "disambiguate_required" for e in events),
            "disambiguate must not fire when only one (cat, sub_cat) tuple is in the yes-set",
        )

    def test_same_sub_category_with_aspect_variations_does_not_disambiguate(self):
        """Regression: corpora often encode aspects (price, time, language, …)
        as distinct `service` strings within the same (category, sub_category).
        These describe the same underlying service and should NOT trigger
        disambiguation."""
        passages = [
            _passage(601, 0.80, "চারিত্রিক সনদের ভাষা",
                     category="জরুরি প্রত্যয়ন ও সনদ", sub_category="জরুরি প্রত্যয়ন",
                     service="চারিত্রিক সনদ: ভাষা"),
            _passage(602, 0.78, "চারিত্রিক সনদের সেবারমূল্য",
                     category="জরুরি প্রত্যয়ন ও সনদ", sub_category="জরুরি প্রত্যয়ন",
                     service="চারিত্রিক সনদ: সেবারমূল্য"),
            _passage(603, 0.75, "চারিত্রিক সনদের সময়সীমা",
                     category="জরুরি প্রত্যয়ন ও সনদ", sub_category="জরুরি প্রত্যয়ন",
                     service="চারিত্রিক সনদ: সময়সীমা"),
        ]
        rerank = {"1": [[601, 0], [602, 0], [603, 0]]}
        http = httpx.AsyncClient(transport=_jiggasha_transport(passages, rerank))
        secondary = _mock_secondary_nli()
        primary = _mock_primary_stream(["চারিত্রিক সনদ: ... [S1] [S2] [S3]।"])

        events = asyncio.run(_drain(run_factual_pipeline(
            raw_query="চারিত্রিক সনদ?",
            router_result=_factual_router(["চারিত্রিক সনদ?"]),
            history=[],
            primary_client=primary,
            primary_model="qwen36",
            secondary_client=secondary,
            secondary_model="qwen36",
            cfg=PipelineConfig(verifier_enabled=False),
            http_client=http,
        )))
        asyncio.run(http.aclose())

        self.assertFalse(
            any(e["type"] == "disambiguate_required" for e in events),
            "aspect variations of the same service must not trigger disambig",
        )


# ============================================================
# Mode-mix paragraph stripper
# ============================================================

class TestModeMixStrip(unittest.TestCase):
    """The composer occasionally writes a "but here's the general procedure"
    paragraph after a partial-gap caveat — mode-mixing. _post_flight strips
    that bridge sentence + paragraph deterministically."""

    def test_strips_bridge_and_paragraph_keeping_bullets(self):
        from cogops.agents.pipeline import _strip_mode_mix_paragraph
        txt = (
            "প্রদত্ত তথ্যে গাড়ি পার্কিং এরিয়ায় মামলা তোলার নির্দিষ্ট পদ্ধতি উল্লেখ নেই। "
            "তবে সাধারণ আইনানুগ মামলা তোলার পদ্ধতি নিচে দেওয়া হলো:\n\n"
            "মামলা তোলার জন্য প্রথমে ফৌজদারি মামলা [S1]। ফৌজদারি মামলা [S1]।\n\n"
            "এই নির্দিষ্ট বিষয়ে সঠিক তথ্য পাওয়া যায়নি — কাছাকাছি বিষয়ে যা পাওয়া গেছে:\n"
            "- এফআইআর কীভাবে হয় [S2]"
        )
        out, stripped = _strip_mode_mix_paragraph(txt)
        self.assertTrue(stripped)
        self.assertNotIn("সাধারণ আইনানুগ", out)
        self.assertNotIn("ফৌজদারি মামলা সাধারণত থানায়", out)
        self.assertIn("নির্দিষ্ট পদ্ধতি উল্লেখ নেই", out)
        self.assertIn("কাছাকাছি বিষয়ে যা পাওয়া গেছে", out)
        self.assertIn("[S2]", out)

    def test_strips_bridge_with_bullet_procedure_no_b_header(self):
        from cogops.agents.pipeline import _strip_mode_mix_paragraph
        txt = (
            "প্রদত্ত তথ্যে নির্দিষ্ট পদ্ধতি উল্লেখ নেই [S1]। "
            "তবে সাধারণভাবে মামলা তুলে নেওয়ার পদ্ধতি নিচে দেওয়া হলো:\n\n"
            "*   মামলা তুলে নেওয়ার জন্য সংশ্লিষ্ট আদালতে [S1]।\n"
            "*   ফৌজদারি মামলায় [S1]।"
        )
        out, stripped = _strip_mode_mix_paragraph(txt)
        self.assertTrue(stripped)
        self.assertNotIn("সাধারণভাবে", out)
        self.assertNotIn("সংশ্লিষ্ট আদালতে", out)
        self.assertIn("নির্দিষ্ট পদ্ধতি উল্লেখ নেই", out)

    def test_pure_b_shape_not_modified(self):
        from cogops.agents.pipeline import _strip_mode_mix_paragraph
        txt = (
            "এই নির্দিষ্ট বিষয়ে সঠিক তথ্য পাওয়া যায়নি — কাছাকাছি বিষয়ে যা পাওয়া গেছে:\n"
            "- দিনাজপুর [S1]\n"
            "- রাজশাহী [S2]\n"
            "উপরের কোনো বিষয়ে বিস্তারিত জানতে চাইলে আবার জিজ্ঞাসা করুন।"
        )
        out, stripped = _strip_mode_mix_paragraph(txt)
        self.assertFalse(stripped)
        self.assertEqual(out, txt)

    def test_pure_a_direct_not_modified(self):
        from cogops.agents.pipeline import _strip_mode_mix_paragraph
        txt = "এনআইডিতে ড./পদবি যুক্ত করার সুযোগ নেই [S1]।"
        out, stripped = _strip_mode_mix_paragraph(txt)
        self.assertFalse(stripped)
        self.assertEqual(out, txt)

    def test_strip_sources_harvests_tags_when_body_uncited(self):
        from cogops.agents.pipeline import _strip_composer_sources_block
        # Composer wrote a perfect answer but put all tags in a trailing
        # Sources block instead of inline. Without harvest the cleaned body
        # has 0 [S#] tags and _post_flight would force a refusal.
        txt = (
            "হ্যাঁ, আপনি প্রতিবন্ধী সনদের জন্য আবেদন করতে পারেন।\n\n"
            "ধাপ ১: অনলাইন আবেদন।\nধাপ ২: তথ্য পূরণ।\n\n"
            "---\n**সূত্র (Sources)**\n- [S1] প্রতিবন্ধী সনদ\n- [S2] শনাক্তকরণ"
        )
        out = _strip_composer_sources_block(txt)
        # Tags from the trailing block harvested onto the last non-empty line.
        self.assertIn("[S1]", out)
        self.assertIn("[S2]", out)
        # The Sources-block scaffolding is gone.
        self.assertNotIn("সূত্র", out)
        self.assertNotIn("---", out)

    def test_strip_sources_no_harvest_when_body_has_inline_cites(self):
        from cogops.agents.pipeline import _strip_composer_sources_block
        # Body already has inline tags — strip the block, no harvest needed.
        txt = (
            "নতুন পাসপোর্টের ফি ৪০২৫ টাকা [S1]।\n\n"
            "---\n**সূত্র (Sources)**\n- [S1] পাসপোর্ট ফি"
        )
        out = _strip_composer_sources_block(txt)
        self.assertEqual(out.count("[S1]"), 1)
        self.assertNotIn("সূত্র", out)

    def test_handles_ullikhito_noy_variant(self):
        from cogops.agents.pipeline import _strip_mode_mix_paragraph
        txt = (
            "ফ্যামিলি কার্ড ছাড়া উপায় এই প্রদত্ত তথ্যে উল্লিখিত নয় [S1]। "
            "তবে, কার্ডের ব্যবহারের পদ্ধতি নিচে দেওয়া হলো:\n\n"
            "*   সেল্ফ-সার্ভিস [S2]।\n"
            "*   কার্ড অ্যাক্টিভেশন [S3]।"
        )
        out, stripped = _strip_mode_mix_paragraph(txt)
        self.assertTrue(stripped)
        self.assertNotIn("সেল্ফ-সার্ভিস", out)


if __name__ == "__main__":
    unittest.main()
