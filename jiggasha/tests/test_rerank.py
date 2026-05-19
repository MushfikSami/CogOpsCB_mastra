"""
jiggasha/tests/test_rerank.py

Unit tests for the in-service LLM relevance reranker. Mocks the secondary
AsyncOpenAI client so tests run offline.

The reranker's contract is the load-bearing piece:
  - One batched LLM call per request.
  - Per-query class-index output: {"<1-based sub_num>":[[pid, cls], ...]}.
  - Policy: yes core; weak backfill only for uncovered subs (per-sub cap);
    global cap on kept; cosine safety net on failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

# Allow `import rerank` from the jiggasha/ directory.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rerank import (  # noqa: E402
    CLASS_WEAK,
    CLASS_YES,
    RerankCandidate,
    RerankResult,
    run_rerank,
)


def _candidate(passage_id: int, score: float, *, sub_indices=None, **kw) -> RerankCandidate:
    return RerankCandidate(
        passage_id=passage_id,
        text=kw.get("text", f"passage {passage_id}"),
        score=score,
        category=kw.get("category", ""),
        sub_category=kw.get("sub_category", ""),
        service=kw.get("service", ""),
        topic=kw.get("topic", ""),
        sub_indices=list(sub_indices or []),
    )


def _llm_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _mock_client(content: str):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_llm_response(content)),
            ),
        ),
    )


# ============================================================
# Empty / fallback paths
# ============================================================

class TestEmptyAndFallback(unittest.TestCase):
    def test_empty_candidates(self):
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=[],
            secondary_client=_mock_client("{}"),
            secondary_model="qwen36",
        ))
        self.assertEqual(res.passages, [])
        self.assertEqual(res.per_query, {})
        self.assertFalse(res.degraded)

    def test_empty_sub_queries(self):
        res = asyncio.run(run_rerank(
            sub_queries=[],
            candidates=[_candidate(1, 0.8, sub_indices=[0])],
            secondary_client=_mock_client("{}"),
            secondary_model="qwen36",
        ))
        self.assertEqual(res.passages, [])
        self.assertEqual(res.per_query, {})

    def test_no_client_falls_back_to_cosine(self):
        cands = [
            _candidate(1, 0.80, sub_indices=[0]),
            _candidate(2, 0.60, sub_indices=[0]),
            _candidate(3, 0.40, sub_indices=[0]),  # below 0.50 floor
        ]
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=None,
            secondary_model="qwen36",
            fallback_cosine_min=0.50,
        ))
        self.assertTrue(res.degraded)
        self.assertEqual([c.passage_id for c in res.passages], [1, 2])
        self.assertEqual(res.per_query["1"], [[1, CLASS_WEAK], [2, CLASS_WEAK]])


# ============================================================
# Policy (yes / weak backfill / caps)
# ============================================================

class TestPolicy(unittest.TestCase):
    def test_yes_kept_no_dropped(self):
        cands = [
            _candidate(1, 0.80, sub_indices=[0]),
            _candidate(2, 0.70, sub_indices=[0]),  # absent from verdicts → no
            _candidate(3, 0.65, sub_indices=[0]),
        ]
        verdicts = {"1": [[1, CLASS_YES], [3, CLASS_YES]]}
        res = asyncio.run(run_rerank(
            sub_queries=["sub1?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        pids = [c.passage_id for c in res.passages]
        self.assertEqual(set(pids), {1, 3})
        self.assertEqual(set(p for p, _ in res.per_query["1"]), {1, 3})
        for _, cls in res.per_query["1"]:
            self.assertEqual(cls, CLASS_YES)

    def test_weak_included_alongside_yes_within_cap(self):
        # Sub 1 has yes (pid 1) + weak (pid 3). Sub 2 has weak only (pid 2).
        # With the per-sub cap on weak (default 3), all three pass.
        cands = [
            _candidate(1, 0.80, sub_indices=[0]),
            _candidate(2, 0.50, sub_indices=[1]),
            _candidate(3, 0.45, sub_indices=[0]),
        ]
        verdicts = {
            "1": [[1, CLASS_YES], [3, CLASS_WEAK]],
            "2": [[2, CLASS_WEAK]],
        }
        res = asyncio.run(run_rerank(
            sub_queries=["sub0?", "sub1?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        pids = [c.passage_id for c in res.passages]
        self.assertEqual(set(pids), {1, 2, 3})
        # Yes must sort before weak in the kept order.
        self.assertEqual(pids[0], 1)

    def test_weak_per_sub_cap_enforced(self):
        # 5 weak candidates all tagged sub 0; cap=2 → only 2 survive.
        cands = [
            _candidate(i + 1, 0.55 - i * 0.01, sub_indices=[0])
            for i in range(5)
        ]
        verdicts = {"1": [[i + 1, CLASS_WEAK] for i in range(5)]}
        res = asyncio.run(run_rerank(
            sub_queries=["sub0?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
            weak_per_sub_cap=2,
        ))
        self.assertEqual(len(res.passages), 2)
        self.assertEqual(
            sorted(p.passage_id for p in res.passages),
            [1, 2],
            "highest-cosine weak passages must win the cap",
        )

    def test_keep_cap_applied(self):
        cands = [
            _candidate(i + 1, 0.95 - i * 0.001, sub_indices=[0])
            for i in range(30)
        ]
        verdicts = {"1": [[i + 1, CLASS_YES] for i in range(30)]}
        res = asyncio.run(run_rerank(
            sub_queries=["sub?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
            keep_cap=10,
        ))
        self.assertEqual(len(res.passages), 10)
        # Highest cosine first.
        self.assertEqual(
            [p.passage_id for p in res.passages],
            list(range(1, 11)),
        )

    def test_passage_in_both_subs_yes_and_weak(self):
        # Pid 1 is yes for sub 1 and weak for sub 2. It should appear in
        # both per_query buckets, and contribute coverage to sub 1 only.
        # Pid 2 weak-only for sub 2 → would normally backfill, but sub 2 is
        # ALREADY covered by pid 1 (which is weak there, not yes) — so pid 2
        # SHOULD still backfill because pid 1's weak doesn't count as coverage.
        cands = [
            _candidate(1, 0.80, sub_indices=[0, 1]),
            _candidate(2, 0.60, sub_indices=[1]),
        ]
        verdicts = {
            "1": [[1, CLASS_YES]],
            "2": [[1, CLASS_WEAK], [2, CLASS_WEAK]],
        }
        res = asyncio.run(run_rerank(
            sub_queries=["sub1?", "sub2?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        pids = {c.passage_id for c in res.passages}
        self.assertEqual(pids, {1, 2})


# ============================================================
# Parsing robustness
# ============================================================

class TestRobustness(unittest.TestCase):
    def test_unknown_passage_id_dropped(self):
        cands = [_candidate(1, 0.80, sub_indices=[0])]
        verdicts = {"1": [[1, CLASS_YES], [999, CLASS_YES], [-1, CLASS_YES]]}
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        self.assertEqual([c.passage_id for c in res.passages], [1])
        self.assertEqual(res.per_query["1"], [[1, CLASS_YES]])

    def test_unknown_sub_key_dropped(self):
        # Sub key "5" when only 1 sub-query exists → drop those entries.
        cands = [_candidate(1, 0.80, sub_indices=[0])]
        verdicts = {"1": [[1, CLASS_YES]], "5": [[1, CLASS_YES]]}
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        self.assertEqual([c.passage_id for c in res.passages], [1])
        self.assertNotIn("5", res.per_query)

    def test_unknown_class_dropped(self):
        # Class 2 is invalid (only 0=yes, 1=weak). The entry must be dropped.
        cands = [_candidate(1, 0.80, sub_indices=[0]), _candidate(2, 0.70, sub_indices=[0])]
        verdicts = {"1": [[1, 2], [2, CLASS_YES]]}
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        # Only pid 2 survives.
        self.assertEqual([c.passage_id for c in res.passages], [2])

    def test_object_form_entries_accepted(self):
        # Some models emit {"id": N, "class": 0} instead of [N, 0].
        cands = [_candidate(1, 0.80, sub_indices=[0])]
        verdicts = {"1": [{"id": 1, "class": 0}]}
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        self.assertEqual([c.passage_id for c in res.passages], [1])

    def test_missing_candidate_in_verdicts_treated_as_no(self):
        # pid 2 absent → no → dropped.
        cands = [_candidate(1, 0.80, sub_indices=[0]), _candidate(2, 0.70, sub_indices=[0])]
        verdicts = {"1": [[1, CLASS_YES]]}
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps(verdicts)),
            secondary_model="qwen36",
        ))
        self.assertEqual([c.passage_id for c in res.passages], [1])


# ============================================================
# Failure fallbacks
# ============================================================

class TestFailureFallback(unittest.TestCase):
    def test_malformed_json_falls_back_to_cosine(self):
        cands = [
            _candidate(1, 0.80, sub_indices=[0]),
            _candidate(2, 0.60, sub_indices=[0]),
            _candidate(3, 0.40, sub_indices=[0]),
        ]
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client("not valid json {{"),
            secondary_model="qwen36",
            fallback_cosine_min=0.50,
        ))
        self.assertTrue(res.degraded)
        self.assertEqual([c.passage_id for c in res.passages], [1, 2])

    def test_timeout_falls_back_to_cosine(self):
        async def _hang(*_, **__):
            await asyncio.sleep(30)

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_hang),
            ),
        )
        cands = [
            _candidate(1, 0.80, sub_indices=[0]),
            _candidate(2, 0.40, sub_indices=[0]),
        ]
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=client,
            secondary_model="qwen36",
            timeout=0.2,
            fallback_cosine_min=0.50,
        ))
        self.assertTrue(res.degraded)
        self.assertEqual([c.passage_id for c in res.passages], [1])

    def test_exception_in_call_falls_back(self):
        async def _boom(*_, **__):
            raise RuntimeError("synthetic")

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_boom),
            ),
        )
        cands = [_candidate(1, 0.80, sub_indices=[0])]
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=client,
            secondary_model="qwen36",
            fallback_cosine_min=0.50,
        ))
        self.assertTrue(res.degraded)
        self.assertEqual([c.passage_id for c in res.passages], [1])

    def test_root_not_a_dict_falls_back(self):
        cands = [_candidate(1, 0.80, sub_indices=[0])]
        res = asyncio.run(run_rerank(
            sub_queries=["x?"],
            candidates=cands,
            secondary_client=_mock_client(json.dumps([1, 2, 3])),
            secondary_model="qwen36",
            fallback_cosine_min=0.50,
        ))
        self.assertTrue(res.degraded)
        self.assertEqual([c.passage_id for c in res.passages], [1])


if __name__ == "__main__":
    unittest.main()
