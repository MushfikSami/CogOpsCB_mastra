"""
jiggasha/tests/test_service.py

Unit tests for Jiggasha service helpers (threshold, budget, merge, hit mapping).
"""

from __future__ import annotations

import sys
import os
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from service import _apply_threshold_and_budget, _merge_candidates_raw, _hit_to_passage


class TestApplyThresholdAndBudget(unittest.TestCase):
    def test_all_pass_threshold(self):
        candidates = [
            {"passage_id": 1, "score": 0.85, "text": "a" * 30, "llm_token_count": 10},
            {"passage_id": 2, "score": 0.80, "text": "b" * 30, "llm_token_count": 10},
            {"passage_id": 3, "score": 0.75, "text": "c" * 30, "llm_token_count": 10},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 100)
        self.assertEqual(len(result), 3)
        self.assertEqual([r["passage_id"] for r in result], [1, 2, 3])

    def test_some_below_threshold_filtered(self):
        candidates = [
            {"passage_id": 1, "score": 0.85, "text": "a", "llm_token_count": 1},
            {"passage_id": 2, "score": 0.68, "text": "b", "llm_token_count": 1},
            {"passage_id": 3, "score": 0.72, "text": "c", "llm_token_count": 1},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 100)
        pids = [r["passage_id"] for r in result]
        self.assertEqual(pids, [1, 3])

    def test_fallback_when_nothing_passes_threshold(self):
        candidates = [
            {"passage_id": 1, "score": 0.60, "text": "a", "llm_token_count": 1},
            {"passage_id": 2, "score": 0.55, "text": "b", "llm_token_count": 1},
            {"passage_id": 3, "score": 0.50, "text": "c", "llm_token_count": 1},
            {"passage_id": 4, "score": 0.40, "text": "d", "llm_token_count": 1},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 100)
        pids = [r["passage_id"] for r in result]
        self.assertEqual(pids, [1, 2, 3])

    def test_budget_truncation(self):
        candidates = [
            {"passage_id": 1, "score": 0.90, "text": "a" * 30, "llm_token_count": 15},
            {"passage_id": 2, "score": 0.85, "text": "b" * 30, "llm_token_count": 15},
            {"passage_id": 3, "score": 0.80, "text": "c" * 30, "llm_token_count": 15},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 25)
        # First passage fits (15 <= 25), second would exceed (15+15=30 > 25)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["passage_id"], 1)

    def test_budget_first_passage_exceeds(self):
        """If the first passage alone exceeds budget, include it anyway."""
        candidates = [
            {"passage_id": 1, "score": 0.90, "text": "a" * 300, "llm_token_count": 100},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 25)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["passage_id"], 1)

    def test_empty_candidates(self):
        self.assertEqual(_apply_threshold_and_budget([], 0.70, 100), [])

    def test_approximation_when_token_count_missing(self):
        """When llm_token_count is 0, use len(text)//3 as approximation."""
        candidates = [
            {"passage_id": 1, "score": 0.90, "text": "a" * 30, "llm_token_count": 0},
        ]
        result = _apply_threshold_and_budget(candidates, 0.70, 5)
        # 30 chars // 3 = 10 tokens, which exceeds budget 5.
        # But first-passage-alone-exceeds rule keeps it.
        self.assertEqual(len(result), 1)


class TestMergeCandidatesRaw(unittest.TestCase):
    def test_merge_no_overlap(self):
        per_sub = [
            [
                {"passage_id": 1, "score": 0.90, "text": "a", "llm_token_count": 1, "_sub_idx": 0},
            ],
            [
                {"passage_id": 2, "score": 0.85, "text": "b", "llm_token_count": 1, "_sub_idx": 1},
            ],
        ]
        result = _merge_candidates_raw(per_sub, global_cap=50)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["passage_id"] for r in result], [1, 2])

    def test_dedupe_keeps_highest_score(self):
        """Same passage from two subs: keep highest score, track both sub indices."""
        per_sub = [
            [
                {"passage_id": 1, "score": 0.80, "text": "a", "llm_token_count": 1, "_sub_idx": 0},
            ],
            [
                {"passage_id": 1, "score": 0.90, "text": "a", "llm_token_count": 1, "_sub_idx": 1},
            ],
        ]
        result = _merge_candidates_raw(per_sub, global_cap=50)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 0.90)
        self.assertEqual(sorted(result[0]["_sub_indices"]), [0, 1])

    def test_global_cap(self):
        per_sub = [
            [{"passage_id": i, "score": 0.90 - i * 0.01, "text": "x", "llm_token_count": 1, "_sub_idx": 0}]
            for i in range(1, 10)
        ]
        result = _merge_candidates_raw(per_sub, global_cap=5)
        self.assertEqual(len(result), 5)
        # Should be sorted by score descending.
        self.assertEqual([r["passage_id"] for r in result], [1, 2, 3, 4, 5])

    def test_skip_zero_pid(self):
        """Candidates with passage_id <= 0 should be skipped."""
        per_sub = [
            [
                {"passage_id": 0, "score": 0.90, "text": "a", "llm_token_count": 1, "_sub_idx": 0},
                {"passage_id": -1, "score": 0.85, "text": "b", "llm_token_count": 1, "_sub_idx": 0},
                {"passage_id": 1, "score": 0.80, "text": "c", "llm_token_count": 1, "_sub_idx": 0},
            ],
        ]
        result = _merge_candidates_raw(per_sub, global_cap=50)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["passage_id"], 1)


class TestHitToPassage(unittest.TestCase):
    def test_unified_schema(self):
        """Test mapping from Qdrant hit with unified schema fields."""
        hit = SimpleNamespace(
            id="abc123",
            score=0.85,
            payload={
                "passage_id": 42,
                "text": "sample text",
                "category": "cat",
                "sub_category": "sub",
                "service": "svc",
                "topic": "topic",
                "chunk_type": "wiki",
                "llm_token_count": 15,
            },
        )
        result = _hit_to_passage(hit)
        self.assertEqual(result["passage_id"], 42)
        self.assertEqual(result["text"], "sample text")
        self.assertEqual(result["category"], "cat")
        self.assertEqual(result["sub_category"], "sub")
        self.assertEqual(result["service"], "svc")
        self.assertEqual(result["topic"], "topic")
        self.assertEqual(result["chunk_type"], "wiki")
        self.assertEqual(result["llm_token_count"], 15)
        self.assertEqual(result["score"], 0.85)

    def test_legacy_schema_fallback(self):
        """Test mapping from Qdrant hit with legacy schema fields."""
        hit = SimpleNamespace(
            id="abc123",
            score=0.75,
            payload={
                "text": "legacy text",
                "page_title": "Page Title",
                "section": "Section",
                "subsection": "Subsection",
            },
        )
        result = _hit_to_passage(hit)
        self.assertEqual(result["category"], "Page Title")
        self.assertEqual(result["sub_category"], "Section")
        self.assertEqual(result["service"], "Subsection")
        self.assertEqual(result["topic"], "Page Title")

    def test_missing_passage_id_uses_hit_id(self):
        hit = SimpleNamespace(
            id="999",
            score=0.60,
            payload={"text": "no pid"},
        )
        result = _hit_to_passage(hit)
        self.assertEqual(result["passage_id"], 999)

    def test_uuid_hit_id(self):
        hit = SimpleNamespace(
            id="550e8400-e29b-41d4-a716-446655440000",
            score=0.60,
            payload={"text": "uuid"},
        )
        result = _hit_to_passage(hit)
        # Last 8 hex chars of UUID stripped → int
        hex_part = "550e8400e29b41d4a716446655440000"[-8:]
        self.assertEqual(result["passage_id"], int(hex_part, 16))


if __name__ == "__main__":
    unittest.main()
