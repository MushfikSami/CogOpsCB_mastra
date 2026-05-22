#!/usr/bin/env python3
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from cogops.pipeline.query_expand import (
    expand_sub_query,
    expand_sub_query_llm,
    expand_sub_queries_llm,
    _EXPANDER_CACHE,
)


def _mock_secondary_llm(response_json: str):
    async def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_json))],
        )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


class TestLlmExpander(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Clear cache before each test.
        _EXPANDER_CACHE._store.clear()

    async def test_hardcoded_fast_path_skips_llm(self):
        """When the hardcoded map already recognizes the query, no LLM call."""
        secondary_create = AsyncMock(
            side_effect=AssertionError("LLM must not be called when hardcoded hits"),
        )
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=secondary_create)),
        )
        query = "বিয়ের সার্টিফিকেটে নাম পরিবর্তন"
        result = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertIn("বিবাহ সনদ", result)
        secondary_create.assert_not_awaited()

    async def test_llm_fallback_adds_synonyms(self):
        """When hardcoded misses, the LLM generates formal synonyms."""
        llm_json = '{"detected":true,"terms":["হজ সনদ","হজ ও ওমরাহ সনদ"]}'
        secondary = _mock_secondary_llm(llm_json)
        query = "হজ সনদের জন্য কী কী কাগজপত্র লাগে?"
        result = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertIn("হজ সনদ", result)
        self.assertIn("হজ ও ওমরাহ সনদ", result)
        self.assertTrue(result.startswith(query))

    async def test_llm_no_detection_returns_query_unchanged(self):
        """When the LLM says no document type detected, query is unchanged."""
        llm_json = '{"detected":false,"terms":[]}'
        secondary = _mock_secondary_llm(llm_json)
        query = "কর রেয়াত কীভাবে পাব?"
        result = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertEqual(result, query)

    async def test_cache_prevents_duplicate_llm_calls(self):
        """The same query should hit cache on second call."""
        call_count = 0
        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"detected":true,"terms":["টেস্ট"]}'))],
            )
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=counting_create)),
        )
        query = "অজানা সনদের জন্য আবেদন"
        r1 = await expand_sub_query_llm(query, secondary, "qwen36")
        r2 = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertEqual(r1, r2)
        self.assertEqual(call_count, 1)

    async def test_llm_error_fails_open(self):
        """On LLM timeout/failure, return query unchanged."""
        async def broken_create(**kwargs):
            raise RuntimeError("boom")
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=broken_create)),
        )
        query = "হজ সনদ"
        result = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertEqual(result, query)

    async def test_batch_expansion(self):
        """expand_sub_queries_llm processes each query independently."""
        call_log = []
        async def logging_create(**kwargs):
            content = kwargs.get("messages", [])[1].get("content", "")
            call_log.append(content)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"detected":false,"terms":[]}'))],
            )
        secondary = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=logging_create)),
        )
        queries = ["এনআইডি হারিয়ে গেলে", "হজ সনদের কাগজপত্র"]
        results = await expand_sub_queries_llm(queries, secondary, "qwen36")
        self.assertEqual(len(results), 2)
        # First query hits hardcoded fast-path (NID), so no LLM call.
        # Second query misses hardcoded, so LLM is called.
        self.assertEqual(len(call_log), 1)
        self.assertIn("হজ সনদের কাগজপত্র", call_log[0])


if __name__ == "__main__":
    unittest.main()
