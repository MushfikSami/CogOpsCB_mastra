#!/usr/bin/env python3
"""
Minimal tests for the query_expand backward-compat stubs.

The pipeline no longer calls the query expander (formalization is handled
by the router + Jiggasha instruction generator).  These tests ensure the
stubs remain functional for any unexpected callers.
"""
import asyncio
import unittest
from types import SimpleNamespace

from cogops.pipeline.query_expand import (
    expand_sub_query_llm,
    expand_sub_queries_llm,
)


def _mock_secondary_llm(response_text: str):
    async def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))],
        )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


class TestLlmExpanderStubs(unittest.IsolatedAsyncioTestCase):
    async def test_stub_returns_query_unchanged(self):
        """The backward-compat stub returns the query unchanged."""
        secondary = _mock_secondary_llm("should not matter")
        query = "অজানা সনদের জন্য আবেদন"
        result = await expand_sub_query_llm(query, secondary, "qwen36")
        self.assertEqual(result, query)

    async def test_batch_stub_returns_queries_unchanged(self):
        """The batch stub returns all queries unchanged."""
        secondary = _mock_secondary_llm("should not matter")
        queries = ["এনআইডি হারিয়ে গেলে", "হজ সনদের কাগজপত্র"]
        results = await expand_sub_queries_llm(queries, secondary, "qwen36")
        self.assertEqual(results, queries)


if __name__ == "__main__":
    unittest.main()
