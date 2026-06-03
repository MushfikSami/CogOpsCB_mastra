"""
jiggasha/tests/test_instruction.py

Unit tests for dynamic instruction generation.
"""

from __future__ import annotations

import asyncio
import sys
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from instruction import generate_instruction, clear_instruction_cache, _build_query_block


def _mock_client(content: str):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=content)
                            )
                        ]
                    )
                )
            )
        )
    )


class TestBuildQueryBlock(unittest.TestCase):
    def test_single_query(self):
        self.assertEqual(_build_query_block("পাসপোর্ট ফি কত?"), "Query: পাসপোর্ট ফি কত?")

    def test_multi_query(self):
        block = _build_query_block("পাসপোর্ট ফি কত?\nএনআইডি সংশোধন কোথায়?")
        self.assertIn("Queries:", block)
        self.assertIn("1. পাসপোর্ট ফি কত?", block)
        self.assertIn("2. এনআইডি সংশোধন কোথায়?", block)

    def test_strips_empty_lines(self):
        block = _build_query_block("  \n  পাসপোর্ট ফি কত?  \n  \n")
        self.assertEqual(block, "Query: পাসপোর্ট ফি কত?")


class TestGenerateInstruction(unittest.TestCase):
    def setUp(self):
        clear_instruction_cache()

    def test_empty_query_returns_none(self):
        result = asyncio.run(generate_instruction(
            query="", secondary_client=_mock_client("x"), secondary_model="m"
        ))
        self.assertIsNone(result)

    def test_no_client_returns_none(self):
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?", secondary_client=None, secondary_model="m"
        ))
        self.assertIsNone(result)

    def test_successful_generation(self):
        client = _mock_client("Retrieve passages about passport fees.")
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertEqual(result, "Retrieve passages about passport fees.")
        client.chat.completions.create.assert_called_once()

    def test_cache_hit_skips_llm(self):
        client = _mock_client("Retrieve passages about passport fees.")
        # First call populates cache.
        r1 = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertEqual(r1, "Retrieve passages about passport fees.")
        # Second call with same query should hit cache.
        client.chat.completions.create.reset_mock()
        r2 = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertEqual(r2, "Retrieve passages about passport fees.")
        client.chat.completions.create.assert_not_called()

    def test_timeout_returns_none(self):
        async def _hang(*_, **__):
            await asyncio.sleep(30)

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_hang)
            )
        )
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
            timeout=0.05,
        ))
        self.assertIsNone(result)

    def test_llm_exception_returns_none(self):
        async def _boom(*_, **__):
            raise RuntimeError("synthetic")

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_boom)
            )
        )
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertIsNone(result)

    def test_empty_llm_response_returns_none(self):
        client = _mock_client("   ")
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertIsNone(result)

    def test_multi_query_in_prompt(self):
        """When multiple sub-queries are passed, the prompt should contain all of them."""
        client = _mock_client("Retrieve passages about passport and NID.")
        result = asyncio.run(generate_instruction(
            query="পাসপোর্ট ফি কত?\nএনআইডি সংশোধন কোথায়?",
            secondary_client=client,
            secondary_model="m",
        ))
        self.assertEqual(result, "Retrieve passages about passport and NID.")
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        self.assertIn("পাসপোর্ট ফি কত?", user_msg)
        self.assertIn("এনআইডি সংশোধন কোথায়?", user_msg)


if __name__ == "__main__":
    unittest.main()
