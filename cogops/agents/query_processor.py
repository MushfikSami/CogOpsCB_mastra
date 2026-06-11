"""
cogops/agents/query_processor.py

Layer 2 — Query Processing Agents.

Three sequential agents:
  2a. QueryDisambiguator — resolve pronouns/references using history
  2b. QueryFormalizer   — casual spoken Bengali → formal document language
  2c. QueryFanOut       — split multi-question, cap at max_concurrent_query

The pipeline uses QueryProcessor.process() which runs all three in order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from cogops.prompts.time_reminder import build_time_reminder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 2a. QueryDisambiguator
# ------------------------------------------------------------------

_DISAMBIG_SYSTEM_PROMPT = """\
You are the QueryDisambiguator for a Bangladesh government-services chatbot.

Given a conversation history and the CURRENT user message, resolve any pronouns
or ambiguous references so the query becomes a STANDALONE question that needs
no context to understand.

Rules:
1. Replace pronouns (এটি, সেটা, তার, এটা, ওইটা, তা) with the actual noun from history.
2. If the user refers to "the previous thing" or "that", make it explicit.
3. Output ONLY the disambiguated query string. No explanation, no JSON.
4. If the query is already standalone, return it unchanged.
5. Keep the language in Bengali."""


class QueryDisambiguator:
    """Resolve pronouns and ambiguous references using conversation history."""

    def __init__(
        self,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
        timeout: float = 4.0,
    ):
        self.client = secondary_client
        self.model = secondary_model
        self.timeout = timeout

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for msg in history[-6:]:  # last 6 messages for context
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def disambiguate(
        self,
        query: str,
        history: List[Dict[str, str]],
    ) -> str:
        """Return a standalone query string. On failure, returns query unchanged."""
        if not history or not self.client:
            return query

        history_block = self._format_history(history)
        user_msg = (
            f"Conversation history:\n{history_block}\n\n"
            f"Current user message: {query}\n\n"
            "Rewrite the current message as a standalone query (no pronouns, no ambiguous references):"
        )

        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _DISAMBIG_SYSTEM_PROMPT},
                        {"role": "assistant", "content": build_time_reminder()},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                ),
                timeout=self.timeout,
            )
            result = (resp.choices[0].message.content or "").strip()
            # Strip quotes if the LLM wrapped the output
            result = result.strip('"').strip("'").strip()
            return result if result else query
        except Exception as e:  # noqa: BLE001
            logger.warning("QueryDisambiguator failed (%s); returning original.", e)
            return query


# ------------------------------------------------------------------
# 2b. QueryFormalizer
# ------------------------------------------------------------------

_FORMALIZER_SYSTEM_PROMPT = '''\
You are the QueryFormalizer for a Bangladesh government-services chatbot.

Convert the user's casual/spoken Bengali query into a FORMAL, search-optimized
Bengali query suitable for embedding retrieval.

Rules:
1. Replace colloquial or English loanwords with standard Bengali government terms:
   • প্লেন → বিমান
   • ট্রেন → রেল
   • টিকেট → টিকিট
   • নিড → এনআইডি
   • ডাক্তার → উপাধি/পদবি (only when context is about adding titles to documents)
2. REMOVE conversational fillers: আচ্ছা, ভাই, দেখেন, শুনুন, বলুন তো, একটু, কিন্তু, তাই, তাহলে.
3. Write CONCISE formal queries. Avoid long story-like framing.
4. If the user mentions a specific document type (passport, NID, marriage certificate, birth certificate), you MUST include that exact document type.
5. Output ONLY the formalized query string. No explanation, no JSON, no markdown.

Examples:
Casual: "নিডে ডাক্তার লাগাইতে পারব?"
Formal: "জাতীয় পরিচয়পত্রে বিশেষ উপাধি যোগ করার বিধান ও প্রক্রিয়া"

Casual: "পাসপোর্ট করতে কি কি লাগবে?"
Formal: "পাসপোর্ট আবেদনের জন্য প্রয়োজনীয় নথিপত্র ও যোগ্যতা"

Casual: "বিয়ে সার্টিফিকেটে নাম ভুল"
Formal: "বিবাহ সনদে নাম সংশোধনের আবেদন প্রক্রিয়া ও নিয়মাবলি"
'''


class QueryFormalizer:
    """Convert casual spoken Bengali → formal document-search language."""

    def __init__(
        self,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
        timeout: float = 4.0,
    ):
        self.client = secondary_client
        self.model = secondary_model
        self.timeout = timeout

    async def formalize(self, query: str) -> str:
        """Return a formalized query string. On failure, returns query unchanged."""
        if not self.client:
            return query

        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _FORMALIZER_SYSTEM_PROMPT},
                        {"role": "assistant", "content": build_time_reminder()},
                        {"role": "user", "content": f"Casual query: {query}\n\nFormalized query:"},
                    ],
                    temperature=0.1,
                    max_tokens=256,
                ),
                timeout=self.timeout,
            )
            result = (resp.choices[0].message.content or "").strip()
            result = result.strip('"').strip("'").strip()
            return result if result else query
        except Exception as e:  # noqa: BLE001
            logger.warning("QueryFormalizer failed (%s); returning original.", e)
            return query


# ------------------------------------------------------------------
# 2c. QueryFanOut
# ------------------------------------------------------------------

_CONNECTORS_RE = re.compile(
    r"(?:\bএবং\b|\bআর\b|\bও\b|\band\b|\balso\b|;|,\s+and\b)",
    re.IGNORECASE,
)

# Synonym mappings: informal / loanword → formal Bengali
_SYNONYM_ROOTS = [
    ("প্লেন", "বিমান"),
    ("ট্রেন", "রেল"),
    ("টিকেট", "টিকিট"),
]

_BN_SUFFIXES = ["ের", "ে", "ি", "া", "ো", "ী", "ু", "ূ", "ৃ", "ং", "ঃ", "়", "ঁ"]
_BN_CHAR_CLASS = r"[\u0980-\u09FF]"

_SYNONYM_RES: List[Tuple[re.Pattern[str], str]] = []
for informal, formal in _SYNONYM_ROOTS:
    for suffix in [""] + _BN_SUFFIXES:
        old_form = informal + suffix
        new_form = formal + suffix
        pattern = (
            r"(?<!" + _BN_CHAR_CLASS + ")"
            + re.escape(old_form)
            + r"(?!" + _BN_CHAR_CLASS + ")"
        )
        _SYNONYM_RES.append((re.compile(pattern), new_form))

_FILLER_RE = re.compile(
    r"(?<![\u0980-\u09FF])"
    r"(আচ্ছা|ভাই|দেখেন|শুনুন|বলুন\s+তো|বলো\s+তো|জানাবেন|জানাব|একটু|কিন্তু|তাই|তাহলে|তো)"
    r"(?![\u0980-\u09FF])",
    re.IGNORECASE,
)

_WS_COLLAPSE_RE = re.compile(r"[ \t]+")
_NL_COLLAPSE_RE = re.compile(r"\n{3,}")


class QueryFanOut:
    """Split multi-question queries, normalize, cap at max_concurrent_query."""

    def __init__(self, max_concurrent_query: int = 3):
        self.max_concurrent_query = max_concurrent_query

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text).strip()
        for regex, replacement in _SYNONYM_RES:
            text = regex.sub(replacement, text)
        text = _FILLER_RE.sub("", text)
        text = _WS_COLLAPSE_RE.sub(" ", text)
        text = _NL_COLLAPSE_RE.sub("\n\n", text)
        text = text.strip(r".,!;:\-")
        text = text.strip()
        return text

    def fan_out(
        self,
        queries: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Normalize and cap queries. Returns (accepted, overflow).

        If the input already contains multiple queries (from IntentClassifier),
        we just normalize them. If it's a single query with conjunctions, we
        could split it — but the IntentClassifier already handles multi-question
        splitting. This agent is mostly a safety net for normalization + capping.
        """
        normalized: List[str] = []
        for q in queries:
            cleaned = self._normalize(q)
            if cleaned:
                normalized.append(cleaned)

        if not normalized:
            # Fallback: return original queries if all became empty
            normalized = [q.strip() for q in queries if q.strip()]

        accepted = normalized[: self.max_concurrent_query]
        overflow = normalized[self.max_concurrent_query :]
        return accepted, overflow


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

@dataclass
class ProcessedQuery:
    queries: List[str]
    overflow: List[str]
    disambiguated: str
    formalized: str


class QueryProcessor:
    """Runs disambiguation → formalization → fan-out in sequence."""

    def __init__(
        self,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
        max_concurrent_query: int = 3,
    ):
        self.disambiguator = QueryDisambiguator(secondary_client, secondary_model)
        self.formalizer = QueryFormalizer(secondary_client, secondary_model)
        self.fanout = QueryFanOut(max_concurrent_query)

    async def process(
        self,
        raw_query: str,
        sub_queries: List[str],
        history: List[Dict[str, str]],
    ) -> ProcessedQuery:
        """Run the full query-processing pipeline.

        1. Disambiguate the main query using history.
        2. Formalize the disambiguated query.
        3. If IntentClassifier provided sub_queries, use those (they're already
           from the LLM which did history-aware splitting). Otherwise formalize
           the main query only.
        4. Fan-out: normalize + cap.
        """
        # Step 1: Disambiguate main query
        disambiguated = await self.disambiguator.disambiguate(raw_query, history)

        # Step 2: Formalize
        if sub_queries and len(sub_queries) > 1:
            # Multi-question: formalize each sub-query
            formalized_queries = []
            for sq in sub_queries:
                formal = await self.formalizer.formalize(sq)
                formalized_queries.append(formal)
            formalized = formalized_queries[0] if formalized_queries else disambiguated
        else:
            formalized = await self.formalizer.formalize(disambiguated)
            formalized_queries = [formalized]

        # Step 3: Fan-out (normalize + cap)
        accepted, overflow = self.fanout.fan_out(formalized_queries)

        return ProcessedQuery(
            queries=accepted,
            overflow=overflow,
            disambiguated=disambiguated,
            formalized=formalized,
        )
