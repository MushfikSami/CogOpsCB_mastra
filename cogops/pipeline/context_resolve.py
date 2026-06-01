"""
cogops/pipeline/context_resolve.py

Context-aware reference resolution for Bengali follow-up queries.

When a user's follow-up contains pronouns like "তার", "তিনি", "তা", "সে",
the router and retrieval pipeline need the pronoun resolved to the actual
entity before embedding search happens.  Otherwise Jiggasha retrieves
passages for the pronoun (which matches nothing) instead of the real subject.

This module provides a lightweight secondary-LLM call that:
  1. Detects if the query contains a resolvable pronoun,
  2. Reads the last assistant turn from history,
  3. Asks the secondary LLM to rewrite the query with the pronoun replaced.

It is fail-soft: on any error or timeout the original query is returned
unchanged so latency or model instability never blocks a user turn.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Bengali pronouns / demonstratives that commonly refer to a previously
# mentioned person or entity in follow-up questions.
_RESOLVABLE_PRONOUNS_RE = re.compile(
    r"\b(তার|তিনি|তাহার|তাহা|সে|ওঁর|ওর|ওনার|ওই|ওহি|উনি|উনার|এর|ইহার)\b"
)

# Maximum length of the assistant excerpt we feed the resolver; keep it
# short so the prompt stays small and fast.
_MAX_ASSISTANT_EXCERPT = 600

_RESOLVER_SYSTEM_PROMPT = """\
You resolve pronouns in Bengali conversations.

You will receive:
- a PREVIOUS assistant answer (short excerpt)
- the user's FOLLOW-UP question

If the follow-up contains pronouns (তার, তিনি, সে, ওঁর, ওর, ওনার, ওই, ওহি, উনি, উনার, এর, ইহার, তাহার, তাহা) that refer to a person or entity mentioned in the previous answer, replace the pronoun with that person's or entity's actual name.

Rules:
- Output ONLY the rewritten question. No explanation, no markdown.
- If there is no pronoun OR the pronoun does not refer to anyone in the previous answer, output the follow-up EXACTLY as given.
- Preserve all other words, punctuation, and spelling.
"""


async def resolve_references(
    query: str,
    history: List[Dict[str, str]],
    secondary_client: Optional[AsyncOpenAI],
    secondary_model: str,
    timeout: float = 3.0,
) -> str:
    """Return a query with pronouns resolved using conversation history.

    Args:
        query: the sanitized user query.
        history: recent user/assistant turns (oldest → newest).
        secondary_client: AsyncOpenAI client for the lightweight resolver call.
        secondary_model: model name (e.g. gemma4).
        timeout: hard timeout in seconds for the LLM call.

    Returns:
        The resolved query, or the original query if no pronouns are found,
        no history exists, or the resolver fails.
    """
    if not query or not secondary_client:
        return query

    # Fast-path: no resolvable pronouns → skip LLM call entirely.
    if not _RESOLVABLE_PRONOUNS_RE.search(query):
        return query

    # Find the most recent assistant turn.
    last_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = (msg.get("content") or "").strip()
            break
    if not last_assistant:
        return query

    # Strip citation tags and sources block so the resolver isn't distracted.
    clean_assistant = re.sub(r"\[S\d+\]", "", last_assistant)
    clean_assistant = re.sub(r"\n+---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b.*", "", clean_assistant, flags=re.DOTALL | re.IGNORECASE)
    clean_assistant = clean_assistant.strip()
    if not clean_assistant:
        return query

    excerpt = clean_assistant[:_MAX_ASSISTANT_EXCERPT]

    prompt = (
        f"PREVIOUS answer excerpt:\n{excerpt}\n\n"
        f"FOLLOW-UP question:\n{query}\n\n"
        "Rewrite the follow-up with pronouns resolved. Output ONLY the rewritten question."
    )

    try:
        resp = await asyncio.wait_for(
            secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[
                    {"role": "system", "content": _RESOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            ),
            timeout=timeout,
        )
        resolved = (resp.choices[0].message.content or "").strip()
        if resolved and resolved != query:
            logger.info(
                "context_resolve: %r → %r (excerpt_len=%d)",
                query, resolved, len(excerpt),
            )
            return resolved
    except asyncio.TimeoutError:
        logger.debug("context_resolve timeout; using original query")
    except Exception as e:  # noqa: BLE001
        logger.debug("context_resolve failed: %s; using original query", e)

    return query
