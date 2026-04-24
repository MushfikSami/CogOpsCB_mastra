"""
cogops/tools/history/query.py

history_query tool: lookup/summarize/recent/ask modes.
"""

import json
import logging
import asyncio
from typing import Optional

from cogops.session.redis_store import RedisSessionStore

logger = logging.getLogger(__name__)


def history_query_lookup(turns: list[dict], query: str) -> str:
    """
    Regex/substring search across stored turns.
    Returns matching Q/A pairs with turn ids.
    """
    if not turns:
        return "No conversation history found."

    query_lower = query.lower()
    matches = []
    for turn in turns:
        user_text = str(turn.get("user", "")).lower()
        assistant_text = str(turn.get("assistant", "")).lower()
        if query_lower in user_text or query_lower in assistant_text:
            matches.append(f"Turn {turn.get('turn_id', '?')}:\n  User: {turn.get('user', '')}\n  AI: {turn.get('assistant', '')}")

    if matches:
        return "\n\n".join(matches)
    return f"No matches found for '{query}' in conversation history."


def history_query_summarize(rolling_summary: str) -> str:
    """Return the rolling summary verbatim."""
    if not rolling_summary:
        return "No conversation summary available yet."
    return f"Recent conversation summary:\n{rolling_summary}"


def history_query_recent(turns: list[dict], n: int = 3) -> str:
    """Return the last N turns verbatim."""
    recent = turns[:n]
    if not recent:
        return "No conversation history yet."
    parts = []
    for t in recent:
        parts.append(f"Turn {t.get('turn_id', '?')}:\n  User: {t.get('user', '')}\n  AI: {t.get('assistant', '')}")
    return "\n\n".join(parts)


async def history_query_ask(
    question: str,
    turns: list[dict],
    secondary_client,
    secondary_model: str,
) -> str:
    """
    Pass the question + raw turns to secondary LLM.
    Used for: 'what did the user mean earlier by X?'
    """
    if not secondary_client:
        return "History query_ask requires a secondary LLM."

    turns_text = "\n---\n".join(
        f"Turn {t.get('turn_id', '?')}: User: {t.get('user', '')} | AI: {t.get('assistant', '')}"
        for t in turns[:10]
    )

    try:
        response = await secondary_client.chat.completions.create(
            model=secondary_model,
            messages=[{
                "role": "user",
                "content": f"Using only this conversation history, answer the question below.\n\nHistory:\n{turns_text}\n\nQuestion: {question}"
            }],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content or "No answer found."
    except Exception as e:
        logger.error(f"history_query_ask failed: {e}")
        return f"Error querying history: {e}"


async def history_query(
    mode: str,
    query: Optional[str] = None,
    n: int = 3,
    user_id: Optional[str] = None,
    store: Optional[RedisSessionStore] = None,
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """
    Main entry point for the history_query tool.

    Args:
        mode: 'lookup' | 'summarize' | 'recent' | 'ask'
        query: search term (for 'lookup' and 'ask' modes)
        n: number of recent turns to return (for 'recent' mode)
        user_id: session user_id (required for all modes)
        store: RedisSessionStore instance
        secondary_client: AsyncOpenAI for the secondary LLM (required for 'ask' mode)
        secondary_model: model name for secondary LLM
    """
    if mode not in ("lookup", "summarize", "recent", "ask"):
        return f"Invalid mode '{mode}'. Use: lookup, summarize, recent, ask."

    if not user_id:
        return "Missing user_id."

    if store is None or not store.available:
        return "History store not available."

    turns = store.get_recent_turns(user_id, n=20)

    if mode == "lookup":
        return history_query_lookup(turns, query or "")

    if mode == "summarize":
        summary = store.get_summary(user_id)
        return history_query_summarize(summary)

    if mode == "recent":
        return history_query_recent(turns, n)

    if mode == "ask":
        return await history_query_ask(
            question=query or "What was discussed?",
            turns=turns,
            secondary_client=secondary_client,
            secondary_model=secondary_model,
        )

    return "Unknown mode."


# --- Tool Schema ---

history_query_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "history_query",
            "description": (
                "Query conversation history. Call this BEFORE any information tool "
                "if the user's message is short, numeric, or refers to a previous "
                "list (e.g. '3', 'the second one', 'tell me more', 'what about that'). "
                "Also call it when the user explicitly asks about earlier turns. "
                "Modes: 'recent' = last N turns verbatim (best for resolving "
                "ambiguous follow-ups), 'lookup' = substring search, "
                "'summarize' = rolling summary, 'ask' = the secondary LLM answers "
                "a question using the full history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["lookup", "summarize", "recent", "ask"],
                        "description": "Query mode."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search term (for 'lookup') or question (for 'ask')."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent turns (for 'recent' mode, default 3)."
                    }
                },
                "required": ["mode"]
            }
        }
    }
]

history_query_tools_map = {
    "history_query": history_query
}
