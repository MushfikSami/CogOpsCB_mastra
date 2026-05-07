"""
cogops/tools/search_history.py

history_query tool: lookup/recent/ask/summarize modes.
Rewritten with lean schema and cleaner parameter names.
"""

import json
import logging
import asyncio
from typing import Optional, Any

logger = logging.getLogger(__name__)


def _get_config():
    from cogops.config.loader import load_config
    try:
        return load_config()
    except Exception:
        return {}


def _ask_turns_limit() -> int:
    return _get_config().get("history_query", {}).get("ask_turns_limit", 10)


def _default_max_turns() -> int:
    return _get_config().get("history_query", {}).get("default_max_turns", 20)


def history_query_lookup(turns: list, query: str) -> str:
    """Substring search across stored turns. Returns matching Q/A pairs."""
    if not turns:
        return "No conversation history found."

    query_lower = query.lower()
    matches = []
    for turn in turns:
        user_text = str(turn.get("user", "")).lower()
        assistant_text = str(turn.get("assistant", "")).lower()
        if query_lower in user_text or query_lower in assistant_text:
            matches.append(
                f"Turn {turn.get('turn_id', '?')}:\n"
                f"  User: {turn.get('user', '')}\n"
                f"  AI: {turn.get('assistant', '')}"
            )

    return "\n\n".join(matches) if matches else f"No matches found for '{query}' in conversation history."


def history_query_summarize(rolling_summary: str) -> str:
    """Return the rolling summary."""
    if not rolling_summary:
        return "No conversation summary available yet."
    return f"Recent conversation summary:\n{rolling_summary}"


def history_query_recent(turns: list, n: int = 3) -> str:
    """Return the last N turns verbatim."""
    recent = turns[:n]
    if not recent:
        return "No conversation history yet."
    parts = []
    for t in recent:
        parts.append(
            f"Turn {t.get('turn_id', '?')}:\n"
            f"  User: {t.get('user', '')}\n"
            f"  AI: {t.get('assistant', '')}"
        )
    return "\n\n".join(parts)


async def history_query_ask(
    question: str,
    turns: list,
    secondary_client,
    secondary_model: str,
) -> str:
    """Pass question + raw turns to secondary LLM for meta-query."""
    if not secondary_client:
        return "History query_ask requires a secondary LLM."

    turns_text = "\n---\n".join(
        f"Turn {t.get('turn_id', '?')}: User: {t.get('user', '')} | AI: {t.get('assistant', '')}"
        for t in turns[:_ask_turns_limit()]
    )

    try:
        response = await secondary_client.chat.completions.create(
            model=secondary_model,
            messages=[{
                "role": "user",
                "content": (
                    f"Using only this conversation history, answer the question below.\n\n"
                    f"History:\n{turns_text}\n\nQuestion: {question}"
                ),
            }],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content or "No answer found."
    except Exception as e:
        logger.error("history_query_ask failed: %s", e)
        return f"Error querying history: {e}"


async def history_query(
    mode: str,
    query: Optional[str] = None,
    n: int = 3,
    user_id: Optional[str] = None,
    store: Optional[Any] = None,
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """
    Main entry point for the history_query tool.

    Args:
        mode: 'lookup' | 'recent' | 'ask' | 'summarize'
        query: search term (for 'lookup' and 'ask' modes)
        n: number of recent turns (for 'recent' mode)
        user_id: session user_id (required for all modes)
        store: RedisSessionStore instance
        secondary_client: AsyncOpenAI for secondary LLM (required for 'ask')
        secondary_model: model name for secondary LLM
    """
    from cogops.session.redis_store import RedisSessionStore

    if mode not in ("lookup", "recent", "ask", "summarize"):
        return f"Invalid mode '{mode}'. Use: lookup, recent, ask, summarize."

    if not user_id:
        return "Missing user_id."

    if store is None or not store.available:
        return "History store not available."

    turns = store.get_recent_turns(user_id, n=_default_max_turns())

    if mode == "summarize":
        summary = store.get_summary(user_id)
        return history_query_summarize(summary)

    if mode == "lookup":
        return history_query_lookup(turns, query or "")

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


# --- Lean Tool Schema ---
history_query_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "history_query",
            "description": (
                "Query the raw conversation history stored in Redis. "
                "Use when the user asks a short/numeric follow-up, references a prior turn, "
                "or when intent is ambiguous. The system only has a condensed rolling summary — "
                "use this tool for exact past context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["lookup", "recent", "ask", "summarize"],
                        "description": (
                            "'recent' — last N turns verbatim (resolve immediate follow-ups). "
                            "'lookup' — substring search across all past turns. "
                            "'ask' — pass history to secondary LLM to answer a meta-question. "
                            "'summarize' — return current rolling summary."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Search term for 'lookup' mode, or meta-question for 'ask' mode.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent turns to return (default 3).",
                    },
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
    }
]

history_query_tools_map = {
    "history_query": history_query,
}
