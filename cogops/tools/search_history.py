"""
cogops/tools/history/query.py

history_query tool: lookup/summarize/recent/ask modes.
"""

import json
import logging
import asyncio
from typing import Optional

from cogops.config.loader import load_config
from cogops.session.redis_store import RedisSessionStore

logger = logging.getLogger(__name__)

# Cached config
_hq_config: Optional[dict] = None


def _get_hq_config() -> dict:
    global _hq_config
    if _hq_config is None:
        _hq_config = load_config()
    return _hq_config


def _ask_turns_limit() -> int:
    return _get_hq_config().get("history_query", {}).get("ask_turns_limit", 10)


def _default_max_turns() -> int:
    return _get_hq_config().get("history_query", {}).get("default_max_turns", 20)


def history_query_lookup(turns: list[dict], query: str) -> str:
    """
    Regex/substring search across stored turns.
    Returns matching Q/A pairs with turn ids.
    """
    if not turns:
        return "No conversation history found."

    query_lower = query.lower()
    matches =[]
    for turn in turns:
        user_text = str(turn.get("user", "")).lower()
        assistant_text = str(turn.get("assistant", "")).lower()
        if query_lower in user_text or query_lower in assistant_text:
            matches.append(f"Turn {turn.get('turn_id', '?')}:\n  User: {turn.get('user', '')}\n  AI: {turn.get('assistant', '')}")

    if matches:
        return "\n\n".join(matches)
    return f"No matches found for '{query}' in conversation history."




def history_query_recent(turns: list[dict], n: int = 3) -> str:
    """Return the last N turns verbatim."""
    recent = turns[:n]
    if not recent:
        return "No conversation history yet."
    parts =[]
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
        for t in turns[:_ask_turns_limit()]
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
        mode: 'lookup' | 'recent' | 'ask'
        query: search term (for 'lookup' and 'ask' modes)
        n: number of recent turns to return (for 'recent' mode)
        user_id: session user_id (required for all modes)
        store: RedisSessionStore instance
        secondary_client: AsyncOpenAI for the secondary LLM (required for 'ask' mode)
        secondary_model: model name for secondary LLM
    """
    if mode not in ("lookup",  "recent", "ask"):
        return f"Invalid mode '{mode}'. Use: lookup,recent, ask."

    if not user_id:
        return "Missing user_id."

    if store is None or not store.available:
        return "History store not available."

    turns = store.get_recent_turns(user_id, n=_default_max_turns())

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


# --- Tool Schema ---

history_query_tools_list =[
    {
        "type": "function",
        "function": {
            "name": "history_query",
            "description": (
                "Query the exact, raw conversation history.\n\n"
                "CRITICAL: Your system prompt only contains a highly condensed rolling summary of past turns. "
                "The summary routinely drops exact quotes, lists, specific options, and nuanced details. "
                "If you need exact past context, or if the user's current message is ambiguous, short, numeric, "
                "or refers to a previous turn (e.g., '3', 'the second one', 'what about that?', 'tell me more', "
                "'how much was the fee again?'), you MUST call this tool FIRST to understand the context before "
                "answering or using external search tools.\n\n"
                "Modes:\n"
                "- 'recent': Returns the last N turns verbatim. Best for resolving immediate ambiguous follow-ups "
                "(e.g., user says 'Yes', 'Option 2', or 'what does that mean?' or You are unsure what the user is refering to). Default N=3.\n"
                "- 'lookup': Substring/keyword search across all past turns. Best for finding specific past details "
                "from earlier in the chat. This is Regex/substring search. FOR EXACT KEYWORDS AND STRINGS. .\n"
                "- 'ask': Passes the entire history to a secondary LLM to answer a specific meta-question "
                "(e.g., 'What exactly did the user want to change their name to?').\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum":["lookup", "summarize", "recent", "ask"],
                        "description": "The specific query mode to use to retrieve the history."
                    },
                    "query": {
                        "type": "string",
                        "description": "The search term (for 'lookup' mode) or the meta-question (for 'ask' mode)."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent turns to return (for 'recent' mode). Default is 3."
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