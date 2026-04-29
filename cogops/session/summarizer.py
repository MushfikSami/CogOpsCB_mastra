"""
cogops/session/summarizer.py

Background summarizer: after each answer, re-summarize recent turns via secondary LLM.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """
Update the conversation summary. Keep it concise (under 300 tokens).
Preserve: unresolved questions, entities mentioned, user preferences.
Drop: resolved details, exact quotes.

Current summary:
{current_summary}

New turn:
User: {user_turn}
Assistant: {assistant_turn}

Updated summary:
"""


async def summarize_and_update(
    secondary_client,
    secondary_model: str,
    user_id: str,
    store,  # RedisSessionStore
    user_turn: str,
    assistant_turn: str,
    max_tokens: int = 300,
) -> Optional[str]:
    """
    Fires on secondary LLM to produce an updated summary.
    Returns the new summary string (or None if failed).
    """
    if not store.available:
        return None

    current_summary = store.get_summary(user_id) or "(empty)"

    try:
        response = await asyncio.wait_for(
            secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[{"role": "user", "content": SUMMARIZER_PROMPT.format(
                    current_summary=current_summary,
                    user_turn=user_turn,
                    assistant_turn=assistant_turn,
                )}],
                max_tokens=max_tokens,
                temperature=0.3,
            ),
            timeout=30.0,
        )
        new_summary = response.choices[0].message.content or ""
        store.set_summary(user_id, new_summary)
        logger.info(f"Summarizer updated summary for {user_id} ({len(new_summary)} chars).")
        return new_summary
    except Exception as e:
        logger.error(f"Summarizer failed for {user_id}: {e}", exc_info=True)
        return None


async def run_summarizer_task(*args, **kwargs) -> Optional[str]:
    """Wrapper to safely run in background without blocking the response."""
    try:
        return await summarize_and_update(*args, **kwargs)
    except Exception as e:
        logger.error(f"Summarizer background task failed: {e}")
        return None
