"""
cogops/session/summarizer.py

Background summarizer: after each answer, re-summarize recent turns
via secondary LLM. Rolls over existing summary, preserving unresolved
questions and entities.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """\
Update the conversation summary. Keep it under {max_tokens} tokens.
Preserve: unresolved questions, entities, user preferences.
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
    """Fire on secondary LLM to produce updated summary. Returns new summary or None."""
    if not secondary_client or not secondary_model:
        return None

    current_summary = store.get_summary(user_id) or "(empty)"

    try:
        response = await asyncio.wait_for(
            secondary_client.chat.completions.create(
                model=secondary_model,
                messages=[{
                    "role": "user",
                    "content": SUMMARIZER_PROMPT.format(
                        max_tokens=max_tokens,
                        current_summary=current_summary,
                        user_turn=user_turn,
                        assistant_turn=assistant_turn,
                    ),
                }],
                max_tokens=max_tokens,
                temperature=0.3,
            ),
            timeout=30.0,
        )
        new_summary = response.choices[0].message.content or ""
        store.set_summary(user_id, new_summary)
        logger.info("Summarizer updated for %s (%d chars).", user_id, len(new_summary))
        return new_summary
    except asyncio.TimeoutError:
        logger.warning("Summarizer timed out for %s.", user_id)
        return None
    except Exception as e:
        logger.error("Summarizer failed for %s: %s", user_id, e, exc_info=True)
        return None


async def run_summarizer_task(*args, **kwargs) -> Optional[str]:
    """Wrapper to safely run in background without blocking response."""
    try:
        return await summarize_and_update(*args, **kwargs)
    except Exception as e:
        logger.error("Summarizer background task failed: %s", e)
        return None
