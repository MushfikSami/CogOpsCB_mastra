"""
cogops/llm/secondary.py

One-shot secondary LLM caller. Non-streaming, used by tools (extract, delegate).
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


async def call_secondary(
    secondary_client,
    secondary_model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    One-shot call to secondary LLM. Returns the assistant message content.

    Args:
        secondary_client: AsyncOpenAI client
        secondary_model: model name
        messages: OpenAI-compatible message list
        max_tokens: token limit
        temperature: sampling temperature
        extra_body: additional vLLM params
    """
    if not secondary_client:
        return "No secondary LLM configured."

    try:
        response = await secondary_client.chat.completions.create(
            model=secondary_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body or {},
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Secondary LLM call failed: {e}")
        return f"Error: {e}"
