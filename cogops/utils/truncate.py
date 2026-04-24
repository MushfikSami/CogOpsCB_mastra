"""
cogops/utils/truncate.py

Truncation utilities for the new architecture:
- truncate_text_to_tokens: cap a string to N tokens
- truncate_messages_to_budget: drop oldest non-system messages to fit context budget
"""

import logging
from typing import List, Dict, Any

from cogops.utils.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def truncate_text_to_tokens(text: str, max_tokens: int, model_name: str) -> str:
    """Truncate a string to at most max_tokens using the tokenizer."""
    if not text:
        return ""
    tk = Tokenizer(model_name)
    if tk.count(text) <= max_tokens:
        return text
    encoded = tk.tokenizer.encode(text)
    truncated = encoded[:max_tokens]
    result = tk.tokenizer.decode(truncated, skip_special_tokens=True)
    logger.warning(f"Text truncated from {len(encoded)} to {max_tokens} tokens.")
    return result


def truncate_messages_to_budget(messages: List[Dict[str, Any]], max_tokens: int, keep_system: bool, model_name: str) -> List[Dict[str, Any]]:
    """
    Drop oldest non-system messages from messages list so the total token count fits max_tokens.
    Always keeps system messages first (keep_system=True).
    """
    tk = Tokenizer(model_name)

    def total_tokens(msgs):
        return sum(tk.count(str(m)) for m in msgs)

    # Keep system messages at the front
    if keep_system:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
    else:
        system_msgs = []
        other_msgs = list(messages)

    # Try dropping from the oldest (beginning) of non-system messages
    while total_tokens(system_msgs + other_msgs) > max_tokens and len(other_msgs) > 1:
        other_msgs.pop(0)

    if total_tokens(system_msgs + other_msgs) > max_tokens:
        # Hard truncate the last message
        logger.warning("Messages still exceed budget after dropping oldest; applying hard truncation.")
        for i in range(len(other_msgs) - 1, -1, -1):
            content = other_msgs[i].get("content", "")
            if content:
                truncated = truncate_text_to_tokens(content, max_tokens - total_tokens(system_msgs + other_msgs[:i]), model_name)
                other_msgs[i] = {**other_msgs[i], "content": truncated}

    return system_msgs + other_msgs
