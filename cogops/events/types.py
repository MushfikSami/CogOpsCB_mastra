"""
cogops/events/types.py

Event types for the dynamic reasoning agent.
Every event has: type, channel, and data fields.
"""

from enum import Enum
from typing import Any, Dict


class Channel(str, Enum):
    USER = "user"
    DEBUG = "debug"
    BOTH = "both"


# Event factory
def event(event_type: str, data: Dict[str, Any], channel: str = "user") -> Dict[str, Any]:
    """Create a tagged event dict."""
    return {"type": event_type, "channel": channel, **data}

# Convenience constants
ANSWER_CHUNK_TYPE = "answer_chunk"
REASONING_CHUNK_TYPE = "reasoning_chunk"
TOOL_CALL_TYPE = "tool_call"
TOOL_RESULT_TYPE = "tool_result"
TURN_START_TYPE = "turn_start"
TURN_END_TYPE = "turn_end"
ANSWER_COMPLETE_TYPE = "answer_complete"
ERROR_TYPE = "error"
