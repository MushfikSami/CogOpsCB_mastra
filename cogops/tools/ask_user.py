"""
cogops/tools/ask_user.py

ask_user tool + ClarificationRequested exception.
Always available, no LLM call.
"""

import uuid
import logging

logger = logging.getLogger(__name__)


class ClarificationRequested(Exception):
    """Raised when the agent wants to ask the user for clarification."""
    def __init__(self, question: str, options=None, reason=None, turn_id=None):
        super().__init__(question)
        self.question = question
        self.options = options or []
        self.reason = reason
        self.turn_id = turn_id or str(uuid.uuid4())[:8]


async def ask_user(question: str, options=None, reason=None) -> str:
    """
    Request clarification from the user.

    This tool should be called when the user's intent is ambiguous
    or a search returned too many unrelated matches.
    It raises ClarificationRequested which terminates the current stream.

    Args:
        question: the question to ask the user (in Bangla)
        options: 2-4 concrete options for the user to choose from
        reason: why clarification is needed

    Raises:
        ClarificationRequested — never returns normally
    """
    raise ClarificationRequested(question=question, options=options, reason=reason)


ask_user_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user for clarification. Use when the query is ambiguous or a search returned too many results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 concrete options for the user."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why clarification is needed."
                    }
                },
                "required": ["question"]
            }
        }
    }
]

ask_user_tools_map = {
    "ask_user": ask_user
}
