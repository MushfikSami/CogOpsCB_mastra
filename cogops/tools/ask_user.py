"""
cogops/tools/ask_user.py

ask_user tool: returns a text question for the user.
Treated as a normal tool — the answer flows through the standard answer pipeline.
"""

import logging

logger = logging.getLogger(__name__)


async def ask_user(question: str, options=None, reason=None) -> str:
    """
    Ask the user for clarification.

    This tool should be called when the user's intent is ambiguous
    or a search returned too many unrelated matches.

    Args:
        question: the question to ask the user (in Bangla)
        options: 2-4 concrete options for the user to choose from
        reason: why clarification is needed

    Returns:
        A formatted string with the question and options.
    """
    result = question
    if options:
        result += "\n\nOptions:\n" + "\n".join(f"- {o}" for o in options)
    if reason:
        result += f"\n\nReason: {reason}"
    return result


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
