"""
cogops/tools/secondary/grep_passage.py

grep_passage: regex grep over a long passage. Pure regex, no LLM.
"""

import re
import logging

logger = logging.getLogger(__name__)


async def grep_passage(passage: str, pattern: str, context_lines: int = 2) -> str:
    """
    Search a passage for regex matches, returning matched lines with context.

    Args:
        passage: the full text to search
        pattern: regex pattern
        context_lines: number of surrounding context lines
    """
    if not passage or not pattern:
        return "No passage or pattern provided."

    try:
        lines = passage.splitlines()
        flags = re.IGNORECASE
        matches = []
        for i, line in enumerate(lines):
            if re.search(pattern, line, flags):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(f"  {j+1}: {lines[j]}" for j in range(start, end))
                matches.append(f"Match at line {i+1}:\n{context}")
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    if matches:
        return "\n\n".join(matches)
    return f"No matches for pattern '{pattern}' in the passage."


grep_passage_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "grep_passage",
            "description": "Search a text passage for a regex pattern, returning matched lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passage": {
                        "type": "string",
                        "description": "The full text to search."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of surrounding context lines (default 2)."
                    }
                },
                "required": ["passage", "pattern"]
            }
        }
    }
]

grep_passage_tools_map = {
    "grep_passage": grep_passage
}
