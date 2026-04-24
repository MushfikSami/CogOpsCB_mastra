"""
cogops/tools/secondary/spawn_subagent.py

spawn_subagent: runs a self-contained reasoning loop on secondary LLM.
"""

import json
import logging
from typing import Any, Dict, List, Callable

logger = logging.getLogger(__name__)


class ClarificationRequested(Exception):
    """Raised when the agent wants to ask the user for clarification."""
    def __init__(self, question: str, options: List[str] = None, reason: str = None, turn_id: str = None):
        super().__init__(question)
        self.question = question
        self.options = options or []
        self.reason = reason
        self.turn_id = turn_id


async def spawn_subagent(
    task: str,
    allowed_tools: List[str],
    tool_map: Dict[str, Callable],
    tools_schema: List[Dict[str, Any]],
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """
    Run a self-contained sub-agent reasoning loop on the secondary LLM.

    Args:
        task: the sub-agent task description
        allowed_tools: tool names from registry
        tool_map: full name -> callable map
        tools_schema: full tool schema list
        secondary_client: AsyncOpenAI client
        secondary_model: model name
    """
    if not secondary_client:
        return "Secondary LLM not configured."

    # Filter tool schema and map to only allowed tools
    filtered_schema = [t for t in tools_schema if t["function"]["name"] in allowed_tools]
    filtered_map = {k: v for k, v in tool_map.items() if k in allowed_tools}

    if not filtered_schema:
        return f"No valid tools found among: {allowed_tools}"

    from cogops.agents.subagent import SubAgent

    sub = SubAgent(
        secondary_client=secondary_client,
        secondary_model=secondary_model,
        tools_schema=filtered_schema,
        tool_map=filtered_map,
    )

    try:
        result = await sub.run(task, max_turns=5)
        return f"SubAgent result:\n{result}"
    except Exception as e:
        logger.error(f"spawn_subagent failed: {e}")
        return f"SubAgent error: {e}"


spawn_subagent_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": "Spawn a self-contained sub-agent on the secondary LLM with a restricted tool set. Use for multi-step subtasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description for the sub-agent."
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the sub-agent is allowed to use."
                    }
                },
                "required": ["task", "allowed_tools"]
            }
        }
    }
]

spawn_subagent_tools_map = {
    "spawn_subagent": spawn_subagent
}
