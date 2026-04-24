"""
cogops/tools/secondary/spawn_subagent.py

spawn_subagent: runs a self-contained reasoning loop on the secondary LLM
with a restricted tool set.

`tool_map` and `tools_schema` are injected by the orchestrator via
ToolContext at bind time — they are NOT exposed to the model in the JSON
schema (they're Python objects, not serialisable).
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


async def spawn_subagent(
    task: str,
    allowed_tools: List[str],
    tool_map: Optional[Dict[str, Callable]] = None,
    tools_schema: Optional[List[Dict[str, Any]]] = None,
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """Run a self-contained sub-agent reasoning loop on the secondary LLM."""
    if not secondary_client:
        return "Secondary LLM not configured."
    if not tool_map or not tools_schema:
        return "SubAgent tool registry not wired. (Server bug.)"

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
            "description": (
                "Spawn a self-contained sub-agent on the secondary LLM with a "
                "restricted tool set. Use for multi-step subtasks where the sub-agent "
                "needs to make several dependent tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description for the sub-agent.",
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names the sub-agent is allowed to use.",
                    },
                },
                "required": ["task", "allowed_tools"],
                "additionalProperties": False,
            },
        },
    }
]

spawn_subagent_tools_map = {"spawn_subagent": spawn_subagent}
