"""
cogops/agents/subagent.py

SubAgent: a self-contained reasoning loop on the secondary LLM with a restricted tool set.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

SUBAGENT_PROMPT = """
You are a task-focused assistant. Complete the following task.

Task: {task}

You have access to the following tools. Only use the ones listed.

TOOLS:
{tools_description}

Respond by calling tools as needed, then provide a final answer.
Keep your answer concise and focused on the task.
"""


class SubAgent:
    """Runs a reasoning loop on secondary LLM with a restricted tool set."""

    def __init__(
        self,
        secondary_client,
        secondary_model: str,
        tools_schema: List[Dict[str, Any]],
        tool_map: Dict[str, Callable],
    ):
        self.secondary_client = secondary_client
        self.secondary_model = secondary_model
        self.tools_schema = tools_schema
        self.tool_map = tool_map

    async def run(
        self,
        task: str,
        max_turns: int = 5,
    ) -> str:
        """
        Run the sub-agent. Returns the final text answer.
        Raises StopOnContent if the sub-agent produces an answer without tool calls.
        """
        tools_desc = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)
        system_prompt = SUBAGENT_PROMPT.format(task=task, tools_description=tools_desc)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\nExecute this task using the available tools."},
        ]

        answer_accumulator = ""
        import asyncio
        from cogops.llm.reasoning_loop import stream_with_tool_calls

        try:
            async for event in stream_with_tool_calls(
                client_llm=self.secondary_client,
                model=self.secondary_model,
                messages=messages,
                tools_schema=self.tools_schema,
                available_tools=self.tool_map,
                max_turns=max_turns,
            ):
                if event["type"] == "answer_chunk":
                    answer_accumulator += event.get("content", "")

            return answer_accumulator.strip()

        except Exception as e:
            logger.error(f"SubAgent failed: {e}")
            return f"SubAgent error: {e}"
