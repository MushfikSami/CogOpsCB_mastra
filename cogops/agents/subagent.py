"""
cogops/agents/subagent.py

SubAgent: a self-contained reasoning loop on the secondary LLM with a
restricted tool set. Spawned via the `spawn_subagent` tool in the
primary orchestrator.
"""

import json
import logging
from typing import Any, Callable, Dict, List

from cogops.prompts.subagent import SUBAGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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

    async def run(self, task: str, max_turns: int = 5) -> str:
        """Run the sub-agent. Returns the final text answer."""
        tools_desc = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)
        system_prompt = (
            SUBAGENT_SYSTEM_PROMPT
            + f"\n\nAvailable tools:\n{tools_desc}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]

        answer_accumulator = ""
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
