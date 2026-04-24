"""
cogops/tools/secondary/delegate_task.py

delegate_task: one-shot instruction task to secondary LLM.
"""

import logging

logger = logging.getLogger(__name__)

DELEGATE_PROMPT = """
{instruction}

Context:
{context}
"""


async def delegate_task(
    instruction: str,
    context: str = "",
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """
    One-shot task delegation to secondary LLM.

    Args:
        instruction: focused task instruction
        context: additional context
        secondary_client: AsyncOpenAI client
        secondary_model: model name
    """
    if not secondary_client:
        return "Secondary LLM not configured."

    from cogops.llm.secondary import call_secondary

    messages = [{"role": "user", "content": DELEGATE_PROMPT.format(instruction=instruction, context=context)}]
    return await call_secondary(secondary_client, secondary_model, messages, max_tokens=2048, temperature=0.7)


delegate_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Delegate a focused task to a secondary LLM. Returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Focused task instruction."
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context (optional)."
                    }
                },
                "required": ["instruction"]
            }
        }
    }
]

delegate_tools_map = {
    "delegate_task": delegate_task
}
