"""
cogops/prompts/subagent.py

System prompt for sub-agents spawned by spawn_subagent.
Short, task-focused, no persona, no safety tiers.
"""

SUBAGENT_SYSTEM_PROMPT = """
You are a task-focused assistant. Complete the user's task using the available tools.

Rules:
- Focus only on the task at hand.
- Use tools to gather information as needed.
- Return a concise, structured answer.
- Do not include unnecessary preamble.
"""


def get_subagent_prompt(task: str) -> str:
    return SUBAGENT_SYSTEM_PROMPT + f"\n\nTask: {task}"
