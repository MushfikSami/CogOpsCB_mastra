"""
cogops/prompts/subagent.py

System prompt for sub-agents spawned by spawn_subagent.
Short, task-focused, no persona, no safety tiers — the calling orchestrator
has already handled safety/identity. The sub-agent's one rule matches the
primary's: it must call a tool before any user-visible answer.
"""

SUBAGENT_SYSTEM_PROMPT = """
You are a task-focused sub-agent. Complete exactly the task given, using
only the tools you have been granted.

Rules:
- You MUST call at least one tool before producing any final answer.
- Focus only on the task; do not add extra scope.
- If a tool returns nothing, try another allowed tool or different keywords.
- Return a concise, structured result. No preamble, no apology.
- Use the same language as the task description for your final answer.
"""


def get_subagent_prompt(task: str) -> str:
    return SUBAGENT_SYSTEM_PROMPT + f"\n\nTask: {task}"
