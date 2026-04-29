"""
cogops/tools/answer_directly.py

Meta-tool used when the model should answer without calling an information tool:
chit-chat, identity questions, and safety-tier replies (deflect, de-escalate, refuse).

The tool_choice="required" enforcement in the reasoning loop means the model must
always pick some tool. For non-factual turns this meta-tool satisfies that
constraint cleanly: the model places its full reply in `text` with a `category`
tag, and the reasoning loop short-circuits to stream that reply verbatim.
"""

ANSWER_DIRECTLY_CATEGORIES = (
    "chitchat",
    "identity",
    "safety_deflect",
    "abuse",
    "illegal",
    "no_info_found",
)

ANSWER_DIRECTLY_SENTINEL = "__ANSWER_DIRECTLY__"


async def answer_directly(category: str, text: str) -> str:
    """Return the model's direct reply wrapped in a sentinel the loop recognises."""
    if category not in ANSWER_DIRECTLY_CATEGORIES:
        return (
            f"Invalid category '{category}'. "
            f"Use one of: {', '.join(ANSWER_DIRECTLY_CATEGORIES)}."
        )
    if not text or not text.strip():
        return "Invalid: `text` must be non-empty."
    return f"{ANSWER_DIRECTLY_SENTINEL}::{category}::{text}"


answer_directly_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "answer_directly",
            "description": (
                "Respond directly to the user WITHOUT calling any information tool. "
                "Use ONLY for: (1) chit-chat / greetings, (2) questions about YOUR OWN "
                "identity or capabilities (e.g. 'who are you?'), NOT about third parties "
                "or public figures, (3) safety replies (deflecting political/"
                "controversial topics, de-escalating abuse, refusing illegal requests). "
                "Never use this for factual questions about services, procedures, fees, "
                "entities, or any information the user wants looked up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": list(ANSWER_DIRECTLY_CATEGORIES),
                        "description": (
                            "chitchat = greetings/small talk; "
                            "identity = questions about YOUR OWN identity only ('who are you?'), NOT about third parties; "
                            "safety_deflect = political/religious/controversial topics; "
                            "abuse = abusive/insulting user input (de-escalate); "
                            "illegal = dangerous/illegal requests (refuse); "
                            "no_info_found = both search_knowledge and search_wiki returned no results (reply politely in Bengali that no information is available)."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "The complete user-facing reply, in Formal Bengali "
                            "(প্রমিত বাংলা). This text is streamed verbatim to the user."
                        ),
                    },
                },
                "required": ["category", "text"],
                "additionalProperties": False,
            },
        },
    }
]

answer_directly_tools_map = {
    "answer_directly": answer_directly,
}
