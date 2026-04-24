"""
cogops/tools/secondary/extract_from_doc.py

extract_from_document: secondary LLM extracts relevant spans from a long doc.
"""

import logging

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """
Extract everything relevant from the document below about the following topic.
Return a concise, structured list. Only include what is relevant.

Topic: {topic}

Document:
{document}
"""


async def extract_from_document(
    document: str,
    topic: str,
    secondary_client=None,
    secondary_model: str = "",
) -> str:
    """
    Use secondary LLM to extract relevant information from a long document.

    Args:
        document: the full document text
        topic: what to extract
        secondary_client: AsyncOpenAI client
        secondary_model: model name
    """
    if not secondary_client:
        return "Secondary LLM not configured. Cannot extract."

    from cogops.llm.secondary import call_secondary

    messages = [{"role": "user", "content": EXTRACT_PROMPT.format(topic=topic, document=document[:8000])}]
    return await call_secondary(secondary_client, secondary_model, messages, max_tokens=2048)


extract_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "extract_from_document",
            "description": "Extract relevant information from a long document using a secondary LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "The full document text."
                    },
                    "topic": {
                        "type": "string",
                        "description": "What to extract from the document."
                    }
                },
                "required": ["document", "topic"]
            }
        }
    }
]

extract_tools_map = {
    "extract_from_document": extract_from_document
}
