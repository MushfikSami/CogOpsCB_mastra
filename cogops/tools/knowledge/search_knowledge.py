"""
cogops/tools/knowledge/search_knowledge.py

search_knowledge: query the Jiggasha knowledge API for Bangladesh government service
passages. Returns formatted node/text results ranked by relevance.

The agent only sees combined_context as the observation.
Results metadata (node paths, scores) is visible in debug logs.
"""

import os
import logging
import httpx

from cogops.config.loader import load_config

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = os.getenv(
    "JIGGASHA_DATA__QUERY_END_POINT",
    "http://172.22.11.241:9210/search",
)

# Cached config for top_k default
_knowledge_config: dict | None = None


def _get_knowledge_config() -> dict:
    global _knowledge_config
    if _knowledge_config is None:
        _knowledge_config = load_config()
    return _knowledge_config


def _default_top_k() -> int:
    return (
        _get_knowledge_config()
        .get("knowledge_search", {})
        .get("top_k_default", 10)
    )


async def search_knowledge(formal_query: str, keyword_string: str) -> str:
    """
    Search the Bangladesh government service database (Jiggasha) for relevant passages.

    Args:
        formal_query: The exact question in formal Bengali (বাংলা), expressing the
            information needed. Use proper Bengali vocabulary as on an official form.
        keyword_string: Space-separated Bengali keywords (3-8 words) extracted from the query.
            These are the key terms that appear in the database text.

    Returns:
        Formatted text with combined_context (the answer) and results metadata.
    """
    if not formal_query or not keyword_string:
        return "No query or keywords provided."

    top_k = _default_top_k()

    try:
        url = _get_endpoint()
        payload = {
            "formal_query": formal_query,
            "keyword_string": keyword_string,
            "top_k": top_k,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            return f"Search failed (HTTP {resp.status_code}): {resp.text}"

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results", [])

        if not combined and not results:
            return "No relevant results found."

        # Format: show combined_context first (what the agent sees as observation)
        lines = [f"Query: {formal_query}"]
        if combined:
            lines.append(f"Combined Context:\n{combined}")

        # Add results metadata for reference
        if results:
            lines.append("\n--- Results ---")
            for i, r in enumerate(results, 1):
                node = r.get("node", "")
                text = r.get("text", "")
                score = r.get("score", 0.0)
                lines.append(
                    f"{i}. node: {node}\ntext: {text} (score: {score:.4f})"
                )

        return "\n\n".join(lines)

    except Exception as e:
        logger.error(f"search_knowledge error: {e}")
        return f"Search error: {e}"


def _get_endpoint() -> str:
    return os.getenv(
        "JIGGASHA_DATA__QUERY_END_POINT",
        "http://172.22.11.241:9210/search",
    )


search_knowledge_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the Bangladesh government service database (Jiggasha) for relevant "
                "passages. Use for ANY question about Bangladesh government services, procedures, "
                "fees, document requirements, offices, boards, departments, or regulations. "
                "Covers 30+ services: education (শিক্ষা), passports (পাসপোর্ট), "
                "NID (জাতীয় পরিচয়পত্র), birth/death registration (জন্ম/মৃত্যু নিবন্ধন), "
                "land (ভূমি), trade licenses (ট্রেড লাইসেন্স), vehicles (যানবাহন), "
                "utilities (ইউটিলিটি), pensions (পেনশন), disaster management (দূর্যোগ "
                "ব্যবস্থাপনা), social safety (সামাজিক সুরক্ষা), law and security (আইন ও "
                "নিরাপত্তা), health (স্বাস্থ্য) and more. "
                "The API handles colloquial-to-formal Bengali reformulation internally. "
                "The combined_context field contains the answer — ranked, LLM-extracted "
                "passages trimmed to 1000 words max. "
                "The results array (visible in debug logs) contains node paths (hierarchical "
                "category structures), full passage text, and relevance scores."
                "\n\n"
                "Parameters:\n"
                "- formal_query: Write the exact question you need answered in formal Bengali (বাংলা). "
                "Use proper Bengali vocabulary — as you would write on an official government form. "
                "For example, use 'কীভাবে' instead of 'কিভাবে', 'পরিবর্তন' instead of 'চেঞ্জ'.\n"
                "- keyword_string: Space-separated Bengali keywords (3-8 words) extracted from your "
                "query. These are the key terms that would appear in the government database text. "
                "Think of the most important terms someone would use to find this information on a "
                "government website. Pipe-separated keywords also work."
                "\n\n"
                "If this tool returns no results (empty combined_context, 'No relevant results "
                "found', or all low scores), call search_wiki as your next action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formal_query": {
                        "type": "string",
                        "description": "The exact question in formal Bengali, expressing the information needed.",
                    },
                    "keyword_string": {
                        "type": "string",
                        "description": "Space-separated Bengali keywords (3-8 words) extracted from the query.",
                    },
                },
                "required": ["formal_query", "keyword_string"],
            },
        },
    }
]

search_knowledge_tools_map = {
    "search_knowledge": search_knowledge,
}
