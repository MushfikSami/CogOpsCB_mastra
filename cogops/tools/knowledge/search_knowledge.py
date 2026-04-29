"""
cogops/tools/knowledge/search_knowledge.py

search_knowledge: query the Jiggasha knowledge API for government service
passages. Returns formatted node/text results ranked by relevance.
"""

import os
import logging
import httpx

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = os.getenv(
    "JIGGASHA_DATA__QUERY_END_POINT",
    "http://172.22.11.241:9210/search",
)


async def search_knowledge(query: str, top_k: int = 10) -> str:
    """
    Search the knowledge base for relevant government service passages.

    Args:
        query: User query in Bengali (colloquial, Banglish, or formal).
        top_k: Number of results to return (5-50, default 10).
    """
    if not query:
        return "No query provided."

    try:
        url = f"{_get_endpoint()}?q={query}&top_k={top_k}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url)

        if resp.status_code != 200:
            return f"Search failed (HTTP {resp.status_code}): {resp.text}"

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No relevant results found."

        # Format as numbered list of node + text
        lines = []
        for i, r in enumerate(results, 1):
            node = r.get("node", "")
            text = r.get("text", "")
            score = r.get("score", 0.0)
            lines.append(
                f"{i}. node: {node}\ntext: {text} (score: {score:.4f})"
            )

        formal = data.get("formal_query", "")
        lines.insert(0, f"Query: {query}")
        if formal:
            lines.insert(1, f"Reformulated: {formal}")

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
                "Search the Bangladesh government knowledge base for relevant "
                "service passages. Reformulates colloquial Bengali queries into "
                "formal Bengali and returns ranked results with node paths. "
                "Covers জন্ম/মৃত্যু নিবন্ধন, শিক্ষা, পাসপোর্ট, ভূমি, ট্রেড লাইসেন্স, "
                "যানবাহন, ইউটিলিটি, পেনশন, দূর্যোগ ব্যবস্থাপনা, সামাজিক সুরক্ষা, "
                "আইন ও নিরাপত্তা, স্বাস্থ্য সহ ৩০+ সরকারি সেবা। "
                "IMPORTANT: formulate the query in the most formal Bengali possible "
                "while maintaining the user's original intent and the specific "
                "information they need. The API handles colloquial→formal reformulation, "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query in Bengali (colloquial, Banglish, or formal).",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

search_knowledge_tools_map = {
    "search_knowledge": search_knowledge,
}
