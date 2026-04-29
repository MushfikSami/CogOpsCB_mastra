"""
cogops/tools/search_wiki.py

search_wiki: query the Bangladesh-focused Wikipedia search API for general knowledge
passages. Returns formatted context with article titles, URLs, and published dates.

The agent only sees combined_context as the observation.
Results metadata (title, url, published_at) is logged in debug mode.
"""

import os
import logging
import httpx

logger = logging.getLogger(__name__)

WIKI_ENDPOINT = os.getenv(
    "WIKIPEDIA_DATA__QUERY_END_POINT",
    "http://172.22.11.241:9220/search",
)


async def search_wiki(formal_query: str, keyword_string: str) -> str:
    """
    Search the Bangladesh-focused Wikipedia database for general knowledge passages.

    Args:
        formal_query: The question in formal Bengali, expressing the exact information sought.
        keyword_string: Space-separated Bengali keywords (3-8 words) extracted from the query.
    """
    if not formal_query or not keyword_string:
        return "No query or keywords provided."

    try:
        url = _get_endpoint()
        payload = {
            "formal_query": formal_query,
            "keyword_string": keyword_string,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            return f"Wiki search failed (HTTP {resp.status_code}): {resp.text}"

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results", [])

        if not combined and not results:
            return "No relevant results found."

        # Format: show combined_context first (what the agent sees as observation)
        lines = [f"Query: {formal_query}"]
        if combined:
            lines.append(f"Context:\n{combined}")

        # Add results metadata for reference
        if results:
            lines.append("\n--- Results ---")
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                pub = r.get("published_at", "")
                lines.append(
                    f"{i}. [{title}]({url})" + (f" (updated: {pub})" if pub else "")
                )

        return "\n\n".join(lines)

    except Exception as e:
        logger.error(f"search_wiki error: {e}")
        return f"Wiki search error: {e}"


def _get_endpoint() -> str:
    return os.getenv(
        "WIKIPEDIA_DATA__QUERY_END_POINT",
        "http://172.22.11.241:9220/search",
    )


search_wiki_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Search the Bangladesh-focused Wikipedia database for general knowledge. "
                "Use for questions about Bangladesh (history, geography, politics, policy, "
                "current events), world events, public figures, or general knowledge not "
                "specific to government procedures. This is the fallback when "
                "search_knowledge returns no results for a government service query, and "
                "the primary choice for non-government general knowledge questions. "
                "The API returns LLM-extracted excerpts (only the most relevant lines from "
                "each article) — the combined_context field contains all excerpts joined "
                "together, already filtered to minimize token usage. "
                "The results array (visible in debug logs) contains article titles, URLs, "
                "and published_at timestamps for reference."
                "\n\n"
                "Parameters:\n"
                "- formal_query: Write the exact question you need answered in formal Bengali (বাংলা). "
                "Use proper Bengali vocabulary and tone.\n"
                "- keyword_string: Space-separated Bengali keywords (3-8 words) extracted from your "
                "query. These are the key terms that appear in the database text. Pipe-separated "
                "keywords also work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formal_query": {
                        "type": "string",
                        "description": "The question in formal Bengali, expressing the exact information sought.",
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

search_wiki_tools_map = {
    "search_wiki": search_wiki,
}
