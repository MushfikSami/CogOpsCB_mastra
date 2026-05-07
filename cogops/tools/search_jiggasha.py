"""
cogops/tools/search_jiggasha.py

Search tool that calls the Jiggasha government services search endpoint.
Uses the same API as the original search_knowledge tool (formal_query + keyword_string)
since the external service at 172.22.11.241:9210/search expects that format.

TODO: When jiggasha_service is deployed and reachable, switch to its /query endpoint.
"""

import os
import logging
import httpx
from typing import Tuple, List, Union

logger = logging.getLogger(__name__)


def _get_endpoint() -> str:
    from cogops.config.loader import load_config
    try:
        cfg = load_config()
        env_name = cfg.get("jiggasha", {}).get("endpoint_env", "JIGGASHA_ENDPOINT")
        default = cfg.get("jiggasha", {}).get("endpoint_default", "http://172.22.11.241:9210/search")
        return os.getenv(env_name, default)
    except Exception:
        return os.getenv("JIGGASHA_ENDPOINT", "http://172.22.11.241:9210/search")


def _get_timeout() -> float:
    from cogops.config.loader import load_config
    try:
        cfg = load_config()
        return cfg.get("jiggasha", {}).get("timeout", 30)
    except Exception:
        return 30.0


async def search_knowledge(query: str, **_injectable) -> Tuple[Union[str, List[str]], List[str]]:
    """
    Search Bangladesh government services via Jiggasha.

    Args:
        query: Natural language query in Bengali or English.
        **_injectable: Server-side injected params — unused.

    Returns:
        (formatted_text_for_model, sources_list)
    """
    if not query or not query.strip():
        return "No query provided.", []

    query = query.strip()
    timeout = _get_timeout()

    try:
        payload = {
            "formal_query": query,
            "keyword_string": query,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(_get_endpoint(), json=payload)

        if resp.status_code != 200:
            return f"Search failed (HTTP {resp.status_code}): {resp.text[:200]}", []

        data = resp.json()
        combined = data.get("combined_context", "")
        results = data.get("results", [])

        if not combined and not results:
            return "No relevant information found in the government services database.", []

        context_parts = [f"Query: {query}"]
        if combined:
            context_parts.append(f"Retrieved Context:\n{combined}")

        sources_list = []
        if results:
            for i, r in enumerate(results[:5]):
                node = r.get("node", "")
                score = r.get("score", 0.0)
                title = r.get("title", "")
                sources_list.append(f"[{i+1}] ({title or node}, score={score:.3f})")

        return context_parts, sources_list

    except httpx.TimeoutException:
        logger.warning("Jiggasha search timed out: %s", query[:50])
        return "Jiggasha search timed out. Try again or try search_wiki.", []
    except Exception as e:
        logger.error("search_knowledge error: %s", e)
        return f"Jiggasha search error: {e}", []


# --- Lean Tool Schema ---
search_knowledge_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search Bangladesh government services via Jiggasha database. "
                "Use for questions about government procedures, fees, documents, "
                "offices, departments, regulations, licenses, permits, benefits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query in Bengali or English about government services.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

search_knowledge_tools_map = {
    "search_knowledge": search_knowledge,
}
