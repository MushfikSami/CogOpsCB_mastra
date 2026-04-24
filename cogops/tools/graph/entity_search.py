"""
cogops/tools/graph/entity_search.py

Find entities by partial/fuzzy name match, ranked by match quality.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def entity_search(search_term: str, max_results: int = 10) -> str:
    """Find entities by partial/fuzzy name match."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver

    md = f"## Entity Search Results for '{search_term}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) CONTAINS toLower($term) "
            "RETURN e.name, e.summary, "
            "CASE WHEN e.name = $term THEN 0 "
            "WHEN toLower(e.name) = toLower($term) THEN 1 "
            "ELSE 2 END AS match_rank "
            "ORDER BY match_rank, e.name "
            "LIMIT $maxResults",
            term=search_term,
            maxResults=max_results
        )
        records = await result.data()

    if not records:
        md += "No matching entities found.\n"
        return md

    md += "| # | Name | Summary | Match Rank |\n"
    md += "|---|------|---------|------------|\n"
    for i, r in enumerate(records, 1):
        rank_label = "Exact" if r["match_rank"] == 0 else ("Case-insensitive" if r["match_rank"] == 1 else "Partial")
        md += f"| {i} | {r['e.name']} | {r['e.summary'] or '(no summary)'} | {rank_label} |\n"

    return md


entity_search_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "entity_search",
            "description": "Find entities by partial/fuzzy name match, ranked by match quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search term (e.g., 'passport')."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 10)."
                    }
                },
                "required": ["search_term"]
            }
        }
    }
]

entity_search_tools_map = {
    "entity_search": entity_search
}
