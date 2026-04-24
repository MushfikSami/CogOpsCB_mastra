"""
cogops/tools/graph/relation_filter.py

Given a relation name, return ALL entity pairs connected by it.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def relation_filter(relation_name: str, max_results: int = 50) -> str:
    """Given a relation name, return all entity pairs connected by it."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'relation_filter')
    max_results = max_results or cfg.get('max_results', 50)

    md = f"## {relation_name} Relationships\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "WHERE r.name = $relName "
            "RETURN a.name AS source, b.name AS target, r.fact AS fact "
            "ORDER BY a.name, b.name "
            "LIMIT $maxResults",
            relName=relation_name,
            maxResults=max_results
        )
        records = await result.data()

    if not records:
        md += f"No relationships found for relation type: '{relation_name}'\n"
        return md

    md += f"Found {len(records)} {relation_name} relationships:\n\n"
    md += "| # | Source | Target | Fact |\n"
    md += "|---|--------|--------|------|\n"
    for i, r in enumerate(records, 1):
        fact = str(r.get("fact", ""))[:120]
        md += f"| {i} | {r['source']} | {r['target']} | {fact} |\n"

    return md


relation_filter_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "relation_filter",
            "description": "Given a relation name, return all entity pairs connected by it with facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relation_name": {
                        "type": "string",
                        "description": "Relation type name (e.g., 'REQUIRES_DOCUMENT')."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 50)."
                    }
                },
                "required": ["relation_name"]
            }
        }
    }
]

relation_filter_tools_map = {
    "relation_filter": relation_filter
}
