"""
cogops/tools/graph/node_explore.py

Given an entity name, return ALL connections (incoming + outgoing).
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def node_explore(entity_name: str, max_results: int = 100) -> str:
    """Given an entity name, return all connections grouped by relation type."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'node_explore')
    max_results = max_results or cfg.get('max_results', 100)

    md = f"## Connections for '{entity_name}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(neighbor:Entity) "
            "RETURN e.name AS entity, neighbor.name AS other_entity, "
            "r.name AS relation_type, r.fact AS fact "
            "ORDER BY r.name, other_entity "
            "LIMIT $max",
            name=entity_name,
            max=max_results
        )
        records = await result.data()

    if not records:
        md += f"No connections found for entity: '{entity_name}'\n"
        return md

    # Group by relation type
    grouped: Dict[str, list] = {}
    for r in records:
        rt = r.get("relation_type", "unknown")
        grouped.setdefault(rt, []).append(r)

    for rel_type, items in sorted(grouped.items()):
        md += f"\n### Relation: {rel_type} ({len(items)} edges)\n"
        for item in items:
            fact = item.get("fact", "")
            fact_preview = fact[:100] + "..." if len(fact) > 100 else fact
            md += f"- **{item['other_entity']}**: {fact_preview}\n"

    return md


node_explore_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "node_explore",
            "description": "Given an entity name, return all connections (incoming + outgoing) grouped by relation type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to explore."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 100)."
                    }
                },
                "required": ["entity_name"]
            }
        }
    }
]

node_explore_tools_map = {
    "node_explore": node_explore
}
