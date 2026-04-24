"""
cogops/tools/graph/similar_entities.py

Given an entity name, find semantically similar entities via vector cosine similarity.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def similar_entities(entity_name: str, max_results: int = 10, min_score: float = 0.5) -> str:
    """Find semantically similar entities via vector cosine similarity."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'similar_entities')
    max_results = max_results or cfg.get('max_results', 10)
    min_score = min_score or cfg.get('min_score', 0.5)

    md = f"## Similar Entities to '{entity_name}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (target:Entity {name: $name}) "
            "WITH target.name_embedding AS vec, target.uuid AS exclude "
            "LIMIT 1 "
            "UNWIND [1] AS _ "
            "MATCH (n:Entity) "
            "WHERE n.uuid <> exclude AND n.name_embedding IS NOT NULL "
            "WITH n, vector.similarity.cosine(n.name_embedding, vec) AS score "
            "WHERE score >= $minScore "
            "RETURN n.name, n.summary, score "
            "ORDER BY score DESC LIMIT $maxResults",
            name=entity_name,
            minScore=min_score,
            maxResults=max_results
        )
        records = await result.data()

    if not records:
        md += f"No similar entities found for '{entity_name}' (min_score={min_score}).\n"
        return md

    md += f"Found {len(records)} similar entities:\n\n"
    md += "| # | Entity | Similarity Score | Summary |\n"
    md += "|---|--------|-----------------|---------|\n"
    for i, r in enumerate(records, 1):
        md += f"| {i} | {r['n.name']} | {r['score']:.4f} | {r['n.summary'] or '(no summary)'} |\n"

    return md


similar_entities_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "similar_entities",
            "description": "Given an entity name, find semantically similar entities via vector cosine similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to find similar entities for."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 10)."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (default 0.5)."
                    }
                },
                "required": ["entity_name"]
            }
        }
    }
]

similar_entities_tools_map = {
    "similar_entities": similar_entities
}
