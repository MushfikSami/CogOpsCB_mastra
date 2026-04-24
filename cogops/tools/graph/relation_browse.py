"""
cogops/tools/graph/relation_browse.py

List ALL available relation-name values with their edge counts.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def relation_browse(filter_prefix=None, top_n: int = 100) -> str:
    """List all available relation-name values with their edge counts."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'relation_browse')
    top_n = top_n or cfg.get('top_n', 100)
    filter_prefix = filter_prefix or cfg.get('filter_prefix')

    md = "## Available Relation Types\n\n"
    md += "| # | Relation Name | Edge Count |\n"
    md += "|---|---------------|------------|\n"

    async with driver.session(database="qwen34neo4j") as session:
        where_clause = "WHERE r.name IS NOT NULL"
        if filter_prefix:
            where_clause += f" AND r.name STARTS WITH $filterPrefix"
        result = await session.run(
            f"MATCH ()-[r:RELATES_TO]->() {where_clause} "
            "WITH r.name AS relName, count(*) AS cnt "
            "ORDER BY cnt DESC "
            "LIMIT $topN "
            "RETURN relName, cnt",
            filterPrefix=filter_prefix,
            topN=top_n
        )
        records = await result.data()

    if not records:
        md += "No relation types found.\n"
        return md

    for i, r in enumerate(records, 1):
        md += f"| {i} | {r['relName']} | {r['cnt']} |\n"

    md += f"\nTotal: {len(records)} relation types shown (out of {top_n} max).\n"
    return md


relation_browse_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "relation_browse",
            "description": "List all available relation-name values with their edge counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_prefix": {
                        "type": "string",
                        "description": "Optional prefix filter (e.g., 'REQUIRES')."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max results (default 100)."
                    }
                },
                "required": []
            }
        }
    }
]

relation_browse_tools_map = {
    "relation_browse": relation_browse
}
