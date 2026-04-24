"""
cogops/tools/graph/graph_stats.py

Get graph-level statistics: node counts, relation type distribution, degree distribution.
"""

import logging
from typing import Dict, Any

from dotenv import load_dotenv
from cogops.config.loader import load_config, get_tool_config

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)


async def graph_stats(detail_level: str = "basic") -> str:
    """Get graph-level statistics."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver
    cfg = get_tool_config(CONFIG, 'graph_stats')
    detail_level = detail_level or cfg.get('detail_level', 'basic')

    md = "## Graph Statistics\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        # Node counts
        result = await session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt "
            "ORDER BY cnt DESC"
        )
        node_counts = await result.data()

        md += "### Node Counts\n\n"
        md += "| Label | Count |\n"
        md += "|-------|-------|\n"
        for r in node_counts:
            md += f"| {r['label']} | {r['cnt']} |\n"

        # Edge count
        result = await session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN count(*) AS cnt"
        )
        edge_count = (await result.single())["cnt"]
        md += f"\n### Edge Count\n\nTotal RELATES_TO edges: {edge_count}\n"

        # Relation type distribution (top 20)
        if detail_level == "detailed":
            result = await session.run(
                "MATCH ()-[r:RELATES_TO]->() WHERE r.name IS NOT NULL "
                "WITH r.name AS relName, count(*) AS cnt "
                "ORDER BY cnt DESC LIMIT 20 "
                "RETURN relName, cnt"
            )
            top_rels = await result.data()

            md += "\n### Top 20 Relation Types\n\n"
            md += "| # | Relation Name | Count |\n"
            md += "|---|---------------|-------|\n"
            for i, r in enumerate(top_rels, 1):
                md += f"| {i} | {r['relName']} | {r['cnt']} |\n"

    return md


graph_stats_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "graph_stats",
            "description": "Get graph-level statistics: node counts, relation type distribution, degree distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail_level": {
                        "type": "string",
                        "enum": ["basic", "detailed"],
                        "description": "Detail level (default 'basic')."
                    }
                },
                "required": []
            }
        }
    }
]

graph_stats_tools_map = {
    "graph_stats": graph_stats
}
