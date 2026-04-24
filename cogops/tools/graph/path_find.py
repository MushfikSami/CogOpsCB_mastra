"""
cogops/tools/graph/path_find.py

Find paths between two entities (1-N hops).
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def path_find(start_entity: str, end_entity: str, max_hops: int = 3, max_paths: int = 5) -> str:
    """Find paths between two entities (1-N hops)."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver

    md = f"## Paths from '{start_entity}' to '{end_entity}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            f"MATCH path=(start:Entity {{name: $start}})-[r:RELATES_TO*1..{max_hops}]-(end:Entity {{name: $end}}) "
            "WHERE start <> end "
            "WITH path, [n IN nodes(path) | n.name] AS chain, "
            "              [r IN relationships(path) | r.name] AS relTypes "
            "ORDER BY size(chain) "
            "LIMIT $maxPaths "
            "RETURN chain, relTypes",
            start=start_entity,
            end=end_entity,
            maxPaths=max_paths
        )
        records = await result.data()

    if not records:
        md += f"No path found between '{start_entity}' and '{end_entity}' (max {max_hops} hops).\n"
        return md

    md += f"Found {len(records)} paths:\n\n"
    for i, r in enumerate(records, 1):
        chain = r["chain"]
        rels = r["relTypes"]
        path_str = " -> ".join(chain)
        rel_str = " --[{}]--> ".join(chain)
        md += f"**Path {i}** (length {len(chain)-1}):\n"
        md += f"  {path_str}\n"
        md += f"  Relations: {', '.join(rels)}\n\n"

    return md


path_find_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "path_find",
            "description": "Find paths between two entities (1-N hops) showing entity chains and relation types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_entity": {
                        "type": "string",
                        "description": "Start entity name."
                    },
                    "end_entity": {
                        "type": "string",
                        "description": "End entity name."
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Max hops (default 3)."
                    },
                    "max_paths": {
                        "type": "integer",
                        "description": "Max paths (default 5)."
                    }
                },
                "required": ["start_entity", "end_entity"]
            }
        }
    }
]

path_find_tools_map = {
    "path_find": path_find
}
