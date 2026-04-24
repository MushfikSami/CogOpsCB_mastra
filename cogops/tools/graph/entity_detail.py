"""
cogops/tools/graph/entity_detail.py

Get full details of a specific entity by exact name or UUID.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def entity_detail(identifier: str) -> str:
    """Get full details of a specific entity by exact name or UUID."""
    from cogops.graph.client import get_graphiti_client
    client = await get_graphiti_client()
    driver = client.driver

    md = f"## Entity Details for '{identifier}'\n\n"

    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE e.name = $id OR e.uuid = $id "
            "RETURN e.uuid, e.name, e.summary, e.group_id, e.created_at",
            id=identifier
        )
        record = await result.single()

    if not record:
        md += f"Entity not found: '{identifier}'\n"
        return md

    md += f"**UUID:** `{record['e.uuid']}`\n"
    md += f"**Name:** {record['e.name']}\n"
    md += f"**Summary:** {record['e.summary'] or '(no summary)'}\n"
    md += f"**Group ID:** {record.get('e.group_id', 'N/A')}\n"
    md += f"**Created:** {record.get('e.created_at', 'N/A')}\n"

    return md


entity_detail_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "entity_detail",
            "description": "Get full details of a specific entity by exact name or UUID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Entity name or UUID."
                    }
                },
                "required": ["identifier"]
            }
        }
    }
]

entity_detail_tools_map = {
    "entity_detail": entity_detail
}
