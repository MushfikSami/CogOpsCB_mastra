"""
cogops/tools/graph/_helpers.py

Shared helpers for graph tools: resolve_entity, markdown formatters.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


async def resolve_entity(driver, name: str):
    """
    Find an entity by exact name first, then case-insensitive exact, then partial.
    Returns {uuid, name, summary, group_id} or list of matches or None.
    """
    from graphiti_core.driver.driver import GraphProvider

    # Try exact match first
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE e.name = $name RETURN e.uuid, e.name, e.summary, e.group_id LIMIT 1",
            name=name
        )
        record = await result.single()
        if record:
            return dict(record)

    # Try partial match
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) "
            "RETURN e.uuid, e.name, e.summary, e.group_id "
            "ORDER BY CASE WHEN toLower(e.name) = toLower($name) THEN 0 WHEN toLower(e.name) CONTAINS toLower($name) THEN 1 ELSE 2 END LIMIT 5",
            name=name
        )
        records = await result.data()
        if records:
            return records

    return None
