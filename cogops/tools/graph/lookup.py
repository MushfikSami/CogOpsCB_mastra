"""
cogops/tools/graph/lookup.py

get_by_uuid: unified tool to fetch full details of any graph entity/episode/edge
by UUID. Returns structured Markdown with all available properties.

Types:
  - entity  → Entity node (name, summary, all properties)
  - episode → Episodic node (parsed JSON content with category, topic, service, text)
  - edge    → RELATES_TO edge (relation type, fact, source/target entities)
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _render_entity(entity_data: Dict[str, Any]) -> str:
    """Render an Entity node as Markdown."""
    lines: List[str] = []
    lines.append(f"## Entity: {entity_data.get('name', 'Unknown')}")
    lines.append(f"**UUID:** `{entity_data.get('uuid', '')}`")

    if entity_data.get("summary"):
        lines.append(f"**Summary:** {entity_data['summary']}")

    # Show all remaining properties
    skip = {"name", "uuid", "summary", "group_id", "created_at", "labels"}
    props = {k: v for k, v in entity_data.items() if k not in skip}
    if props:
        lines.append("")
        lines.append("**Properties:**")
        for k, v in props.items():
            val = str(v)[:200] if v else ""
            lines.append(f"  - **{k}:** {val}")

    # Fetch edges
    edges = entity_data.get("_edges", [])
    if edges:
        lines.append("")
        lines.append("**Relations:**")
        # Group by relation type
        grouped: Dict[str, List] = {}
        for e in edges:
            rt = e.get("rel_type", "unnamed")
            grouped.setdefault(rt, []).append(e)
        for rt in sorted(grouped.keys()):
            rels = grouped[rt]
            lines.append(f"  **{rt}** ({len(rels)}):")
            for rel in rels[:10]:  # cap display
                target = rel.get("target_name", "unknown")
                target_uuid = rel.get("target_uuid", "")
                fact = (rel.get("fact", "") or "")[:150]
                fact_display = fact.replace("\n", " ")
                lines.append(f"    - {fact_display} → [{target}]({target_uuid})")
            if len(rels) > 10:
                lines.append(f"    ... and {len(rels) - 10} more")

    return "\n".join(lines)


def _render_episode(episode_data: Dict[str, Any]) -> str:
    """Render an Episodic node as Markdown."""
    content = episode_data.get("_parsed", {})
    lines: List[str] = []
    lines.append(f"## Episode: {episode_data.get('uuid', '')}")

    # Metadata
    metadata_keys = ["category", "sub_category", "service", "topic", "passage_id"]
    lines.append("")
    for key in metadata_keys:
        val = content.get(key, "")
        if val:
            lines.append(f"**{key}:** {val}")

    # Full text
    text = content.get("text", "")
    if text:
        lines.append("")
        if len(text) > 500:
            lines.append(f"**Content:** {text[:500]}...")
        else:
            lines.append(f"**Content:** {text}")

    return "\n".join(lines)


def _render_edge(edge_data: Dict[str, Any]) -> str:
    """Render a RELATES_TO edge as Markdown."""
    lines: List[str] = []
    lines.append(f"## Edge: {edge_data.get('rel_type', 'unnamed')}")
    lines.append(f"**UUID:** `{edge_data.get('edge_uuid', '')}`")

    fact = edge_data.get("fact", "")
    if fact:
        lines.append(f"**Fact:** {fact}")

    source = edge_data.get("source_name", "")
    source_uuid = edge_data.get("source_uuid", "")
    if source:
        lines.append(f"**Source:** [{source}]({source_uuid})")

    target = edge_data.get("target_name", "")
    target_uuid = edge_data.get("target_uuid", "")
    if target:
        lines.append(f"**Target:** [{target}]({target_uuid})")

    episodes = edge_data.get("_episodes", [])
    if episodes:
        lines.append("")
        lines.append("**Episodes:**")
        for ep in episodes:
            title = ep.get("topic", ep.get("category", ep.get("snippet", "")[:40]))
            lines.append(f"  - [{ep.get('uuid', '')}] {title}")

    return "\n".join(lines)


async def get_by_uuid(uuid: str, entity_type: str = "episode") -> str:
    """
    Fetch full details of any graph node or edge by UUID.

    Args:
        uuid: The UUID to look up.
        entity_type: One of 'entity', 'episode', 'edge'.

    Returns:
        Markdown string with all details.
    """
    from cogops.graph.client import get_graphiti_client

    client = await get_graphiti_client()
    driver = client.driver

    if entity_type == "entity":
        return await _lookup_entity(driver, uuid)
    elif entity_type == "episode":
        return await _lookup_episode(driver, uuid)
    elif entity_type == "edge":
        return await _lookup_edge(driver, uuid)
    else:
        return f"Unknown entity type: {entity_type}. Must be one of: entity, episode, edge."


async def _lookup_entity(driver, uuid: str) -> str:
    """Look up an Entity node by UUID and return Markdown."""
    query = (
        "MATCH (e:Entity) WHERE e.uuid = $uuid "
        "RETURN e.uuid AS uuid, e.name AS name, e.summary AS summary, "
        "e.group_id AS group_id, e.created_at AS created_at, e.labels AS labels "
        "LIMIT 1"
    )
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(query, uuid=uuid)
        records = await result.data()

    if not records:
        return f"Entity not found: {uuid}"

    row = records[0]
    entity_data: Dict[str, Any] = {
        "uuid": row.get("uuid"),
        "name": row.get("name"),
        "summary": row.get("summary"),
        "group_id": row.get("group_id"),
        "created_at": row.get("created_at"),
        "labels": row.get("labels"),
    }

    # Fetch relations
    edges_query = (
        "MATCH (e:Entity {uuid: $uuid})-[r:RELATES_TO]-(neighbor:Entity) "
        "RETURN r.uuid AS edge_uuid, r.name AS rel_type, r.fact AS fact, "
        "r.episodes AS edge_episodes, "
        "neighbor.uuid AS target_uuid, neighbor.name AS target_name, "
        "neighbor.summary AS target_summary "
        "LIMIT 50"
    )
    async with driver.session(database="qwen34neo4j") as session:
        res = await session.run(edges_query, uuid=uuid)
        edge_records = await res.data()

    entity_data["_edges"] = []
    for er in edge_records:
        entity_data["_edges"].append({
            "edge_uuid": er.get("edge_uuid"),
            "rel_type": er.get("rel_type"),
            "fact": er.get("fact", ""),
            "target_uuid": er.get("target_uuid"),
            "target_name": er.get("target_name"),
            "target_summary": er.get("target_summary"),
        })

    return _render_entity(entity_data)


async def _lookup_episode(driver, uuid: str) -> str:
    """Look up an Episodic node by UUID and return Markdown."""
    query = (
        "MATCH (ep:Episodic) WHERE ep.uuid = $uuid "
        "RETURN ep.uuid AS uuid, ep.content AS content, "
        "ep.source AS source, ep.source_description AS source_description, "
        "ep.created_at AS created_at, "
        "ep.entity_edges AS entity_edges "
        "LIMIT 1"
    )
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(query, uuid=uuid)
        records = await result.data()

    if not records:
        return f"Episode not found: {uuid}"

    row = records[0]
    content = row.get("content", "")
    parsed: Dict[str, Any] = {}
    try:
        if isinstance(content, str):
            parsed = json.loads(content)
        else:
            parsed = content
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw": str(content)}

    episode_data = {
        "uuid": row.get("uuid"),
        "_parsed": parsed,
        "source": row.get("source"),
        "created_at": row.get("created_at"),
        "_episodes": [],  # episodes on this — usually itself
    }

    return _render_episode(episode_data)


async def _lookup_edge(driver, uuid: str) -> str:
    """Look up a RELATES_TO edge by UUID and return Markdown."""
    query = (
        "MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity) "
        "WHERE r.uuid = $uuid "
        "RETURN r.uuid AS edge_uuid, r.name AS rel_type, r.fact AS fact, "
        "r.episodes AS edge_episodes, "
        "source.uuid AS source_uuid, source.name AS source_name, "
        "target.uuid AS target_uuid, target.name AS target_name, "
        "target.summary AS target_summary "
        "LIMIT 1"
    )
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(query, uuid=uuid)
        records = await result.data()

    if not records:
        return f"Edge not found: {uuid}"

    row = records[0]
    edge_data: Dict[str, Any] = {
        "edge_uuid": row.get("edge_uuid"),
        "rel_type": row.get("rel_type"),
        "fact": row.get("fact", ""),
        "source_uuid": row.get("source_uuid"),
        "source_name": row.get("source_name"),
        "target_uuid": row.get("target_uuid"),
        "target_name": row.get("target_name"),
        "target_summary": row.get("target_summary"),
    }

    # Fetch episodes on this edge
    episodes = row.get("edge_episodes") or []
    if episodes:
        unique_eps = list(set(str(ep) for ep in episodes if ep))[:20]
        if unique_eps:
            ep_query = (
                "UNWIND $uuids AS ep_uuid "
                "MATCH (ep:Episodic) WHERE ep.uuid = ep_uuid "
                "RETURN ep.uuid AS uuid, ep.content AS content"
            )
            async with driver.session(database="qwen34neo4j") as session:
                res = await session.run(ep_query, uuids=unique_eps)
                ep_records = await res.data()

            edge_data["_episodes"] = []
            for ep_row in ep_records:
                try:
                    ep_content = json.loads(ep_row["content"]) if isinstance(ep_row["content"], str) else ep_row["content"]
                    edge_data["_episodes"].append({
                        "uuid": ep_row["uuid"],
                        "category": ep_content.get("category", ""),
                        "topic": ep_content.get("topic", ""),
                        "snippet": (ep_content.get("text", "") or "")[:60],
                    })
                except (json.JSONDecodeError, TypeError):
                    pass

    return _render_edge(edge_data)


# ── Tool Schema & Mapping ─────────────────────────────────────────────

get_by_uuid_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "get_by_uuid",
            "description": "Look up full details of any graph node or edge by UUID. Use when the user provides or the tree output contains a UUID to inspect in detail. Accepts UUIDs for entities, episodic passages, or edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uuid": {
                        "type": "string",
                        "description": "The UUID to look up (e.g., from tree navigation IDs or entity detail output).",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["entity", "episode", "edge"],
                        "description": "The type of graph element. Default: entity.",
                    },
                },
                "required": ["uuid"],
            },
        },
    }
]

get_by_uuid_tools_map = {
    "get_by_uuid": get_by_uuid,
}
