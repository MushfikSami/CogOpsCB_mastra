"""
cogops/tools/graph/tree_explorer.py

tree_explorer: query-aware graph tree builder.
Uses Graphiti hybrid search for broad retrieval, then applies deep semantic
reranking via QwenRerankerClient to perfectly prune irrelevant branches.

Returns episode summaries (not full text) with UUIDs for the LLM to call
deeper tools later. Two APIs:
- async def tree_explorer(query)  — for the pipeline/reasoning loop
- def tree_explorer_sync(query)   — for Jupyter notebooks (no async)
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from cogops.config.loader import load_config, get_tool_config
from cogops.graph.client import get_graphiti_client
from cogops.llm.reranker import QwenRerankerClient
from graphiti_core.llm_client.config import LLMConfig

load_dotenv()
CONFIG = load_config()

logger = logging.getLogger(__name__)

# ── Episode Parsing ────────────────────────────────────────────────────

def _parse_episode_content(content_str: str) -> Optional[Dict[str, Any]]:
    """Parse Episodic JSON content into a summary dict."""
    try:
        if isinstance(content_str, str):
            parsed = json.loads(content_str)
        else:
            parsed = content_str

        text = parsed.get("text", "")
        snippet = text[:80] if text else ""

        return {
            "episode_id": parsed.get("passage_id", ""),
            "passage_id": parsed.get("passage_id", ""),
            "category": parsed.get("category", ""),
            "topic": parsed.get("topic", ""),
            "service": parsed.get("service", ""),
            "sub_category": parsed.get("sub_category", ""),
            "snippet": snippet,
        }
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


# ── Graph Data Fetching ────────────────────────────────────────────────

async def _fetch_entity_edges(
    driver, entity_uuid: str, max_edges: int = 50
) -> List[Dict[str, Any]]:
    """Fetch all edges + neighbors for a given entity UUID."""
    query = (
        "MATCH (e:Entity {uuid: $uuid})-[r:RELATES_TO]-(neighbor:Entity) "
        "RETURN r.uuid AS edge_uuid, r.name AS rel_type, r.fact AS fact, "
        "r.episodes AS edge_episodes, "
        "neighbor.uuid AS neighbor_uuid, neighbor.name AS neighbor_name, "
        "neighbor.summary AS neighbor_summary "
        "LIMIT $max"
    )
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(query, uuid=entity_uuid, max=max_edges)
        return await result.data()


async def _fetch_episode_summaries(
    driver, episode_uuids: List[str], limit: int = 200
) -> Dict[str, Dict[str, Any]]:
    """Batch-fetch episode summaries by UUIDs."""
    if not episode_uuids:
        return {}

    unique_uuids = list(set(episode_uuids))[:limit]
    if not unique_uuids:
        return {}

    query = (
        "UNWIND $uuids AS ep_uuid "
        "MATCH (ep:Episodic) WHERE ep.uuid = ep_uuid "
        "RETURN ep.uuid AS episode_uuid, ep.content AS content"
    )
    async with driver.session(database="qwen34neo4j") as session:
        result = await session.run(query, uuids=unique_uuids)
        summaries = {}
        for row in await result.data():
            parsed = _parse_episode_content(row["content"])
            if parsed:
                parsed["episode_id"] = row["episode_uuid"]
                summaries[row["episode_uuid"]] = parsed
        return summaries


# ── Async: tree_explorer ───────────────────────────────────────────────

async def tree_explorer(query: str) -> str:
    """Build a query-aware graph tree.

    Uses Graphiti hybrid search for high recall, then uses Qwen deep
    semantic reranking to strictly filter paths by true relevance.
    """
    cfg = get_tool_config(CONFIG, "tree_explorer")
    min_score = cfg.get("min_score", 0.50)  # Qwen softmax threshold
    keep_top_n = cfg.get("keep_top_n", 15)  # More initial candidates since AI prunes perfectly
    max_edges_per_entity = cfg.get("max_edges_per_entity", 30)
    
    # Initialize Qwen Reranker using your custom .env variables
    reranker_base_url = os.getenv("RERANKER_BASE_URL")
    reranker_api_key = os.getenv("RERANKER_API_KEY")
    reranker_model = os.getenv("RERANKER_MODEL_NAME", "qwen36")

    # Pass the custom URL and Key to the OpenAI client
    aclient = AsyncOpenAI(
        base_url=reranker_base_url,
        api_key=reranker_api_key
    )
    
    llm_config = LLMConfig(model=reranker_model)
    reranker = QwenRerankerClient(client=aclient, config=llm_config)
    client = await get_graphiti_client()
    driver = client.driver

    # STEP 1: Broad Graphiti Search (High Recall)
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
    from graphiti_core.search.search_config import NodeReranker

    search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    search_config.node_config.reranker = NodeReranker.rrf
    results = await client._search(query=query, config=search_config)

    tree_data: Dict[str, Any] = {
        "query": query,
        "entities":[],
        "total_edges": 0,
        "total_episodes": 0,
    }

    if not results.nodes:
        return _render_tree(tree_data)

    # STEP 2: Deep Semantic Reranking of Nodes
    node_candidates = []
    node_passages =[]
    
    for node in results.nodes[:keep_top_n]:
        # Build passage for the AI to read
        passage = f"Entity Name: {node.name}\nEntity Summary: {node.summary or ''}"
        node_candidates.append({"node": node, "passage": passage})
        node_passages.append(passage)

    # Score nodes concurrently
    ranked_nodes_list = await reranker.rank(query, node_passages)
    passage_to_score = {passage: score for passage, score in ranked_nodes_list}

    root_entities =[]
    for candidate in node_candidates:
        score = passage_to_score.get(candidate["passage"], 0.0)
        if score >= min_score:
            root_entities.append({
                "node": candidate["node"],
                "relevance_score": score
            })

    # Fallback if AI rejects everything (to prevent crashing the agent)
    if not root_entities and ranked_nodes_list:
        top_passage, top_score = ranked_nodes_list[0]
        for candidate in node_candidates:
            if candidate["passage"] == top_passage:
                root_entities.append({"node": candidate["node"], "relevance_score": top_score})
                break

    # STEP 3 & 4: Fetch Edges & Deep Semantic Pruning of Branches
    all_edge_uuids =[]
    all_episode_uuids = []

    for root_info in root_entities:
        root = root_info["node"]
        edges = await _fetch_entity_edges(driver, root.uuid, max_edges=max_edges_per_entity)

        edge_candidates = []
        edge_passages =[]
        
        for edge_row in edges:
            rel_type = edge_row.get("rel_type", "")
            fact = edge_row.get("fact", "")
            neighbor_name = edge_row.get("neighbor_name", "")
            
            # Context string for Qwen
            passage = f"Relation: {rel_type}\nFact: {fact}\nTarget Entity: {neighbor_name}"
            edge_candidates.append({"edge_row": edge_row, "passage": passage})
            edge_passages.append(passage)

        # Score edges concurrently
        ranked_edges_list = await reranker.rank(query, edge_passages)
        edge_passage_to_score = {passage: score for passage, score in ranked_edges_list}

        grouped_edges: Dict[str, List] = {}
        
        # Keep only semantically relevant edges
        for candidate in edge_candidates:
            score = edge_passage_to_score.get(candidate["passage"], 0.0)
            
            # Prune branches irrelevant to query
            if score >= min_score:
                er = candidate["edge_row"]
                rt = er.get("rel_type") or "(unnamed)"
                
                # Track episodes for summaries
                episodes = er.get("edge_episodes") or[]
                for ep in episodes:
                    if ep and isinstance(ep, str):
                        all_episode_uuids.append(ep)
                        all_edge_uuids.append(ep)
                if er.get("neighbor_uuid"):
                    all_episode_uuids.append(er["neighbor_uuid"])
                
                candidate["relevance_score"] = score
                grouped_edges.setdefault(rt,[]).append(candidate)

        entity_entry = {
            "name": root.name,
            "uuid": root.uuid,
            "summary": root.summary or "",
            "relevance_score": root_info["relevance_score"],
            "relations": {},
        }

        for rel_type, rel_edges in sorted(grouped_edges.items()):
            edges_list =[]
            for re in rel_edges:
                er = re["edge_row"]
                episode_ids = list(set(er.get("edge_episodes") or []))
                episode_ids =[ep for ep in episode_ids if ep]
                edges_list.append({
                    "edge_uuid": er.get("edge_uuid"),
                    "rel_type": er.get("rel_type"),
                    "fact": er.get("fact", ""),
                    "neighbor_name": er.get("neighbor_name"),
                    "neighbor_uuid": er.get("neighbor_uuid"),
                    "neighbor_summary": er.get("neighbor_summary", ""),
                    "neighbor_relevance": re["relevance_score"],
                    "episode_ids": episode_ids,
                })
            entity_entry["relations"][rel_type] = edges_list
            tree_data["total_edges"] += len(edges_list)

        tree_data["entities"].append(entity_entry)

    # STEP 5: Render episodes and Markdown
    episode_summaries = await _fetch_episode_summaries(driver, list(set(all_episode_uuids)))
    tree_data["total_episodes"] = len(episode_summaries)
    tree_data["episode_summaries"] = episode_summaries

    return _render_tree(tree_data)


# ── Sync wrapper ───────────────────────────────────────────────────────

def tree_explorer_sync(query: str) -> str:
    """Synchronous wrapper for tree_explorer — use in Jupyter notebooks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(tree_explorer(query))
    finally:
        loop.close()


# ── Markdown Rendering ─────────────────────────────────────────────────

def _render_tree(tree_data: Dict[str, Any]) -> str:
    """Convert structured tree data to Markdown using token-efficient tables."""
    md_lines: List[str] =[]

    query = tree_data.get("query", "")
    md_lines.append(f"## Tree Explorer: \"{query}\"")
    md_lines.append(
        f'*Query-relevant entities: {len(tree_data["entities"])} '
        f'| Depth explored: 1 | Edges: {tree_data["total_edges"]} '
        f'| Episodes: {tree_data["total_episodes"]}*\n'
    )

    entity_ids = []
    edge_ids = []
    episode_ids =[]

    for entity in tree_data["entities"]:
        md_lines.append("---")
        # Put the Node ID in the title to save lines
        md_lines.append(f"### Entity: {entity['name']} (ID: `{entity['uuid']}`)")
        
        summary = entity["summary"]
        if summary:
            # Truncate summary and remove linebreaks for compactness
            if len(summary) > 150:
                summary = summary[:150] + "..."
            md_lines.append(f"**Summary:** {summary.replace(chr(10), ' ')}\n")

        entity_ids.append(entity["uuid"])

        relations = entity.get("relations", {})
        if not relations:
            md_lines.append("*(No query-relevant edges found for this entity)*\n")
            continue

        # --- Table Header ---
        md_lines.append("| Relation | Fact | Target Entity (ID) | Edge ID | Episodes (ID: Topic) |")
        md_lines.append("|---|---|---|---|---|")

        for rel_type in sorted(relations.keys()):
            edges_list = relations[rel_type]

            for edge in edges_list:
                edge_uuid = edge.get("edge_uuid", "")
                fact = edge.get("fact", "")
                neighbor_name = edge.get("neighbor_name", "")
                neighbor_uuid = edge.get("neighbor_uuid", "")
                episode_ids_list = edge.get("episode_ids",[])

                if edge_uuid:
                    edge_ids.append(edge_uuid)

                # Clean fact to prevent table breaking (remove newlines and pipes)
                fact_display = fact.replace("\n", " ").replace("|", "/")
                if len(fact_display) > 120:
                    fact_display = fact_display[:120] + "..."

                # Format Target Entity & ID in one cell
                neighbor_display = neighbor_name.replace("\n", " ").replace("|", "/")
                target_cell = f"{neighbor_display}<br>`{neighbor_uuid}`" if neighbor_uuid else neighbor_display

                # Format Episodes
                ep_cells =[]
                episode_ids.extend(episode_ids_list)
                for ep_id in episode_ids_list:
                    ep_summary = tree_data.get("episode_summaries", {}).get(ep_id)
                    if ep_summary:
                        cat = ep_summary.get("category", "")
                        top = ep_summary.get("topic", "")
                        title = "/".join(filter(None,[cat, top]))
                        if not title:
                            title = ep_summary.get("snippet", "")[:30]
                        
                        title = title.replace("\n", " ").replace("|", "/")
                        ep_cells.append(f"`{ep_id}`: {title}")
                    else:
                        ep_cells.append(f"`{ep_id}`")
                
                # Join multiple episodes with HTML break so they stay in one table cell
                episodes_cell = "<br>".join(ep_cells) if ep_cells else "-"
                
                rel_cell = rel_type.replace("\n", " ").replace("|", "/")
                edge_id_cell = f"`{edge_uuid}`" if edge_uuid else "-"

                # Append Row
                md_lines.append(f"| {rel_cell} | {fact_display} | {target_cell} | {edge_id_cell} | {episodes_cell} |")
        
        md_lines.append("")

    # Deduplicate Navigation IDs
    entity_ids = list(set(entity_ids))
    edge_ids = list(set(edge_ids))
    episode_ids = list(set(episode_ids))

    md_lines.append("---")
    md_lines.append("*Tree Navigation IDs for the LLM to call deeper tools:*")
    md_lines.append(f"**Entities:** {entity_ids}")
    md_lines.append(f"**Edges:** {edge_ids}")
    md_lines.append(f"**Episodes:** {episode_ids}")

    return "\n".join(md_lines)

# ── Tool Schema & Mapping ─────────────────────────────────────────────

tree_explorer_tools_list =[
    {
        "type": "function",
        "function": {
            "name": "tree_explorer",
            "description": "Build a query-aware hierarchical information tree from the Bangladesh Government Knowledge Graph. Returns entities, edges (with relation types), and episode summaries relevant to the query. Use when the user asks about any government service, procedure, fee, document requirement, or process — especially when the answer requires understanding relationships between entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's full query about government services, processes, fees, documents, or procedures.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

tree_explorer_tools_map = {
    "tree_explorer": tree_explorer,
}