"""
cogops/tools/graph/search.py

graph_search: hybrid search (BM25 + vector + BFS) with cross_encoder reranking.
Moved from cogops/tools/graphiti_tools.py (the function + schema part).
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from ast import literal_eval
from dotenv import load_dotenv
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER

from cogops.config.loader import load_config
from cogops.graph.client import get_graphiti_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = load_config()


async def graph_search(query: str) -> str:
    """
    Searches the Government Knowledge Graph for relevant facts, regulations, and procedures.

    Args:
        query (str): The specific search query (e.g., "passport renewal fee", "birth registration process").

    Returns:
        str: A formatted text summary of the findings.
    """
    client = await get_graphiti_client()
    search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    search_config_params = CONFIG.get('graph_search', {})
    limit = search_config_params.get('limit', 5)
    reranker_thresh = search_config_params.get('min_score', '0.9')

    logger.info(f"Executing Graph Search: '{query}' (Limit: {limit})")

    try:
        results = await client._search(
            query=query, config=search_config
        )

        md_content = ""

        # Nodes Section
        md_content += "\n## Nodes\n"
        node_summaries = []
        for node, score in zip(results.nodes, results.node_reranker_scores):
            if score > reranker_thresh:
                node_summaries.append(f"**{node.name}**:{node.summary}")
        if node_summaries:
            md_content += "- " + "\n- ".join(node_summaries[:limit]) + "\n\n"
        else:
            md_content += "No relevant nodes found.\n\n"

        # Edges Section
        md_content += "## Edges\n"
        edge_facts = []
        for edge, score in zip(results.edges, results.edge_reranker_scores):
            if score > reranker_thresh:
                edge_facts.append(edge.fact)
        if edge_facts:
            md_content += "- " + "\n- ".join(edge_facts[:limit]) + "\n\n"
        else:
            md_content += "No edges found.\n\n"

        # Episodes Section
        md_content += "## Passages\n"
        episode_data = []
        for episode, score in zip(results.episodes, results.episode_reranker_scores):
            if score > reranker_thresh:
                json_episode = literal_eval(episode.content)
                passage = json_episode["text"].split("Category")[0].strip()
                context = json_episode["text"].split("Category")[-1].strip()
                url = json_episode.get("url", "")
                text = f"# passage_context:\n {context}\n\n # passage_text:\n{passage} \n # Sources:\n{url}"
                episode_data.append(text)
        if episode_data:
            md_content += "- " + "\n- ".join(episode_data[:limit]) + "\n\n"
        else:
            md_content += "No Passages found.\n\n"

        return md_content
    except Exception as e:
        logger.error(f"Error during graph search: {e}", exc_info=True)
        return f"System Error: Unable to retrieve data due to {str(e)}"


# --- The Tool Schema & Mapping ---

graph_search_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Search the official Bangladesh Government Knowledge Graph. Use this tool whenever the user asks about any information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific topic to search for (e.g., 'driving license fee', 'NID correction documents')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

graph_search_tools_map = {
    "graph_search": graph_search
}
