"""
cogops/tools/registry.py

Build the full tool registry: tools_schema + name-to-callback map.
Each tool group registers its own schemas and mappers.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def build_tool_registry(
    secondary_client=None,
    secondary_model: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, callable]]:
    """
    Build the complete tool registry from all available tool groups.

    Returns (tools_schema, name_to_callable).
    """
    all_schema = []
    all_map = {}

    # --- Graph tools ---
    from cogops.tools.graph.search import graph_search_tools_list as g1, graph_search_tools_map as g2
    from cogops.tools.graph.entity_search import entity_search_tools_list as g3, entity_search_tools_map as g4
    from cogops.tools.graph.entity_detail import entity_detail_tools_list as g5, entity_detail_tools_map as g6
    from cogops.tools.graph.node_explore import node_explore_tools_list as g7, node_explore_tools_map as g8
    from cogops.tools.graph.relation_browse import relation_browse_tools_list as g9, relation_browse_tools_map as g10
    from cogops.tools.graph.relation_filter import relation_filter_tools_list as g11, relation_filter_tools_map as g12
    from cogops.tools.graph.similar_entities import similar_entities_tools_list as g13, similar_entities_tools_map as g14
    from cogops.tools.graph.path_find import path_find_tools_list as g15, path_find_tools_map as g16
    from cogops.tools.graph.episodic_search import episodic_search_tools_list as g17, episodic_search_tools_map as g18
    from cogops.tools.graph.graph_stats import graph_stats_tools_list as g19, graph_stats_tools_map as g20

    for s, m in [(g1, g2), (g3, g4), (g5, g6), (g7, g8), (g9, g10),
                 (g11, g12), (g13, g14), (g15, g16), (g17, g18), (g19, g20)]:
        all_schema.extend(s)
        all_map.update(m)

    # --- Secondary-LLM tools ---
    from cogops.tools.secondary.grep_passage import grep_passage_tools_list as s1, grep_passage_tools_map as s2
    from cogops.tools.secondary.extract_from_doc import extract_tools_list as s3, extract_tools_map as s4
    from cogops.tools.secondary.delegate_task import delegate_tools_list as s5, delegate_tools_map as s6
    from cogops.tools.secondary.spawn_subagent import spawn_subagent_tools_list as s7, spawn_subagent_tools_map as s8

    for s, m in [(s1, s2), (s3, s4), (s5, s6), (s7, s8)]:
        all_schema.extend(s)
        all_map.update(m)

    # --- Interaction tools ---
    from cogops.tools.ask_user import ask_user_tools_list as i1, ask_user_tools_map as i2
    for s, m in [(i1, i2)]:
        all_schema.extend(s)
        all_map.update(m)

    # --- History tool ---
    from cogops.tools.history.query import history_query_tools_list as h1, history_query_tools_map as h2
    all_schema.extend(h1)
    all_map.update(h2)

    logger.info(f"Tool registry built: {len(all_schema)} tools, {len(all_map)} entries.")
    return all_schema, all_map


def get_tool_names(tools_schema: List[Dict[str, Any]]) -> List[str]:
    """Extract tool names from schema list."""
    return [t["function"]["name"] for t in tools_schema]
