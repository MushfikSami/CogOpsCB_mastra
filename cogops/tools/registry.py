"""
cogops/tools/registry.py

Build the full tool registry: tools_schema + name-to-callback map.

Tools fall into two groups:
- pure tools whose only inputs come from the model (handled as-is)
- context-dependent tools whose handler needs server-side state
  (user_id, Redis store, secondary LLM client/model, full tool_map for
  spawn_subagent). The model-visible JSON schema never exposes those;
  they are bound per-request via bind_tools(ctx).
"""

import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Parameters that must be injected server-side, never surfaced to the model.
_INJECTABLE_PARAMS = (
    "user_id",
    "store",
    "secondary_client",
    "secondary_model",
    "tool_map",
    "tools_schema",
)


@dataclass
class ToolContext:
    """Per-request / per-init state that context-dependent tools need.

    `user_id` is per-request (so bind_tools must run per process_query call).
    Everything else can be bound once at orchestrator init.
    """
    user_id: Optional[str] = None
    store: Optional[Any] = None            # RedisSessionStore
    secondary_client: Optional[Any] = None
    secondary_model: str = ""
    tool_map: Optional[Dict[str, Callable]] = None
    tools_schema: Optional[List[Dict[str, Any]]] = None


def build_tool_registry() -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Build the raw tool registry. Returned handlers are UNBOUND — callers must
    pass through `bind_tools(raw_map, ctx)` before dispatching.
    """
    all_schema: List[Dict[str, Any]] = []
    all_map: Dict[str, Callable] = {}

    # --- Graph tools (pure, no context injection) ---
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

    # --- Wiki fallback tools (pure, no context injection) ---
    from cogops.tools.wiki import (
        wikipedia_tools_list as w1,
        wikipedia_tools_map as w2,
    )
    all_schema.extend(w1)
    all_map.update(w2)

    # --- Secondary-LLM tools (need secondary_client/secondary_model) ---
    from cogops.tools.secondary.grep_passage import grep_passage_tools_list as s1, grep_passage_tools_map as s2
    from cogops.tools.secondary.extract_from_doc import extract_tools_list as s3, extract_tools_map as s4
    from cogops.tools.secondary.delegate_task import delegate_tools_list as s5, delegate_tools_map as s6
    from cogops.tools.secondary.spawn_subagent import spawn_subagent_tools_list as s7, spawn_subagent_tools_map as s8

    for s, m in [(s1, s2), (s3, s4), (s5, s6), (s7, s8)]:
        all_schema.extend(s)
        all_map.update(m)

    # --- Interaction tools ---
    from cogops.tools.ask_user import ask_user_tools_list as i1, ask_user_tools_map as i2
    all_schema.extend(i1)
    all_map.update(i2)

    from cogops.tools.answer_directly import (
        answer_directly_tools_list as a1,
        answer_directly_tools_map as a2,
    )
    all_schema.extend(a1)
    all_map.update(a2)

    # --- History tool (needs user_id + store + secondary for 'ask' mode) ---
    from cogops.tools.history.query import history_query_tools_list as h1, history_query_tools_map as h2
    all_schema.extend(h1)
    all_map.update(h2)

    # Enforce additionalProperties: false on every schema so the vLLM tool
    # parser rejects extra model-invented keys cleanly.
    for spec in all_schema:
        params = spec.get("function", {}).get("parameters", {})
        params.setdefault("additionalProperties", False)

    logger.info(f"Tool registry built: {len(all_schema)} tools, {len(all_map)} entries.")
    return all_schema, all_map


def bind_tools(
    raw_tool_map: Dict[str, Callable],
    ctx: ToolContext,
) -> Dict[str, Callable]:
    """
    Wrap each handler with functools.partial so context-dependent tools get
    their server-side inputs. The model-visible parameters remain unbound
    and are supplied at call time from parsed JSON.
    """
    ctx_dict = {
        "user_id": ctx.user_id,
        "store": ctx.store,
        "secondary_client": ctx.secondary_client,
        "secondary_model": ctx.secondary_model,
        "tool_map": ctx.tool_map,
        "tools_schema": ctx.tools_schema,
    }

    bound: Dict[str, Callable] = {}
    for name, fn in raw_tool_map.items():
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            bound[name] = fn
            continue

        inject = {
            k: ctx_dict[k]
            for k in _INJECTABLE_PARAMS
            if k in sig.parameters and ctx_dict.get(k) is not None
        }
        bound[name] = partial(fn, **inject) if inject else fn
    return bound


def get_tool_names(tools_schema: List[Dict[str, Any]]) -> List[str]:
    """Extract tool names from schema list."""
    return [t["function"]["name"] for t in tools_schema]
