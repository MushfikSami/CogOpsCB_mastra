"""
cogops/tools/registry.py

Build the tool registry: JSON schemas + name-to-callback map.
Context-dependent tools (needing user_id, Redis store, etc.) are
bound per-request via bind_tools(ctx).
"""

import inspect
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Parameters injected server-side, never surfaced to the model.
_INJECTABLE_PARAMS = (
    "user_id",
    "store",
    "secondary_client",
    "secondary_model",
)


@dataclass
class ToolContext:
    """Per-request state that context-dependent tools need."""
    user_id: Optional[str] = None
    store: Optional[Any] = None            # RedisSessionStore
    secondary_client: Optional[Any] = None
    secondary_model: str = ""
    tool_map: Optional[Any] = None         # Raw tool map (for history_query 'ask' mode)
    tools_schema: Optional[Any] = None     # JSON schemas (for nested tool calls)


def build_tool_registry() -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Build the raw tool registry. Returned handlers are UNBOUND.
    Callers must pass through bind_tools(raw_map, ctx).
    """
    all_schema: List[Dict[str, Any]] = []
    all_map: Dict[str, Callable] = {}

    # Knowledge search (Jiggasha RAG)
    from cogops.tools.search_jiggasha import (
        search_knowledge_tools_list as k1,
        search_knowledge_tools_map as k2,
    )
    all_schema.extend(k1)
    all_map.update(k2)

    # Wiki search
    from cogops.tools.search_wiki import (
        search_wiki_tools_list as w1,
        search_wiki_tools_map as w2,
    )
    all_schema.extend(w1)
    all_map.update(w2)

    # History query (needs user_id + store + secondary)
    from cogops.tools.search_history import (
        history_query_tools_list as h1,
        history_query_tools_map as h2,
    )
    all_schema.extend(h1)
    all_map.update(h2)

    # Enforce additionalProperties: false on every schema
    for spec in all_schema:
        params = spec.get("function", {}).get("parameters", {})
        params.setdefault("additionalProperties", False)

    logger.info("Tool registry built: %d tools, %d entries.", len(all_schema), len(all_map))
    return all_schema, all_map


def bind_tools(
    raw_tool_map: Dict[str, Callable],
    ctx: ToolContext,
) -> Dict[str, Callable]:
    """
    Wrap each handler with functools.partial so context-dependent tools get
    their server-side inputs. Model-visible parameters remain unbound.
    """
    ctx_dict = {
        "user_id": ctx.user_id,
        "store": ctx.store,
        "secondary_client": ctx.secondary_client,
        "secondary_model": ctx.secondary_model,
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
