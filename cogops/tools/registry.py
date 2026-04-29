"""
cogops/tools/registry.py

Build the full tool registry: tools_schema + name-to-callback map.

Tools fall into two groups:
- pure tools whose only inputs come from the model (handled as-is)
- context-dependent tools whose handler needs server-side state
  (user_id, Redis store, secondary LLM client/model). The model-visible
  JSON schema never exposes those;
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

    # --- Knowledge search tool ---
    from cogops.tools.knowledge.search_knowledge import (
        search_knowledge_tools_list as k1,
        search_knowledge_tools_map as k2,
    )
    all_schema.extend(k1)
    all_map.update(k2)

    
    # --- Wiki search tool ---
    from cogops.tools.search_wiki import (
        search_wiki_tools_list as w1,
        search_wiki_tools_map as w2,
    )
    all_schema.extend(w1)
    all_map.update(w2)

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
